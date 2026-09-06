from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
from neo4j import GraphDatabase

log = structlog.get_logger()
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WEB_DIR = ROOT / "web"
SUPPORTED = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}
TYPE_EXPR = "[l IN labels(n) WHERE l <> 'Entity'][0]"

app = FastAPI(title="Municipality Knowledge Graph")

GROUP_FA = {
    "Project": "پروژه",
    "Contractor": "پیمانکار",
    "Location": "مکان",
    "Official": "مسئول",
    "Budget": "بودجه",
    "Complaint": "شکایت",
}


def _driver():
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USER", "").strip()
    pw = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not (uri and user and pw):
        raise HTTPException(503, "Neo4j credentials missing in .env")
    return GraphDatabase.driver(uri, auth=(user, pw))


def _run(step: str, *args: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", step, *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("pipeline_step_failed", step=step, stderr=proc.stderr[-2000:])
        raise HTTPException(500, f"{step} failed: {proc.stderr[-500:]}")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/upload")
async def upload(file: UploadFile) -> JSONResponse:
    name = Path(file.filename or "").name
    if not name or Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, "فقط pdf / docx / txt / png / jpg پشتیبانی می‌شود")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / name
    dest.write_bytes(await file.read())

    _run("src.ingest", "data/raw")
    _run("src.extract", "data/processed/chunks.jsonl")
    _run("src.load_neo4j", "data/extracted/triplets.jsonl")
    log.info("uploaded_and_loaded", file=name)
    return JSONResponse({"ok": True, "file": name})


def _node(r) -> dict:
    return {"id": r["id"], "label": r["name"], "group": r["type"], "groupFa": GROUP_FA.get(r["type"], r["type"])}


@app.get("/api/stats")
def stats() -> dict:
    driver = _driver()
    db = os.environ.get("NEO4J_DATABASE", "").strip() or None
    try:
        with driver.session(database=db) as s:
            labels = {
                r["label"]: r["n"]
                for r in s.run(
                    "MATCH (n) WHERE NOT n:Evidence "
                    f"WITH {TYPE_EXPR} AS label RETURN label, count(*) AS n ORDER BY n DESC"
                )
            }
            rels = {
                r["type"]: r["n"]
                for r in s.run(
                    "MATCH ()-[r]->() WHERE type(r) <> 'HAS_EVIDENCE' "
                    "RETURN type(r) AS type, count(*) AS n ORDER BY n DESC"
                )
            }
            evidence = s.run("MATCH (e:Evidence) RETURN count(e) AS n").single()["n"]
    finally:
        driver.close()
    return {
        "entities": sum(labels.values()), "facts": sum(rels.values()), "evidence": evidence,
        "by_type": labels, "by_relation": rels,
    }


@app.get("/api/graph")
def graph(doc: str = "", q: str = "", limit: int = 300) -> dict:
    driver = _driver()
    db = os.environ.get("NEO4J_DATABASE", "").strip() or None
    limit = max(1, min(limit, 3000))
    nodes, edges, seen = [], [], set()
    try:
        with driver.session(database=db) as s:
            if doc:
                edge_query = (
                    "MATCH (e:Evidence) WHERE e.chunk_id STARTS WITH $doc "
                    "MATCH (a)-[:HAS_EVIDENCE]->(e)<-[:HAS_EVIDENCE]-(b) "
                    "MATCH (a)-[rel]->(b) WHERE type(rel) <> 'HAS_EVIDENCE' "
                    "RETURN DISTINCT a.key AS src, type(rel) AS type, b.key AS dst, e.text AS evidence LIMIT $limit"
                )
            elif q:
                edge_query = (
                    "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($q) "
                    "WITH n ORDER BY size(n.name) LIMIT 20 "
                    "MATCH (n)-[rel]-(m:Entity) WHERE type(rel) <> 'HAS_EVIDENCE' "
                    "WITH startNode(rel) AS a, endNode(rel) AS b, rel "
                    "OPTIONAL MATCH (a)-[:HAS_EVIDENCE]->(e:Evidence)<-[:HAS_EVIDENCE]-(b) "
                    "RETURN DISTINCT a.key AS src, type(rel) AS type, b.key AS dst, collect(e.text)[0] AS evidence LIMIT $limit"
                )
            else:
                edge_query = (
                    "MATCH (n:Entity) WITH n, COUNT { (n)--() } AS degree ORDER BY degree DESC LIMIT 5 "
                    "MATCH (n)-[rel]-(m:Entity) WHERE type(rel) <> 'HAS_EVIDENCE' "
                    "WITH startNode(rel) AS a, endNode(rel) AS b, rel "
                    "OPTIONAL MATCH (a)-[:HAS_EVIDENCE]->(e:Evidence)<-[:HAS_EVIDENCE]-(b) "
                    "RETURN DISTINCT a.key AS src, type(rel) AS type, b.key AS dst, collect(e.text)[0] AS evidence LIMIT $limit"
                )
            for r in s.run(edge_query, doc=doc, q=q, limit=limit):
                edges.append({"from": r["src"], "to": r["dst"], "label": r["type"], "evidence": r["evidence"] or ""})
                seen.add(r["src"])
                seen.add(r["dst"])
            if seen:
                node_query = (
                    "MATCH (n:Entity) WHERE n.key IN $keys "
                    f"RETURN n.key AS id, n.name AS name, {TYPE_EXPR} AS type"
                )
                nodes = [_node(r) for r in s.run(node_query, keys=list(seen))]
    finally:
        driver.close()
    return {"nodes": nodes, "edges": edges}
