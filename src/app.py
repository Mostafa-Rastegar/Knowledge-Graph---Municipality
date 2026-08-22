r"""Frontend API: upload a document, run the pipeline, view the knowledge graph.

Run:
  .\.venv\Scripts\python.exe -m uvicorn src.app:app --reload
Then open http://localhost:8000

Two things only, per spec:
  - POST /api/upload : save file to data/raw, run ingest->extract->load, return counts
  - GET  /api/graph  : return entities + relationships for the graph view
"""
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
    """Run a pipeline module with the same interpreter; raise on failure."""
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

    # Full rebuild over data/raw — MERGE keeps it duplicate-free.
    _run("src.ingest", "data/raw")
    _run("src.extract", "data/processed/chunks.jsonl")
    _run("src.load_neo4j", "data/extracted/triplets.jsonl")
    log.info("uploaded_and_loaded", file=name)
    return JSONResponse({"ok": True, "file": name})


@app.get("/api/graph")
def graph(doc: str = "") -> dict:
    """The whole graph, or one document's graph when `doc` is given.

    The database holds the municipality graph and the English benchmark graph
    side by side. Drawing every node at once is unreadable, so `doc` limits the
    answer to the facts whose evidence came from that document.
    """
    driver = _driver()
    db = os.environ.get("NEO4J_DATABASE", "").strip() or None
    nodes, edges = [], []
    try:
        with driver.session(database=db) as s:
            node_query = (
                "MATCH (n)-[:HAS_EVIDENCE]->(e:Evidence) "
                "WHERE NOT n:Evidence AND e.chunk_id STARTS WITH $doc "
                "RETURN DISTINCT n.key AS id, n.name AS name, labels(n)[0] AS type"
            ) if doc else (
                "MATCH (n) WHERE NOT n:Evidence "
                "RETURN n.key AS id, n.name AS name, labels(n)[0] AS type"
            )
            for r in s.run(node_query, doc=doc):
                nodes.append({
                    "id": r["id"], "label": r["name"],
                    "group": r["type"], "groupFa": GROUP_FA.get(r["type"], r["type"]),
                })
            edge_query = (
                "MATCH (a)-[rel]->(b) WHERE type(rel) <> 'HAS_EVIDENCE' "
                "MATCH (a)-[:HAS_EVIDENCE]->(e:Evidence)<-[:HAS_EVIDENCE]-(b) "
                "WHERE e.chunk_id STARTS WITH $doc "
                "RETURN a.key AS src, type(rel) AS type, b.key AS dst, "
                "collect(e.text)[0] AS evidence"
            ) if doc else (
                "MATCH (a)-[rel]->(b) WHERE type(rel) <> 'HAS_EVIDENCE' "
                "OPTIONAL MATCH (a)-[:HAS_EVIDENCE]->(e:Evidence)<-[:HAS_EVIDENCE]-(b) "
                "RETURN a.key AS src, type(rel) AS type, b.key AS dst, "
                "collect(e.text)[0] AS evidence"
            )
            for r in s.run(edge_query, doc=doc):
                edges.append({
                    "from": r["src"], "to": r["dst"],
                    "label": r["type"], "evidence": r["evidence"] or "",
                })
    finally:
        driver.close()
    return {"nodes": nodes, "edges": edges}
