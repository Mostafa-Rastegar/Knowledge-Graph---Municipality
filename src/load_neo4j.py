from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv
from neo4j import GraphDatabase

log = structlog.get_logger()
load_dotenv()

ENTITY_LABELS = ["Project", "Contractor", "Location", "Official", "Budget", "Complaint"]

ALLOWED = {
    ("Contractor", "EXECUTOR_OF", "Project"),
    ("Project", "LOCATED_IN", "Location"),
    ("Official", "SUPERVISOR_OF", "Project"),
    ("Budget", "FINANCES", "Project"),
    ("Complaint", "COMPLAINS_ABOUT", "Project"),
    ("Complaint", "COMPLAINS_ABOUT", "Location"),
}


def rel_type(relation: str) -> str:
    return re.sub(r"\W+", "_", relation).strip("_").upper()


def load_ontology(path: Path) -> None:
    global ENTITY_LABELS, ALLOWED
    spec = json.loads(path.read_text(encoding="utf-8"))
    ENTITY_LABELS = sorted(spec["entity_types"])
    ALLOWED = {tuple(item) for item in spec["allowed"]}


def ensure_schema(session) -> None:
    for label in ENTITY_LABELS:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.key IS UNIQUE"
        )
    session.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE")
    session.run("CREATE CONSTRAINT evidence_id IF NOT EXISTS FOR (e:Evidence) REQUIRE e.fact_id IS UNIQUE")
    session.run("CREATE INDEX evidence_chunk IF NOT EXISTS FOR (e:Evidence) ON (e.chunk_id)")
    session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
    session.run("MATCH (n) WHERE NOT n:Evidence AND NOT n:Entity SET n:Entity")


def load_batch(tx, records: list[dict]) -> int:
    entities: dict[str, dict[str, str]] = {}
    relations: dict[str, list[dict]] = {}
    evidence: list[dict] = []
    for rec in records:
        s, o = rec["subject"], rec["object"]
        entities.setdefault(s["type"], {})[s["key"]] = s["name"]
        entities.setdefault(o["type"], {})[o["key"]] = o["name"]
        relations.setdefault(rel_type(rec["predicate"]), []).append({"sk": s["key"], "ok": o["key"]})
        evidence.append({
            "sk": s["key"], "ok": o["key"], "fid": rec["fact_id"], "text": rec["evidence"],
            "chunk_id": rec["chunk_id"], "source_path": rec["source_path"], "predicate": rec["predicate"],
        })
    for label, rows in entities.items():
        tx.run(
            f"UNWIND $rows AS r MERGE (n:Entity {{key:r.key}}) SET n.name=r.name, n:{label}",
            rows=[{"key": k, "name": v} for k, v in rows.items()],
        )
    for rtype, rows in relations.items():
        tx.run(
            f"UNWIND $rows AS r MATCH (s:Entity {{key:r.sk}}), (o:Entity {{key:r.ok}}) "
            f"MERGE (s)-[:{rtype}]->(o)",
            rows=rows,
        )
    tx.run(
        "UNWIND $rows AS r "
        "MATCH (s:Entity {key:r.sk}), (o:Entity {key:r.ok}) "
        "MERGE (e:Evidence {fact_id:r.fid}) "
        "SET e.text=r.text, e.chunk_id=r.chunk_id, e.source_path=r.source_path, e.predicate=r.predicate "
        "MERGE (s)-[:HAS_EVIDENCE]->(e) MERGE (o)-[:HAS_EVIDENCE]->(e)",
        rows=evidence,
    )
    return len(records)


def iter_records(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def driver_from_env():
    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USER", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "").strip()
    if not uri or not user or not password:
        raise SystemExit("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD missing. Set them in .env.")
    return GraphDatabase.driver(uri, auth=(user, password))


def main() -> None:
    parser = argparse.ArgumentParser(description="Load triplets into Neo4j")
    parser.add_argument("triplets", nargs="?", default="data/extracted/triplets.jsonl")
    parser.add_argument("--ontology", help="JSON ontology file; default is the municipality one")
    parser.add_argument("--batch", type=int, default=int(os.environ.get("NEO4J_BATCH", "2000")))
    args = parser.parse_args()

    if args.ontology:
        load_ontology(Path(args.ontology))
        log.info("ontology_loaded", path=args.ontology, labels=len(ENTITY_LABELS), relations=len(ALLOWED))

    path = Path(args.triplets)
    if not path.exists():
        parser.error(f"triplets file not found: {path}")

    database = os.environ.get("NEO4J_DATABASE", "").strip() or None
    driver = driver_from_env()
    loaded = skipped = total = 0
    started = time.time()
    batch: list[dict] = []
    try:
        with driver.session(database=database) as session:
            ensure_schema(session)
            for rec in iter_records(path):
                total += 1
                if (rec["subject"]["type"], rec["predicate"], rec["object"]["type"]) not in ALLOWED:
                    skipped += 1
                    continue
                batch.append(rec)
                if len(batch) >= args.batch:
                    loaded += session.execute_write(load_batch, batch)
                    batch = []
                    if loaded % (args.batch * 25) == 0:
                        rate = loaded / (time.time() - started)
                        log.info("progress", loaded=loaded, skipped=skipped, per_sec=round(rate, 1))
            if batch:
                loaded += session.execute_write(load_batch, batch)
    finally:
        driver.close()

    log.info("done", facts_loaded=loaded, skipped=skipped, total_records=total,
             seconds=round(time.time() - started, 1), source=str(path))


if __name__ == "__main__":
    main()
