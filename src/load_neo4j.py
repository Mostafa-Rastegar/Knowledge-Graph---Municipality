"""Ticket 3: load extracted triplets into the Neo4j knowledge graph.

CLI:
  python -m src.load_neo4j data/extracted/triplets.jsonl

Reads validated triplet records, creates constraints/indexes once, then MERGEs
entities by their stable `key` and MERGEs the allowed directed relationship
between them. Evidence text for each fact is stored as an Evidence node linked
to the relationship's endpoints.

Rerun-safe: everything is MERGE, so rerunning the same triplets creates no
duplicate nodes or edges; only new facts add to the graph.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import structlog
from dotenv import load_dotenv
from neo4j import GraphDatabase

log = structlog.get_logger()
load_dotenv()

ENTITY_LABELS = ["Project", "Contractor", "Location", "Official", "Budget", "Complaint"]

# Cypher per allowed (subject_type, predicate, object_type). Labels are fixed
# from the ontology (not user input), so they are safe to inline.
RELATION_CYPHER = {
    ("Contractor", "EXECUTOR_OF", "Project"):
        "MATCH (s:Contractor {key:$sk}), (o:Project {key:$ok}) MERGE (s)-[:EXECUTOR_OF]->(o)",
    ("Project", "LOCATED_IN", "Location"):
        "MATCH (s:Project {key:$sk}), (o:Location {key:$ok}) MERGE (s)-[:LOCATED_IN]->(o)",
    ("Official", "SUPERVISOR_OF", "Project"):
        "MATCH (s:Official {key:$sk}), (o:Project {key:$ok}) MERGE (s)-[:SUPERVISOR_OF]->(o)",
    ("Budget", "FINANCES", "Project"):
        "MATCH (s:Budget {key:$sk}), (o:Project {key:$ok}) MERGE (s)-[:FINANCES]->(o)",
    ("Complaint", "COMPLAINS_ABOUT", "Project"):
        "MATCH (s:Complaint {key:$sk}), (o:Project {key:$ok}) MERGE (s)-[:COMPLAINS_ABOUT]->(o)",
    ("Complaint", "COMPLAINS_ABOUT", "Location"):
        "MATCH (s:Complaint {key:$sk}), (o:Location {key:$ok}) MERGE (s)-[:COMPLAINS_ABOUT]->(o)",
}


def ensure_schema(session) -> None:
    for label in ENTITY_LABELS:
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.key IS UNIQUE"
        )
    session.run(
        "CREATE CONSTRAINT evidence_id IF NOT EXISTS "
        "FOR (e:Evidence) REQUIRE e.fact_id IS UNIQUE"
    )


def merge_entity(session, entity: dict) -> None:
    label = entity["type"]
    session.run(
        f"MERGE (n:{label} {{key:$key}}) SET n.name=$name",
        key=entity["key"], name=entity["name"],
    )


def load_record(session, rec: dict) -> bool:
    key = (rec["subject"]["type"], rec["predicate"], rec["object"]["type"])
    cypher = RELATION_CYPHER.get(key)
    if cypher is None:
        log.warning("skip_unknown_relation", relation=key, fact_id=rec.get("fact_id"))
        return False
    merge_entity(session, rec["subject"])
    merge_entity(session, rec["object"])
    session.run(cypher, sk=rec["subject"]["key"], ok=rec["object"]["key"])
    # Evidence node tied to both endpoints, idempotent on fact_id.
    session.run(
        """
        MATCH (s {key:$sk}), (o {key:$ok})
        MERGE (e:Evidence {fact_id:$fid})
        SET e.text=$text, e.chunk_id=$chunk_id, e.source_path=$source_path, e.predicate=$predicate
        MERGE (s)-[:HAS_EVIDENCE]->(e)
        MERGE (o)-[:HAS_EVIDENCE]->(e)
        """,
        sk=rec["subject"]["key"], ok=rec["object"]["key"], fid=rec["fact_id"],
        text=rec["evidence"], chunk_id=rec["chunk_id"],
        source_path=rec["source_path"], predicate=rec["predicate"],
    )
    return True


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
    args = parser.parse_args()

    path = Path(args.triplets)
    if not path.exists():
        parser.error(f"triplets file not found: {path}")

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    database = os.environ.get("NEO4J_DATABASE", "").strip() or None
    driver = driver_from_env()
    loaded = 0
    try:
        with driver.session(database=database) as session:
            ensure_schema(session)
            for rec in records:
                if load_record(session, rec):
                    loaded += 1
    finally:
        driver.close()

    log.info("done", facts_loaded=loaded, total_records=len(records), source=str(path))


if __name__ == "__main__":
    main()
