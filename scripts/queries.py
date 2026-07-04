"""Sample KG queries + counts. Run: python -m scripts.queries"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
drv = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)
db = os.environ.get("NEO4J_DATABASE") or None


def one(session, q, **p):
    return session.run(q, **p).single()[0]


def rows(session, q, **p):
    return [r.values() for r in session.run(q, **p)]


with drv.session(database=db) as s:
    print("Project nodes     :", one(s, "MATCH (n:Project) RETURN count(n)"))
    print("Contractor nodes  :", one(s, "MATCH (n:Contractor) RETURN count(n)"))
    print("Complaint nodes   :", one(s, "MATCH (n:Complaint) RETURN count(n)"))
    print("Total relationships:", one(
        s,
        "MATCH ()-[r]->() WHERE type(r) <> 'HAS_EVIDENCE' RETURN count(r)",
    ))
    print("\nContractor of «پروژه بهسازی پل صدر»:")
    for r in rows(
        s,
        "MATCH (c:Contractor)-[:EXECUTOR_OF]->(p:Project) "
        "WHERE p.name CONTAINS 'پل صدر' RETURN c.name, p.name",
    ):
        print("  ", r)
    print("\nComplaints about «محله پونک»:")
    for r in rows(
        s,
        "MATCH (c:Complaint)-[:COMPLAINS_ABOUT]->(l:Location) "
        "WHERE l.name CONTAINS 'پونک' RETURN c.name, l.name",
    ):
        print("  ", r)

drv.close()
