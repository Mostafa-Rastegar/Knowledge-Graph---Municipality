"""Offline self-check for ontology enforcement (no LLM, no DB needed).

Run: python -m tests.test_extract
"""
from src.extract import parse_triplets, fact_id, entity_key, Entity, Triplet


def test_keeps_valid_allowed_relation():
    content = '{"triplets":[{"subject":{"type":"Contractor","name":"شرکت آلفا"},"predicate":"EXECUTOR_OF","object":{"type":"Project","name":"پل صدر"},"evidence":"شرکت آلفا مجری پروژه پل صدر است."}]}'
    out = parse_triplets(content)
    assert len(out) == 1
    assert out[0].subject.type == "Contractor"


def test_drops_unknown_entity_type():
    content = '{"triplets":[{"subject":{"type":"Vehicle","name":"x"},"predicate":"EXECUTOR_OF","object":{"type":"Project","name":"y"},"evidence":"z"}]}'
    assert parse_triplets(content) == []


def test_drops_disallowed_relation():
    # Project -EXECUTOR_OF-> Contractor is not in ALLOWED_RELATIONS
    content = '{"triplets":[{"subject":{"type":"Project","name":"x"},"predicate":"EXECUTOR_OF","object":{"type":"Contractor","name":"y"},"evidence":"z"}]}'
    assert parse_triplets(content) == []


def test_drops_missing_evidence():
    content = '{"triplets":[{"subject":{"type":"Budget","name":"x"},"predicate":"FINANCES","object":{"type":"Project","name":"y"},"evidence":"  "}]}'
    assert parse_triplets(content) == []


def test_bad_json_returns_empty():
    assert parse_triplets("not json") == []


def test_fact_id_stable_and_key_format():
    t = Triplet(
        subject=Entity(type="Budget", name=" ردیف  ۱۴۰۳ "),
        predicate="FINANCES",
        object=Entity(type="Project", name="پل صدر"),
        evidence="...",
    )
    # whitespace-collapsed, deterministic
    assert entity_key(t.subject) == "Budget:ردیف ۱۴۰۳"
    assert fact_id("chunk_x", t) == fact_id("chunk_x", t)
    assert fact_id("chunk_x", t) != fact_id("chunk_y", t)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
