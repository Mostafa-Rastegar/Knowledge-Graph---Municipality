"""Offline self-check for ontology enforcement (no LLM, no DB needed).

Run: python -m tests.test_extract
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.extract import parse_triplets, fact_id, entity_key, Entity, Triplet, chunk_sha


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


def _record(chunk_id: str, text: str, name: str) -> dict:
    return {
        "fact_id": f"fact_{name}",
        "chunk_id": chunk_id,
        "chunk_sha": chunk_sha(text),
        "document_id": chunk_id.rsplit("_", 1)[0],
        "source_path": "test",
        "subject": {"type": "Contractor", "name": name, "key": f"Contractor:{name}"},
        "predicate": "EXECUTOR_OF",
        "object": {"type": "Project", "name": "پل صدر", "key": "Project:پل صدر"},
        "evidence": "متن شاهد",
    }


def test_resume_keeps_the_work_of_a_run_that_died():
    """A stopped run must not lose finished chunks.

    The extractor writes to a .part file and moves it into place at the end. A
    later run reads both the output file and the .part file, so the work of
    every attempt adds up. The api key here is invalid on purpose: a cache miss
    would try to call the model and the run would fail.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        chunks = work / "chunks.jsonl"
        texts = {f"doc_{i}_c0": f"sentence number {i}" for i in range(3)}
        chunks.write_text(
            "\n".join(
                json.dumps({"chunk_id": cid, "document_id": cid.rsplit("_", 1)[0],
                            "source_path": "test", "text": text}, ensure_ascii=False)
                for cid, text in texts.items()
            ),
            encoding="utf-8",
        )
        out = work / "triplets.jsonl"
        # The output of an early run holds one chunk.
        out.write_text(json.dumps(_record("doc_0_c0", texts["doc_0_c0"], "alpha"),
                                  ensure_ascii=False) + "\n", encoding="utf-8")
        # A later run died, so its progress sits in the .part file.
        part = out.with_suffix(out.suffix + ".part")
        part.write_text(
            "\n".join(
                json.dumps(_record(cid, text, f"name{i}"), ensure_ascii=False)
                for i, (cid, text) in enumerate(texts.items())
            ) + "\n",
            encoding="utf-8",
        )

        env = {**__import__("os").environ, "LLM_API_KEY": "invalid-on-purpose",
               "LLM_BASE_URL": "http://127.0.0.1:1", "PYTHONUTF8": "1"}
        run = subprocess.run(
            [sys.executable, "-m", "src.extract", str(chunks), "--out", str(out)],
            capture_output=True, text=True, env=env, timeout=120,
        )
        assert run.returncode == 0, run.stderr
        log = run.stdout + run.stderr
        assert "resumed_from_partial_run" in log, log
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 3, lines
        assert not part.exists(), "the .part file must move into place"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
