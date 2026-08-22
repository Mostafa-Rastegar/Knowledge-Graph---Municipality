"""Offline checks for the Re-DocRED harness. No LLM call, no network.

Run: python -m tests.test_redocred
"""
import argparse
import collections
import json
from pathlib import Path

from src import redocred

GOLD = Path("data/benchmark/gold_dev.jsonl")
TMP = Path("data/benchmark/_test_pred.jsonl")
REPORT = Path("data/benchmark/_test_report.json")


def gold_docs():
    return [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_pred(rows):
    TMP.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def pred_row(doc, tr, subject_name=None, object_name=None):
    ents = {e["idx"]: e for e in doc["entities"]}
    return {
        "document_id": doc["document_id"],
        "subject": {"type": ents[tr["h"]]["type"], "name": subject_name or ents[tr["h"]]["forms"][0]},
        "predicate": tr["r_name"],
        "object": {"type": ents[tr["t"]]["type"], "name": object_name or ents[tr["t"]]["forms"][0]},
        "evidence": "x",
    }


def run_eval(report=REPORT):
    redocred.cmd_eval(argparse.Namespace(pred=str(TMP), gold=str(GOLD),
                                         report=str(report), strict_types=False))
    return json.loads(report.read_text(encoding="utf-8"))


def test_oracle_hits_the_matching_ceiling():
    """An oracle that returns every gold fact by name must score ~100.

    It cannot reach exactly 100: a few documents hold two different entities
    with the same surface form and the same type, so a name-based answer is
    ambiguous. That gap is the ceiling of any name-based system on this data,
    and the evaluation report states it. The ceiling drops a little as the
    corpus grows, because a bigger corpus holds more colliding names, so the
    test checks the ceiling instead of a fixed number.
    """
    docs = gold_docs()
    write_pred([pred_row(d, tr) for d in docs for tr in d["triples"]])
    rep = run_eval()
    assert rep["precision"] >= 99.9, rep
    assert rep["f1"] >= 99.5, rep
    assert rep["entity_unmatched"] == 0, rep
    print(f"ok test_oracle_hits_the_matching_ceiling "
          f"({len(docs)} docs, ceiling F1={rep['f1']})")


def test_same_name_same_type_entities_are_the_ceiling():
    """Name-based matching cannot separate two entities that look identical.

    This is the reason the oracle scores 99.76 and not 100. The check builds
    the collision directly instead of hoping the corpus contains one.
    """
    doc = {
        "document_id": "d0",
        "title": "t",
        "entities": [
            {"idx": 0, "type": "LOC", "forms": ["Springfield"]},
            {"idx": 1, "type": "LOC", "forms": ["Springfield"]},
            {"idx": 2, "type": "LOC", "forms": ["Illinois"]},
        ],
        "triples": [{"h": 1, "t": 2, "r": "P17", "r_name": "country"}],
    }
    gold = Path("data/benchmark/_test_collision_gold.jsonl")
    gold.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    write_pred([pred_row(doc, doc["triples"][0])])
    redocred.cmd_eval(argparse.Namespace(pred=str(TMP), gold=str(gold),
                                         report=str(REPORT), strict_types=False))
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    # The prediction names entity 1, but the lookup answers with entity 0.
    assert rep["entity_unmatched"] == 0, rep
    assert rep["recall"] == 0.0, rep
    gold.unlink()
    print("ok test_same_name_same_type_entities_are_the_ceiling")


def test_half_recall():
    docs = gold_docs()
    rows = [pred_row(d, tr) for d in docs for tr in d["triples"]]
    write_pred(rows[: len(rows) // 2])
    rep = run_eval()
    assert rep["precision"] >= 99.9, rep
    assert 40 < rep["recall"] < 60, rep
    print("ok test_half_recall")


def test_unmatched_entity_is_dropped():
    doc = gold_docs()[0]
    write_pred([pred_row(doc, doc["triples"][0], subject_name="no such entity at all")])
    rep = run_eval()
    assert rep["entity_unmatched"] == 1, rep
    assert rep["predicted_triples"] == 0, rep
    print("ok test_unmatched_entity_is_dropped")


def test_surface_form_variant_still_matches():
    """A different mention of the same entity must map to the same gold index."""
    doc = next(d for d in gold_docs() if any(len(e["forms"]) > 1 for e in d["entities"]))
    ents = {e["idx"]: e for e in doc["entities"]}
    tr = next(t for t in doc["triples"] if len(ents[t["h"]]["forms"]) > 1)
    write_pred([pred_row(doc, tr, subject_name=ents[tr["h"]]["forms"][1].upper() + " .")])
    rep = run_eval()
    assert rep["entity_unmatched"] == 0, rep
    assert rep["predicted_triples"] == 1, rep
    print("ok test_surface_form_variant_still_matches")


def test_ign_drops_train_facts():
    """A fact already present in train must leave BOTH pools, not one.

    Dropping it from gold only would punish a correct prediction; dropping it
    from predictions only would reward a miss. The check below takes a real
    train fact and an unseen one, and predicts both.
    """
    docs = gold_docs()
    train_keys = set(json.loads((redocred.OUT / "train_fact_keys.json").read_text(encoding="utf-8")))

    def key_of(doc, tr):
        forms = {e["idx"]: e["forms"] for e in doc["entities"]}
        return [
            f"{redocred.norm(h)}|{tr['r_name']}|{redocred.norm(t)}"
            for h in forms[tr["h"]] for t in forms[tr["t"]]
        ]

    seen = unseen = None
    for d in docs:
        for tr in d["triples"]:
            hit = any(k in train_keys for k in key_of(d, tr))
            if hit and seen is None:
                seen = (d, tr)
            elif not hit and unseen is None:
                unseen = (d, tr)
        if seen and unseen:
            break
    assert seen and unseen, "need one train-seen and one unseen gold fact"

    write_pred([pred_row(seen[0], seen[1]), pred_row(unseen[0], unseen[1])])
    rep = run_eval()
    assert rep["predicted_triples"] == 2, rep
    # The train fact leaves BOTH pools. So the prediction pool loses exactly one
    # entry, and precision stays 100 instead of counting the removal as a miss.
    assert rep["predicted_triples_unseen"] == 1, rep
    assert rep["gold_triples_unseen"] < rep["gold_triples"], rep
    assert rep["ign_precision"] == 100.0, rep
    print("ok test_ign_drops_train_facts (train fact removed from both pools)")


def test_closure_rules():
    """R1 inverse, R2 transitive and R3 chain must each fire exactly once."""
    src = Path("data/benchmark/_test_closure_in.jsonl")
    dst = Path("data/benchmark/_test_closure_out.jsonl")
    rows = [
        {
            "document_id": "d0",
            "subject": {"type": "LOC", "name": "Sadr Bridge"},
            "predicate": redocred.LOCATED_IN,
            "object": {"type": "LOC", "name": "Tehran"},
            "evidence": "The bridge is in Tehran.",
        },
        {
            "document_id": "d0",
            "subject": {"type": "LOC", "name": "Tehran"},
            "predicate": redocred.LOCATED_IN,
            "object": {"type": "LOC", "name": "Tehran Province"},
            "evidence": "Tehran lies in Tehran Province.",
        },
        {
            "document_id": "d0",
            "subject": {"type": "LOC", "name": "Tehran"},
            "predicate": redocred.COUNTRY,
            "object": {"type": "LOC", "name": "Iran"},
            "evidence": "Tehran is in Iran.",
        },
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    redocred.cmd_closure(argparse.Namespace(pred=str(src), out=str(dst)))
    got = {
        (r["subject"]["name"], r["predicate"], r["object"]["name"])
        for r in (json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines() if l.strip())
    }
    assert ("Tehran", redocred.CONTAINS, "Sadr Bridge") in got, "R1 inverse failed"
    assert ("Sadr Bridge", redocred.LOCATED_IN, "Tehran Province") in got, "R2 transitive failed"
    assert ("Sadr Bridge", redocred.COUNTRY, "Iran") in got, "R3 chain failed"
    derived = [
        r
        for r in (json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines() if l.strip())
        if r.get("derived_by")
    ]
    assert all(r["evidence"] for r in derived), "every derived fact needs evidence"
    # The counts matter, not only the presence. A rule that fires twice on this
    # input would build a fact the three premises do not support.
    fired = collections.Counter(r["derived_by"] for r in derived)
    # Two of the three premise facts use LOCATED_IN, which has an inverse.
    # The third uses COUNTRY, which has none, so R1 fires twice, not three times.
    assert fired["R1_inverse"] == 2, fired
    assert fired["R2_transitive"] == 1, fired
    assert fired["R3_chain_country"] == 1, fired
    src.unlink()
    dst.unlink()
    print(f"ok test_closure_rules ({len(derived)} derived facts, all with evidence)")


if __name__ == "__main__":
    for fn in [
        test_oracle_hits_the_matching_ceiling,
        test_same_name_same_type_entities_are_the_ceiling,
        test_half_recall,
        test_unmatched_entity_is_dropped,
        test_surface_form_variant_still_matches,
        test_ign_drops_train_facts,
        test_closure_rules,
    ]:
        fn()
    TMP.unlink(missing_ok=True)
    REPORT.unlink(missing_ok=True)
    print("all passed")
