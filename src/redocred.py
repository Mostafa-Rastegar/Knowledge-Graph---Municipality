"""Phase 2: run the municipality KG pipeline on the Re-DocRED benchmark.

CLI:
  python -m src.redocred prepare --limit 50
  python -m src.redocred eval data/benchmark/triplets_dev.jsonl

`prepare` turns Re-DocRED documents into the same chunk format that
`src.extract` already reads, and derives the ontology (entity types, 96
relations, allowed type-relation-type combinations) from the train split.

`eval` maps the extracted triplets back onto the gold entity indices and
reports Precision / Recall / F1 and Ign F1 (the DocRED metric that drops
facts already seen in the train split, so memorisation earns no score).

Task setup: we follow the standard document-level relation extraction setup.
The entity list of each document is given to the model; the model predicts
which relations hold between them and quotes the evidence sentence.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

BENCH = Path("data/benchmark/re-docred")
OUT = Path("data/benchmark")
ONTOLOGY = Path("configs/ontology_redocred.json")


def load_split(name: str) -> list[dict]:
    return json.loads((BENCH / "data" / f"{name}_revised.json").read_text(encoding="utf-8"))


def rel_names() -> dict[str, str]:
    return json.loads((BENCH / "rel_info.json").read_text(encoding="utf-8"))


def norm(text: str) -> str:
    """Loose surface-form key: case, punctuation and spacing do not matter."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def doc_text(doc: dict) -> str:
    return " ".join(" ".join(sent) for sent in doc["sents"])


def entity_block(doc: dict) -> str:
    """The entity list handed to the model, one line per entity."""
    lines = []
    for ent in doc["vertexSet"]:
        forms = sorted({m["name"] for m in ent})
        lines.append(f"- {forms[0]} [{ent[0]['type']}]" + (f" (also: {', '.join(forms[1:])})" if len(forms) > 1 else ""))
    return "\n".join(lines)


def build_ontology(train: list[dict], names: dict[str, str]) -> dict:
    """Allowed (subject_type, relation, object_type) triples, learned from train.

    This is the same strict-ontology filter the municipality pipeline uses; here
    the ontology is derived from data instead of written by hand.
    """
    allowed = set()
    for doc in train:
        for lab in doc["labels"]:
            allowed.add(
                (
                    doc["vertexSet"][lab["h"]][0]["type"],
                    names[lab["r"]],
                    doc["vertexSet"][lab["t"]][0]["type"],
                )
            )
    return {
        "name": "re-docred",
        "language": "en",
        "entity_types": sorted({t for t, _, _ in allowed} | {t for _, _, t in allowed}),
        "relations": names,
        "allowed": sorted(list(t) for t in allowed),
    }


def train_fact_keys(train: list[dict], names: dict[str, str]) -> set[str]:
    """Surface-form facts seen in train, used by Ign F1."""
    keys = set()
    for doc in train:
        for lab in doc["labels"]:
            h = norm(doc["vertexSet"][lab["h"]][0]["name"])
            t = norm(doc["vertexSet"][lab["t"]][0]["name"])
            keys.add(f"{h}|{names[lab['r']]}|{t}")
    return keys


def cmd_prepare(args: argparse.Namespace) -> None:
    names = rel_names()
    train = load_split("train")
    dev = load_split(args.split)[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    ONTOLOGY.parent.mkdir(parents=True, exist_ok=True)
    ONTOLOGY.write_text(
        json.dumps(build_ontology(train, names), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chunks_path = OUT / f"chunks_{args.split}.jsonl"
    gold_path = OUT / f"gold_{args.split}.jsonl"
    with chunks_path.open("w", encoding="utf-8") as ch, gold_path.open("w", encoding="utf-8") as gd:
        for i, doc in enumerate(dev):
            doc_id = f"redocred_{args.split}_{i:04d}"
            ch.write(
                json.dumps(
                    {
                        "chunk_id": f"{doc_id}_c0",
                        "document_id": doc_id,
                        "source_path": str(BENCH / "data" / f"{args.split}_revised.json"),
                        "text": f"Document:\n{doc_text(doc)}\n\nEntities:\n{entity_block(doc)}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            gd.write(
                json.dumps(
                    {
                        "document_id": doc_id,
                        "title": doc["title"],
                        "entities": [
                            {"idx": j, "type": e[0]["type"], "forms": sorted({m["name"] for m in e})}
                            for j, e in enumerate(doc["vertexSet"])
                        ],
                        "triples": [
                            {"h": l["h"], "t": l["t"], "r": l["r"], "r_name": names[l["r"]]}
                            for l in doc["labels"]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    keys_path = OUT / "train_fact_keys.json"
    keys_path.write_text(json.dumps(sorted(train_fact_keys(train, names))), encoding="utf-8")

    print(f"docs={len(dev)} chunks={chunks_path} gold={gold_path}")
    print(f"ontology={ONTOLOGY} relations={len(names)}")
    print(f"train_fact_keys={keys_path}")


LOCATED_IN = "located in the administrative territorial entity"
CONTAINS = "contains administrative territorial entity"
COUNTRY = "country"

# Relations that always hold in both directions, under two different names.
INVERSE = {LOCATED_IN: CONTAINS, CONTAINS: LOCATED_IN, "part of": "has part", "has part": "part of"}


def cmd_closure(args: argparse.Namespace) -> None:
    """Add the facts that follow from the extracted ones by ontology rules.

    The language model reports what a sentence says. A knowledge graph also holds
    what the graph implies. Three rules run here:
      R1  inverse      A located in B            -> B contains A
      R2  transitive   A located in B, B in C    -> A located in C
      R3  chain        A located in B, B country C -> A country C
    Every derived fact keeps the evidence of the facts it came from.
    """
    rows = [
        json.loads(line)
        for line in Path(args.pred).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_doc[row["document_id"]].append(row)

    out: list[dict] = []
    added = collections.Counter()
    for doc_id, doc_rows in by_doc.items():
        facts = {(r["subject"]["name"], r["predicate"], r["object"]["name"]): r for r in doc_rows}
        types = {}
        evidence = {}
        for r in doc_rows:
            types[r["subject"]["name"]] = r["subject"]["type"]
            types[r["object"]["name"]] = r["object"]["type"]
            evidence[(r["subject"]["name"], r["predicate"], r["object"]["name"])] = r["evidence"]

        def add(subject: str, predicate: str, obj: str, why: list[str], rule: str) -> bool:
            key = (subject, predicate, obj)
            if key in facts or subject == obj:
                return False
            facts[key] = {
                "fact_id": "fact_" + hashlib.sha1(
                    f"{doc_id}|{subject}|{predicate}|{obj}".encode("utf-8")
                ).hexdigest()[:12],
                "chunk_id": f"{doc_id}_c0",
                "document_id": doc_id,
                "source_path": rule,
                "subject": {"type": types.get(subject, "MISC"), "name": subject},
                "predicate": predicate,
                "object": {"type": types.get(obj, "MISC"), "name": obj},
                "evidence": " | ".join(w for w in why if w)[:600],
                "derived_by": rule,
            }
            added[rule] += 1
            return True

        # R1 inverse
        for (s, p, o), rec in list(facts.items()):
            if p in INVERSE:
                add(o, INVERSE[p], s, [rec["evidence"]], "R1_inverse")

        # R2 transitive closure of located_in (repeat until nothing new appears)
        for _ in range(4):
            chain = [(s, o) for (s, p, o) in list(facts) if p == LOCATED_IN]
            grew = False
            for a, b in chain:
                for c, d in chain:
                    if b == c:
                        grew |= add(
                            a,
                            LOCATED_IN,
                            d,
                            [evidence.get((a, LOCATED_IN, b), ""), evidence.get((c, LOCATED_IN, d), "")],
                            "R2_transitive",
                        )
            if not grew:
                break

        # R3 located_in + country -> country
        countries = [(s, o) for (s, p, o) in list(facts) if p == COUNTRY]
        for a, b in [(s, o) for (s, p, o) in list(facts) if p == LOCATED_IN]:
            for c, d in countries:
                if b == c:
                    add(a, COUNTRY, d, [evidence.get((a, LOCATED_IN, b), "")], "R3_chain_country")

        out.extend(facts.values())

    Path(args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8"
    )
    print(f"input rows {len(rows)} -> output rows {len(out)}")
    for rule, n in added.most_common():
        print(f"  {rule}: +{n}")
    print("out:", args.out)


def cmd_eval(args: argparse.Namespace) -> None:
    gold_docs = {
        rec["document_id"]: rec
        for rec in (
            json.loads(line)
            for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    train_keys = set(json.loads(Path(OUT / "train_fact_keys.json").read_text(encoding="utf-8")))

    # surface form -> entity index, per document. The type-aware table is tried
    # first because two different entities of a document can share a surface form.
    lookup: dict[str, dict[tuple[str, str], int]] = {}
    lookup_any: dict[str, dict[str, int]] = {}
    for doc_id, rec in gold_docs.items():
        typed: dict[tuple[str, str], int] = {}
        plain: dict[str, int] = {}
        for ent in rec["entities"]:
            for form in ent["forms"]:
                typed.setdefault((norm(form), ent["type"]), ent["idx"])
                plain.setdefault(norm(form), ent["idx"])
        lookup[doc_id] = typed
        lookup_any[doc_id] = plain

    def resolve(doc_id: str, entity: dict) -> int | None:
        key = norm(entity["name"])
        hit = lookup[doc_id].get((key, entity.get("type", "")))
        return hit if hit is not None else lookup_any[doc_id].get(key)

    first_form = {
        doc_id: {e["idx"]: e["forms"][0] for e in rec["entities"]}
        for doc_id, rec in gold_docs.items()
    }

    def seen_in_train(fact: tuple[str, int, str, int]) -> bool:
        doc_id, h, rel, t = fact
        return f"{norm(first_form[doc_id][h])}|{rel}|{norm(first_form[doc_id][t])}" in train_keys

    gold_set: set[tuple[str, int, str, int]] = set()
    for doc_id, rec in gold_docs.items():
        for tr in rec["triples"]:
            gold_set.add((doc_id, tr["h"], tr["r_name"], tr["t"]))

    pred_set: set[tuple[str, int, str, int]] = set()
    unmatched = total_pred = 0
    for line in Path(args.pred).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        doc_id = rec["document_id"]
        if doc_id not in lookup:
            continue
        total_pred += 1
        h = resolve(doc_id, rec["subject"])
        t = resolve(doc_id, rec["object"])
        if h is None or t is None:
            unmatched += 1
            continue
        pred_set.add((doc_id, h, rec["predicate"], t))

    def score(gold: set, pred: set) -> tuple[float, float, float]:
        hit = len(gold & pred)
        p = hit / len(pred) * 100 if pred else 0.0
        r = hit / len(gold) * 100 if gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f1 = score(gold_set, pred_set)
    # Ign F1: drop every fact already present in the train split, on both sides,
    # so a model that memorised the train facts earns nothing for repeating them.
    gold_ign = {k for k in gold_set if not seen_in_train(k)}
    pred_ign = {k for k in pred_set if not seen_in_train(k)}
    ip, ir, if1 = score(gold_ign, pred_ign)

    per_rel = collections.Counter(rel for _, _, rel, _ in gold_set)
    print(f"documents        : {len(gold_docs)}")
    print(f"gold triples     : {len(gold_set)}  (unseen in train: {len(gold_ign)})")
    print(f"predicted rows   : {total_pred}  (entity not matched: {unmatched})")
    print(f"predicted triples: {len(pred_set)}")
    print(f"Precision {p:.2f}  Recall {r:.2f}  F1 {f1:.2f}")
    print(f"Ign Precision {ip:.2f}  Ign Recall {ir:.2f}  Ign F1 {if1:.2f}")
    print(f"distinct gold relations in this subset: {len(per_rel)}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "documents": len(gold_docs),
                    "gold_triples": len(gold_set),
                    "gold_triples_unseen": len(gold_ign),
                    "predicted_rows": total_pred,
                    "predicted_triples": len(pred_set),
                    "entity_unmatched": unmatched,
                    "precision": round(p, 2),
                    "recall": round(r, 2),
                    "f1": round(f1, 2),
                    "ign_precision": round(ip, 2),
                    "ign_recall": round(ir, 2),
                    "ign_f1": round(if1, 2),
                    "top_gold_relations": per_rel.most_common(10),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print("report:", args.report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-DocRED benchmark harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="convert Re-DocRED to chunks + gold + ontology")
    prep.add_argument("--split", default="dev", choices=["dev", "test"])
    prep.add_argument("--limit", type=int, default=50)
    prep.set_defaults(func=cmd_prepare)

    cl = sub.add_parser("closure", help="add facts implied by ontology rules")
    cl.add_argument("pred")
    cl.add_argument("--out", default=str(OUT / "triplets_dev_closed.jsonl"))
    cl.set_defaults(func=cmd_closure)

    ev = sub.add_parser("eval", help="score extracted triplets against gold")
    ev.add_argument("pred")
    ev.add_argument("--gold", default=str(OUT / "gold_dev.jsonl"))
    ev.add_argument("--report", default=str(OUT / "eval_report.json"))
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
