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
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def doc_text(doc: dict) -> str:
    return " ".join(" ".join(sent) for sent in doc["sents"])


def entity_block(doc: dict) -> str:
    lines = []
    for ent in doc["vertexSet"]:
        forms = sorted({m["name"] for m in ent})
        lines.append(f"- {forms[0]} [{ent[0]['type']}]" + (f" (also: {', '.join(forms[1:])})" if len(forms) > 1 else ""))
    return "\n".join(lines)


def build_ontology(train: list[dict], names: dict[str, str]) -> dict:
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
    keys = set()
    for doc in train:
        for lab in doc["labels"]:
            heads = {norm(m["name"]) for m in doc["vertexSet"][lab["h"]]}
            tails = {norm(m["name"]) for m in doc["vertexSet"][lab["t"]]}
            rel = names[lab["r"]]
            for h in heads:
                for t in tails:
                    keys.add(f"{h}|{rel}|{t}")
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


def cmd_fewshot(args: argparse.Namespace) -> None:
    names = rel_names()
    train = load_split("train")


    ranked = sorted(
        (d for d in train if 100 <= len(doc_text(d).split()) <= 200 and 25 <= len(d["labels"]) <= 40),
        key=lambda d: -len({names[l["r"]] for l in d["labels"]}),
    )
    examples = []
    for doc in ranked[: args.n]:
        examples.append(
            {
                "input": f"Document:\n{doc_text(doc)}\n\nEntities:\n{entity_block(doc)}",
                "output": {
                    "triplets": [
                        {
                            "subject": {
                                "type": doc["vertexSet"][l["h"]][0]["type"],
                                "name": sorted({m["name"] for m in doc["vertexSet"][l["h"]]})[0],
                            },
                            "predicate": names[l["r"]],
                            "object": {
                                "type": doc["vertexSet"][l["t"]][0]["type"],
                                "name": sorted({m["name"] for m in doc["vertexSet"][l["t"]]})[0],
                            },
                            "evidence": " ".join(doc["sents"][l["evidence"][0]])
                            if l.get("evidence")
                            else doc_text(doc)[:200],
                        }
                        for l in doc["labels"]
                    ]
                },
            }
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    for ex in examples:
        print(f"example words={len(ex['input'].split())} triplets={len(ex['output']['triplets'])}")
    print(f"wrote {path}")


def cmd_graph(args: argparse.Namespace) -> None:
    rows = [
        json.loads(line)
        for line in Path(args.pred).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    doc_id = args.doc or collections.Counter(r["document_id"] for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if r["document_id"] == doc_id][: args.limit]

    ids: dict[str, str] = {}

    def node(ent: dict) -> str:
        name = ent["name"]
        if name not in ids:
            ids[name] = f"n{len(ids)}"
        return ids[name]

    lines = ["```mermaid", "graph LR"]
    for r in rows:
        s, o = node(r["subject"]), node(r["object"])
        label = r["predicate"].replace("the administrative territorial entity", "admin. entity")
        style = "-.->" if r.get("derived_by") else "-->"
        lines.append(f'  {s}["{r["subject"]["name"]} ({r["subject"]["type"]})"] {style}|{label}| '
                     f'{o}["{r["object"]["name"]} ({r["object"]["type"]})"]')
    lines.append("```")
    out = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"doc={doc_id} edges={len(rows)} nodes={len(ids)} -> {args.out}")
    else:
        print(out)


LOCATED_IN = "located in the administrative territorial entity"
CONTAINS = "contains administrative territorial entity"
COUNTRY = "country"


_DOC_ID = re.compile(rb'"document_id":\s*"([^"]+)"')

INVERSE = {LOCATED_IN: CONTAINS, CONTAINS: LOCATED_IN, "part of": "has part", "has part": "part of"}


def doc_offsets(path: Path) -> dict[str, list[int]]:
    offsets: dict[str, list[int]] = {}
    with Path(path).open("rb") as fh:
        pos = 0
        for raw in fh:
            if raw.strip():
                head = raw[: raw.find(b'"document_id"') + 200] if b'"document_id"' in raw else raw
                m = _DOC_ID.search(head)
                doc_id = m.group(1).decode("utf-8") if m else json.loads(raw.decode("utf-8-sig"))["document_id"]
                offsets.setdefault(doc_id, []).append(pos)
            pos += len(raw)
    return offsets


def grouped_rows(path: Path):
    offsets = doc_offsets(path)
    with Path(path).open("rb") as fh:
        for doc_id, positions in offsets.items():
            rows = []
            for pos in positions:
                fh.seek(pos)
                rows.append(json.loads(fh.readline().decode("utf-8-sig")))
            yield doc_id, rows


def close_doc(doc_id: str, doc_rows: list[dict], added: collections.Counter) -> list[dict]:
    facts = {(r["subject"]["name"], r["predicate"], r["object"]["name"]): r for r in doc_rows}
    types = {}
    evidence = {}
    for r in doc_rows:
        types[r["subject"]["name"]] = r["subject"]["type"]
        types[r["object"]["name"]] = r["object"]["type"]
        evidence[(r["subject"]["name"], r["predicate"], r["object"]["name"])] = r["evidence"]

    def node_key(name: str) -> str:
        return f"{types.get(name, 'MISC')}:{' '.join(name.split())}"

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
            "subject": {"type": types.get(subject, "MISC"), "name": subject, "key": node_key(subject)},
            "predicate": predicate,
            "object": {"type": types.get(obj, "MISC"), "name": obj, "key": node_key(obj)},
            "evidence": " | ".join(w for w in why if w)[:600],
            "derived_by": rule,
        }
        added[rule] += 1
        return True

    for (s, p, o), rec in list(facts.items()):
        if p in INVERSE:
            add(o, INVERSE[p], s, [rec["evidence"]], "R1_inverse")

    for _ in range(4):
        chain = [(s, o) for (s, p, o) in list(facts) if p == LOCATED_IN]
        grew = False
        for a, b in chain:
            for c, d in chain:
                if b == c:
                    grew |= add(
                        a, LOCATED_IN, d,
                        [evidence.get((a, LOCATED_IN, b), ""), evidence.get((c, LOCATED_IN, d), "")],
                        "R2_transitive",
                    )
        if not grew:
            break

    countries = [(s, o) for (s, p, o) in list(facts) if p == COUNTRY]
    for a, b in [(s, o) for (s, p, o) in list(facts) if p == LOCATED_IN]:
        for c, d in countries:
            if b == c:
                add(a, COUNTRY, d, [evidence.get((a, LOCATED_IN, b), "")], "R3_chain_country")
    return list(facts.values())


def cmd_closure(args: argparse.Namespace) -> None:
    added = collections.Counter()
    n_in = n_out = 0
    with Path(args.out).open("w", encoding="utf-8") as out:
        for doc_id, doc_rows in grouped_rows(Path(args.pred)):
            n_in += len(doc_rows)
            for rec in close_doc(doc_id, doc_rows, added):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    print(f"input rows {n_in} -> output rows {n_out}")
    for rule, n in added.most_common():
        print(f"  {rule}: +{n}")
    print("out:", args.out)


def cmd_eval(args: argparse.Namespace) -> None:
    train_keys = set(json.loads(Path(args.train_keys).read_text(encoding="utf-8")))
    gold_at = {doc_id: pos[0] for doc_id, pos in doc_offsets(Path(args.gold)).items()}
    gold_fh = Path(args.gold).open("rb")

    def gold_doc(doc_id: str) -> dict:
        gold_fh.seek(gold_at[doc_id])
        return json.loads(gold_fh.readline().decode("utf-8-sig"))

    totals = collections.Counter()
    per_rel_gold = collections.Counter()
    per_rel_hit = collections.Counter()

    def score_doc(doc_id: str, rec: dict, rows: list[dict]) -> None:
        typed: dict[tuple[str, str], int] = {}
        plain: dict[str, int] = {}
        forms: dict[int, list[str]] = {}
        for ent in rec["entities"]:
            forms[ent["idx"]] = [norm(f) for f in ent["forms"]]
            for form in forms[ent["idx"]]:
                typed.setdefault((form, ent["type"]), ent["idx"])
                plain.setdefault(form, ent["idx"])

        def resolve(entity: dict) -> int | None:
            key = norm(entity["name"])
            hit = typed.get((key, entity.get("type", "")))
            if hit is not None:
                return hit
            return None if args.strict_types else plain.get(key)

        def seen(h: int, rel: str, t: int) -> bool:
            return any(f"{hf}|{rel}|{tf}" in train_keys for hf in forms[h] for tf in forms[t])

        gold = {(tr["h"], tr["r_name"], tr["t"]) for tr in rec["triples"]}
        pred = set()
        for row in rows:
            totals["predicted_rows"] += 1
            h, t = resolve(row["subject"]), resolve(row["object"])
            if h is None or t is None:
                totals["entity_unmatched"] += 1
                continue
            pred.add((h, row["predicate"], t))
        hit = gold & pred
        gold_ign = {k for k in gold if not seen(*k)}
        pred_ign = {k for k in pred if not seen(*k)}
        totals["gold"] += len(gold)
        totals["pred"] += len(pred)
        totals["hit"] += len(hit)
        totals["gold_ign"] += len(gold_ign)
        totals["pred_ign"] += len(pred_ign)
        totals["hit_ign"] += len(gold_ign & pred_ign)
        totals["documents"] += 1
        for _, rel, _ in gold:
            per_rel_gold[rel] += 1
        for _, rel, _ in hit:
            per_rel_hit[rel] += 1

    scored: set[str] = set()
    for doc_id, rows in grouped_rows(Path(args.pred)):
        if doc_id not in gold_at:
            continue
        scored.add(doc_id)
        score_doc(doc_id, gold_doc(doc_id), rows)
    for doc_id in gold_at:
        if doc_id not in scored:
            score_doc(doc_id, gold_doc(doc_id), [])
    gold_fh.close()

    def score(gold: int, pred: int, hit: int) -> tuple[float, float, float]:
        p = hit / pred * 100 if pred else 0.0
        r = hit / gold * 100 if gold else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    p, r, f1 = score(totals["gold"], totals["pred"], totals["hit"])
    ip, ir, if1 = score(totals["gold_ign"], totals["pred_ign"], totals["hit_ign"])
    print(f"documents        : {totals['documents']}")
    print(f"gold triples     : {totals['gold']}  (unseen in train: {totals['gold_ign']})")
    print(f"predicted rows   : {totals['predicted_rows']}  (entity not matched: {totals['entity_unmatched']})")
    print(f"predicted triples: {totals['pred']}")
    print(f"Precision {p:.2f}  Recall {r:.2f}  F1 {f1:.2f}")
    print(f"Ign Precision {ip:.2f}  Ign Recall {ir:.2f}  Ign F1 {if1:.2f}")
    print(f"distinct gold relations in this subset: {len(per_rel_gold)}")

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "entity_matching": "strict_types" if args.strict_types else "name_fallback",
                    "documents": totals["documents"],
                    "gold_triples": totals["gold"],
                    "gold_triples_unseen": totals["gold_ign"],
                    "predicted_rows": totals["predicted_rows"],
                    "predicted_triples": totals["pred"],
                    "predicted_triples_unseen": totals["pred_ign"],
                    "entity_unmatched": totals["entity_unmatched"],
                    "precision": round(p, 2),
                    "recall": round(r, 2),
                    "f1": round(f1, 2),
                    "ign_precision": round(ip, 2),
                    "ign_recall": round(ir, 2),
                    "ign_f1": round(if1, 2),
                    "top_gold_relations": per_rel_gold.most_common(10),
                    "recall_per_relation": {
                        rel: {"gold": count, "recall": round(per_rel_hit[rel] / count * 100, 1)}
                        for rel, count in per_rel_gold.most_common(10)
                    },
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

    fs = sub.add_parser("fewshot", help="build prompt examples from the train split")
    fs.add_argument("--n", type=int, default=2)
    fs.add_argument("--out", default="configs/fewshot_redocred.json")
    fs.set_defaults(func=cmd_fewshot)

    gr = sub.add_parser("graph", help="print one document's graph as Mermaid")
    gr.add_argument("pred")
    gr.add_argument("--doc", help="document id; default is the document with most facts")
    gr.add_argument("--limit", type=int, default=25)
    gr.add_argument("--out")
    gr.set_defaults(func=cmd_graph)

    cl = sub.add_parser("closure", help="add facts implied by ontology rules")
    cl.add_argument("pred")
    cl.add_argument("--out", default=str(OUT / "triplets_dev_closed.jsonl"))
    cl.set_defaults(func=cmd_closure)

    ev = sub.add_parser("eval", help="score extracted triplets against gold")
    ev.add_argument("pred")
    ev.add_argument("--gold", default=str(OUT / "gold_dev.jsonl"))
    ev.add_argument("--report", default=str(OUT / "eval_report.json"))
    ev.add_argument("--train-keys", default=str(OUT / "train_fact_keys.json"))
    ev.add_argument("--strict-types", action="store_true",
                    help="drop a prediction whose entity type does not match gold")
    ev.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
