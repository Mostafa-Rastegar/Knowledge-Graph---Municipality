from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

DOCRED = Path("data/benchmark/docred")
SPLITS = ["validation", "test", "train_annotated", "train_distant"]


def iter_docs(split: str, batch_size: int = 2000):
    pf = pq.ParquetFile(DOCRED / f"{split}.parquet")
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            lab = row["labels"]
            yield {
                "title": row["title"],
                "sents": row["sents"],
                "vertexSet": row["vertexSet"],
                "labels": [
                    {"h": h, "t": t, "r": r, "r_name": name, "evidence": ev}
                    for h, t, r, name, ev in zip(
                        lab["head"], lab["tail"], lab["relation_id"], lab["relation_text"], lab["evidence"]
                    )
                ],
            }


def doc_text(doc: dict) -> str:
    return " ".join(" ".join(sent) for sent in doc["sents"])


def entity_block(doc: dict) -> str:
    lines = []
    for ent in doc["vertexSet"]:
        forms = sorted({m["name"] for m in ent})
        lines.append(f"- {forms[0]} [{ent[0]['type']}]" + (f" (also: {', '.join(forms[1:])})" if len(forms) > 1 else ""))
    return "\n".join(lines)


def cmd_prepare(args: argparse.Namespace) -> None:
    for split in args.splits:
        chunks_path = DOCRED / f"chunks_{split}.jsonl"
        gold_path = DOCRED / f"gold_{split}.jsonl"
        n = facts = 0
        with chunks_path.open("w", encoding="utf-8") as ch, gold_path.open("w", encoding="utf-8") as gd:
            for i, doc in enumerate(iter_docs(split)):
                if args.limit and i >= args.limit:
                    break
                doc_id = f"docred_{split}_{i:06d}"
                entities = [
                    {"idx": j, "type": e[0]["type"], "forms": sorted({m["name"] for m in e})}
                    for j, e in enumerate(doc["vertexSet"])
                ]
                ch.write(json.dumps({
                    "chunk_id": f"{doc_id}_c0",
                    "document_id": doc_id,
                    "source_path": str(DOCRED / f"{split}.parquet"),
                    "text": f"Document:\n{doc_text(doc)}\n\nEntities:\n{entity_block(doc)}",
                    "sents": [" ".join(sent) for sent in doc["sents"]],
                    "entities": entities,
                }, ensure_ascii=False) + "\n")
                gd.write(json.dumps({
                    "document_id": doc_id,
                    "title": doc["title"],
                    "entities": entities,
                    "triples": [
                        {"h": l["h"], "t": l["t"], "r": l["r"], "r_name": l["r_name"]} for l in doc["labels"]
                    ],
                }, ensure_ascii=False) + "\n")
                n += 1
                facts += len(doc["labels"])
        print(f"{split}: docs={n} gold_facts={facts} chunks={chunks_path} gold={gold_path}")


def cmd_stats(args: argparse.Namespace) -> None:
    out = {}
    for split in args.splits:
        docs = entities = mentions = facts = words = 0
        relations = set()
        types = set()
        for doc in iter_docs(split):
            docs += 1
            entities += len(doc["vertexSet"])
            mentions += sum(len(e) for e in doc["vertexSet"])
            facts += len(doc["labels"])
            words += sum(len(s) for s in doc["sents"])
            relations.update(l["r"] for l in doc["labels"])
            types.update(e[0]["type"] for e in doc["vertexSet"])
        out[split] = {"documents": docs, "entities": entities, "mentions": mentions, "facts": facts,
                      "words": words, "relations": len(relations), "entity_types": sorted(types)}
        print(split, out[split])
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description="DocRED benchmark harness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    prep.add_argument("--limit", type=int, default=0)
    prep.set_defaults(func=cmd_prepare)
    st = sub.add_parser("stats")
    st.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    st.add_argument("--out", default=str(DOCRED / "stats.json"))
    st.set_defaults(func=cmd_stats)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
