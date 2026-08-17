"""Ticket 2: extract ontology triplets from chunks with an LLM.

CLI:
  python -m src.extract data/processed/chunks.jsonl

Reads LLM-ready chunks, calls the Hormouz OpenAI-compatible API, validates the
output against the strict municipality ontology, drops anything that does not
fit, and writes one triplet record per allowed fact (with evidence) to
data/extracted/triplets.jsonl.

Rerun-safe: a chunk already present in the output is reused as-is, so the same
chunk never produces duplicate extraction records and only new chunks cost an
LLM call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import structlog
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()
load_dotenv()

ENTITY_TYPES = {"Project", "Contractor", "Location", "Official", "Budget", "Complaint"}

# (subject_type, predicate, object_type) -> the only directed facts we keep.
ALLOWED_RELATIONS = {
    ("Contractor", "EXECUTOR_OF", "Project"),
    ("Project", "LOCATED_IN", "Location"),
    ("Official", "SUPERVISOR_OF", "Project"),
    ("Budget", "FINANCES", "Project"),
    ("Complaint", "COMPLAINS_ABOUT", "Project"),
    ("Complaint", "COMPLAINS_ABOUT", "Location"),
}

SYSTEM_PROMPT = """تو یک استخراج‌کننده دانش برای اسناد شهرداری هستی.
از متن داده‌شده فقط موجودیت‌ها و رابطه‌های زیر را استخراج کن. هیچ نوع دیگری مجاز نیست.

موجودیت‌های مجاز:
- Project (پروژه عمرانی/شهری)
- Contractor (پیمانکار)
- Location (محله/منطقه/میدان/خیابان)
- Official (مسئول/ناظر/مدیر)
- Budget (ردیف یا اعتبار بودجه)
- Complaint (شکایت شهروندی)

رابطه‌های مجاز (جهت‌دار):
- Contractor EXECUTOR_OF Project
- Project LOCATED_IN Location
- Official SUPERVISOR_OF Project
- Budget FINANCES Project
- Complaint COMPLAINS_ABOUT Project
- Complaint COMPLAINS_ABOUT Location

قواعد:
- فقط رابطه‌هایی را برگردان که صراحتاً در متن آمده‌اند.
- برای هر رابطه باید عین جمله یا عبارت متن به عنوان evidence آورده شود.
- اگر چیزی در ontology نیست، آن را نادیده بگیر.
- نام موجودیت‌ها را تمیز و کوتاه بنویس (بدون کلمات اضافه).

فقط JSON معتبر با این ساختار برگردان:
{"triplets":[{"subject":{"type":"...","name":"..."},"predicate":"...","object":{"type":"...","name":"..."},"evidence":"..."}]}
اگر رابطه‌ای پیدا نشد: {"triplets":[]}
"""


def load_ontology(path: Path) -> None:
    """Replace the hand-written municipality ontology with one defined in JSON.

    The validation code, the evidence rule and the dedup logic stay the same;
    only the schema changes. The Re-DocRED benchmark run uses this.
    """
    global ENTITY_TYPES, ALLOWED_RELATIONS, SYSTEM_PROMPT
    spec = json.loads(path.read_text(encoding="utf-8"))
    ENTITY_TYPES = set(spec["entity_types"])
    ALLOWED_RELATIONS = {tuple(item) for item in spec["allowed"]}
    relations = "\n".join(f"- {name}" for name in sorted(set(spec["relations"].values())))
    SYSTEM_PROMPT = (
        "You are a document-level relation extraction system.\n"
        "You get a document and the list of entities that appear in it.\n"
        "Return every relation that the document states between two of those entities.\n\n"
        f"Entity types: {', '.join(sorted(ENTITY_TYPES))}\n\n"
        f"Allowed relations (use the exact name):\n{relations}\n\n"
        "Rules:\n"
        "- Be exhaustive. Check every pair of entities in the list, not only the\n"
        "  pairs that appear in the same sentence. A document of this size normally\n"
        "  holds 20 to 40 relations.\n"
        "- Return a relation when the document states it, and also when the document\n"
        "  makes it certain. Example: the text says a place is in a city, and says the\n"
        "  city is in a country, so the place is in that country too.\n"
        "- When the list holds a relation and its inverse, return both directions.\n"
        "  Example: 'A located in the administrative territorial entity B' and\n"
        "  'B contains administrative territorial entity A'.\n"
        "- Copy each entity name exactly as it appears in the entity list.\n"
        "- Relations are directed. Put the subject first.\n"
        "- Give the sentence from the document that proves the relation as evidence.\n"
        "  For an inferred relation, give the sentence that starts the chain.\n"
        "- Ignore any relation that is not in the list.\n\n"
        "Return only valid JSON with this structure:\n"
        '{"triplets":[{"subject":{"type":"...","name":"..."},"predicate":"...",'
        '"object":{"type":"...","name":"..."},"evidence":"..."}]}\n'
        'If you find nothing: {"triplets":[]}'
    )


def load_fewshot(path: Path) -> None:
    """Append worked examples to the system prompt.

    The examples come from the train split, never from the split we score on.
    They show the model the answer format and the relation density we expect.
    """
    global SYSTEM_PROMPT
    examples = json.loads(path.read_text(encoding="utf-8"))
    blocks = []
    for i, ex in enumerate(examples, 1):
        answer = json.dumps(ex["output"], ensure_ascii=False)
        blocks.append(f"Example {i} input:\n{ex['input']}\n\nExample {i} answer:\n{answer}")
    SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + "\n\n".join(blocks)


class Entity(BaseModel):
    type: str
    name: str

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type: {value}")
        return value

    @field_validator("name")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty entity name")
        return value.strip()


class Triplet(BaseModel):
    subject: Entity
    predicate: str
    object: Entity
    evidence: str

    @field_validator("evidence")
    @classmethod
    def _has_evidence(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("missing evidence text")
        return value.strip()

    @field_validator("predicate")
    @classmethod
    def _known_predicate(cls, value: str) -> str:
        return value.strip()

    def is_allowed(self) -> bool:
        return (self.subject.type, self.predicate, self.object.type) in ALLOWED_RELATIONS


import re as _re

# A budget line has a stable code like ۱۴۰۳-۳-۱۷; use it as the identity so the
# same budget named "ردیف بودجه عمرانی ۱۴۰۳-۳-۱۷" and "۱۴۰۳-۳-۱۷" collapse to one.
_BUDGET_CODE = _re.compile(r"[۰-۹0-9]{4}\s*-\s*[۰-۹0-9]{1,2}\s*-\s*[۰-۹0-9]{1,3}")


def entity_key(entity: Entity) -> str:
    """Stable id reused by the Neo4j loader: collapses whitespace, keeps text.

    For Budget, key on the code number when present so the same budget referenced
    across documents resolves to a single node.
    """
    name = " ".join(entity.name.split())
    if entity.type == "Budget":
        m = _BUDGET_CODE.search(name)
        if m:
            name = _re.sub(r"\s*-\s*", "-", m.group(0))
    return f"{entity.type}:{name}"


def fact_id(chunk_id: str, triplet: Triplet) -> str:
    raw = f"{chunk_id}|{entity_key(triplet.subject)}|{triplet.predicate}|{entity_key(triplet.object)}"
    return "fact_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def client_from_env() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "").strip()
    if not api_key:
        raise SystemExit("LLM_API_KEY is empty. Put the key in .env (never hardcode it).")
    if not base_url:
        raise SystemExit("LLM_BASE_URL is empty. Set it in .env.")
    return OpenAI(api_key=api_key, base_url=base_url)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
def call_llm(client: OpenAI, text: str) -> str:
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2000")),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content or "{}"


def parse_triplets(content: str) -> list[Triplet]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("llm_bad_json")
        return []
    out: list[Triplet] = []
    for item in data.get("triplets", []):
        try:
            triplet = Triplet.model_validate(item)
        except ValidationError as exc:
            log.warning("rejected_invalid_triplet", error=str(exc.errors()[:1]))
            continue
        if not triplet.is_allowed():
            log.warning(
                "rejected_outside_ontology",
                relation=(triplet.subject.type, triplet.predicate, triplet.object.type),
            )
            continue
        out.append(triplet)
    return out


def chunk_sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def extract_chunk(client: OpenAI, chunk: dict) -> list[dict]:
    triplets = parse_triplets(call_llm(client, chunk["text"]))
    sha = chunk_sha(chunk["text"])
    records = []
    for triplet in triplets:
        records.append(
            {
                "fact_id": fact_id(chunk["chunk_id"], triplet),
                "chunk_id": chunk["chunk_id"],
                "chunk_sha": sha,
                "document_id": chunk["document_id"],
                "source_path": chunk["source_path"],
                "subject": {**triplet.subject.model_dump(), "key": entity_key(triplet.subject)},
                "predicate": triplet.predicate,
                "object": {**triplet.object.model_dump(), "key": entity_key(triplet.object)},
                "evidence": triplet.evidence,
            }
        )
    return records


def load_existing(path: Path) -> dict[str, tuple[str, list[dict]]]:
    """chunk_id -> (chunk_sha, records). Reused only when the text is unchanged,
    so editing a file's content re-extracts instead of serving stale triplets."""
    by_chunk: dict[str, tuple[str, list[dict]]] = {}
    if not path.exists():
        return by_chunk
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        sha, records = by_chunk.setdefault(rec["chunk_id"], (rec.get("chunk_sha", ""), []))
        records.append(rec)
    return by_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ontology triplets from chunks")
    parser.add_argument("chunks", nargs="?", default="data/processed/chunks.jsonl")
    parser.add_argument("--out", default="data/extracted/triplets.jsonl")
    parser.add_argument("--force", action="store_true", help="re-extract even cached chunks")
    parser.add_argument("--ontology", help="JSON ontology file; default is the municipality one")
    parser.add_argument("--fewshot", help="JSON file with worked examples for the prompt")
    args = parser.parse_args()

    if args.ontology:
        load_ontology(Path(args.ontology))
        log.info("ontology_loaded", path=args.ontology, entity_types=len(ENTITY_TYPES),
                 allowed_relations=len(ALLOWED_RELATIONS))
    if args.fewshot:
        load_fewshot(Path(args.fewshot))
        log.info("fewshot_loaded", path=args.fewshot, prompt_chars=len(SYSTEM_PROMPT))

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        parser.error(f"chunks file not found: {chunks_path}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {} if args.force else load_existing(out_path)
    client: Optional[OpenAI] = None

    seen_facts: set[str] = set()
    chunks_done = chunks_cached = facts = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for line in chunks_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            chunk_id = chunk["chunk_id"]
            cached = existing.get(chunk_id)
            if cached and cached[0] == chunk_sha(chunk["text"]):
                records = cached[1]
                chunks_cached += 1
            else:
                if client is None:
                    client = client_from_env()
                records = extract_chunk(client, chunk)
                log.info("extracted", chunk_id=chunk_id, triplets=len(records))
            chunks_done += 1
            for rec in records:
                if rec["fact_id"] in seen_facts:
                    continue  # dedup identical fact across chunks
                seen_facts.add(rec["fact_id"])
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                facts += 1

    log.info("done", chunks=chunks_done, cached=chunks_cached, facts=facts, output=str(out_path))


if __name__ == "__main__":
    main()
