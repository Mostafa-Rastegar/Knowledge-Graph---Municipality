from __future__ import annotations

import argparse
import hashlib
import json
import os
import re as _re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


COMPACT = False


def load_compact_ontology(path: Path) -> None:
    global ENTITY_TYPES, ALLOWED_RELATIONS, SYSTEM_PROMPT, COMPACT
    load_ontology(path)
    COMPACT = True
    spec = json.loads(path.read_text(encoding="utf-8"))
    relations = "\n".join(f"- {name}" for name in sorted(set(spec["relations"].values())))
    SYSTEM_PROMPT = (
        "You are a document-level relation extraction system.\n"
        "You get a document as numbered sentences and a numbered list of its entities.\n"
        "Return every relation that the document states between two listed entities.\n\n"
        f"Allowed relations (use the exact name):\n{relations}\n\n"
        "Rules:\n"
        "- Be exhaustive. Check every pair of entities, not only the pairs that\n"
        "  appear in the same sentence. A document normally holds 20 to 40 relations.\n"
        "- Return a relation when the document states it, and also when the document\n"
        "  makes it certain. Example: a place is in a city, the city is in a country,\n"
        "  so the place is in that country too.\n"
        "- When the list holds a relation and its inverse, return both directions.\n"
        "- Name each entity by its number from the list. Never write entity names.\n"
        "- Give the numbers of the sentences that prove the relation.\n"
        "- Ignore any relation that is not in the list.\n\n"
        "Answer with JSON only, in this exact shape:\n"
        '{"r":[[subject_number,"relation name",object_number,[sentence_numbers]]]}\n'
        'If you find nothing: {"r":[]}'
    )


def compact_input(chunk: dict) -> str:
    sents = "\n".join(f"{i}: {s}" for i, s in enumerate(chunk["sents"]))
    ents = []
    for ent in chunk["entities"]:
        also = ent["forms"][1:]
        line = f"{ent['idx']}: {ent['forms'][0]} [{ent['type']}]"
        ents.append(line + (f" (also: {', '.join(also)})" if also else ""))
    return f"Sentences:\n{sents}\n\nEntities:\n" + "\n".join(ents)


def parse_compact(content: str, chunk: dict) -> list[Triplet]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        log.warning("llm_bad_json")
        return []
    entities = {e["idx"]: e for e in chunk["entities"]}
    sents = chunk["sents"]
    out: list[Triplet] = []
    for row in data.get("r", []):
        if not isinstance(row, list) or len(row) < 3:
            continue
        head, predicate, tail = row[0], row[1], row[2]
        evidence_ids = row[3] if len(row) > 3 and isinstance(row[3], list) else []
        if not isinstance(head, int) or not isinstance(tail, int):
            log.warning("rejected_non_integer_entity", head=str(head)[:40], tail=str(tail)[:40])
            continue
        if head not in entities or tail not in entities:
            log.warning("rejected_unknown_entity_number", head=head, tail=tail)
            continue
        text = " ".join(sents[i] for i in evidence_ids if isinstance(i, int) and 0 <= i < len(sents))
        if not text:
            text = " ".join(sents)
        item = {
            "subject": {"type": entities[head]["type"], "name": entities[head]["forms"][0]},
            "predicate": predicate if isinstance(predicate, str) else "",
            "object": {"type": entities[tail]["type"], "name": entities[tail]["forms"][0]},
            "evidence": text[:600],
        }
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


def load_fewshot(path: Path) -> None:
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


_BUDGET_CODE = _re.compile(r"[۰-۹0-9]{4}\s*-\s*[۰-۹0-9]{1,2}\s*-\s*[۰-۹0-9]{1,3}")


def entity_key(entity: Entity) -> str:
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
    if os.environ.get("LLM_PROVIDER", "").strip().lower() == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
        base_url = os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").strip()
        if not api_key:
            raise SystemExit("MISTRAL_API_KEY is empty. Put the key in .env (never hardcode it).")
    else:
        api_key = os.environ.get("LLM_API_KEY", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip()
        if not api_key:
            raise SystemExit("LLM_API_KEY is empty. Put the key in .env (never hardcode it).")
        if not base_url:
            raise SystemExit("LLM_BASE_URL is empty. Set it in .env.")
    timeout = float(os.environ.get("LLM_TIMEOUT", "180"))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def model_name() -> str:
    if os.environ.get("LLM_PROVIDER", "").strip().lower() == "mistral":
        return os.environ.get("MISTRAL_MODEL", "ministral-14b-latest")
    return os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini")


USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "calls": 0}
_usage_lock = threading.Lock()


def _count_usage(resp) -> None:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    with _usage_lock:
        USAGE["prompt_tokens"] += usage.prompt_tokens or 0
        USAGE["completion_tokens"] += usage.completion_tokens or 0
        USAGE["cached_tokens"] += cached
        USAGE["calls"] += 1


_rate_lock = threading.Lock()
_next_slot = [0.0]


def _wait_for_slot() -> None:
    rps = float(os.environ.get("LLM_RPS", "0"))
    if rps <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _next_slot[0])
        _next_slot[0] = start + 1.0 / rps
    delay = start - now
    if delay > 0:
        time.sleep(delay)


PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
_prompt_file = [None]


def _claude_exe() -> str:
    exe = os.environ.get("CLAUDE_CLI", "").strip()
    if exe:
        return exe
    found = shutil.which("claude")
    if found:
        native = Path(found).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.exists():
            return str(native)
        return found
    raise SystemExit("claude CLI not found. Install Claude Code or set CLAUDE_CLI in .env.")


def _claude_prompt_file() -> str:
    if _prompt_file[0] is None:
        fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        fh.write(SYSTEM_PROMPT)
        fh.close()
        _prompt_file[0] = fh.name
    return _prompt_file[0]


def call_claude_cli(text: str) -> str:
    cmd = [
        _claude_exe(), "-p",
        "--model", os.environ.get("CLAUDE_MODEL", "sonnet"),
        "--tools", "",
        "--output-format", "json",
        "--no-session-persistence",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "",
        "--disable-slash-commands",
        "--system-prompt-file", _claude_prompt_file(),
    ]
    effort = os.environ.get("LLM_EFFORT", "").strip()
    if effort:
        cmd += ["--effort", effort]
    proc = subprocess.run(
        cmd, input=text, capture_output=True, text=True, encoding="utf-8",
        timeout=float(os.environ.get("LLM_TIMEOUT", "180")),
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), "")
    if proc.returncode != 0 or not line:
        raise RuntimeError(f"claude exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}")
    data = json.loads(line)
    if data.get("is_error"):
        raise RuntimeError(f"claude error: {str(data.get('result'))[:300]}")
    usage = data.get("usage", {})
    with _usage_lock:
        USAGE["prompt_tokens"] += (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
        USAGE["cached_tokens"] += usage.get("cache_read_input_tokens") or 0
        USAGE["completion_tokens"] += usage.get("output_tokens") or 0
        USAGE["calls"] += 1
    result = data.get("result") or ""
    start, end = result.find("{"), result.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError(f"claude answer is not json: {result[:200]}")
    result = result[start:end + 1]
    json.loads(result)
    return result


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=60))
def call_llm(client: Optional[OpenAI], text: str) -> str:
    _wait_for_slot()
    if PROVIDER == "claude-cli":
        return call_claude_cli(text)
    resp = client.chat.completions.create(
        model=model_name(),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2000")),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    _count_usage(resp)
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
    mode = "compact|" if COMPACT else ""
    return hashlib.sha1((mode + text).encode("utf-8")).hexdigest()[:12]


def extract_chunk(client: OpenAI, chunk: dict) -> list[dict]:
    if COMPACT:
        answer = call_llm(client, compact_input(chunk))
        triplets = parse_compact(answer, chunk)
    else:
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


def empty_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".empty")


def load_existing(path: Path) -> dict[str, tuple[str, list[dict]]]:
    by_chunk: dict[str, tuple[str, list[dict]]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sha, records = by_chunk.setdefault(rec["chunk_id"], (rec.get("chunk_sha", ""), []))
            records.append(rec)
    marker = empty_path(path)
    if marker.exists():
        for line in marker.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                chunk_id, sha = line.split("\t", 1)
                by_chunk.setdefault(chunk_id, (sha, []))
    return by_chunk


def read_chunks(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ontology triplets from chunks")
    parser.add_argument("chunks", nargs="?", default="data/processed/chunks.jsonl")
    parser.add_argument("--out", default="data/extracted/triplets.jsonl")
    parser.add_argument("--force", action="store_true", help="re-extract even cached chunks")
    parser.add_argument("--ontology", help="JSON ontology file; default is the municipality one")
    parser.add_argument("--fewshot", help="JSON file with worked examples for the prompt")
    parser.add_argument("--compact", action="store_true",
                        help="answer with entity and sentence numbers instead of names and quotes")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LLM_WORKERS", "1")))
    parser.add_argument("--limit", type=int, default=0, help="stop after this many chunks")
    args = parser.parse_args()

    if args.ontology:
        (load_compact_ontology if args.compact else load_ontology)(Path(args.ontology))
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
    if not args.force:
        leftover = load_existing(out_path.with_suffix(out_path.suffix + ".part"))
        if leftover:
            log.info("resumed_from_partial_run", chunks=len(leftover))
            existing.update(leftover)

    chunks = read_chunks(chunks_path)
    if args.limit:
        chunks = chunks[: args.limit]
    todo: list[dict] = []
    seen_facts: set[str] = set()
    chunks_done = chunks_cached = chunks_failed = facts = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    started = time.time()

    def write(fh, records: list[dict]) -> int:
        n = 0
        for rec in records:
            if rec["fact_id"] in seen_facts:
                continue
            seen_facts.add(rec["fact_id"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
        fh.flush()
        return n

    with tmp_path.open("w", encoding="utf-8") as fh, empty_path(out_path).open("a", encoding="utf-8") as empty_fh:
        for chunk in chunks:
            cached = existing.get(chunk["chunk_id"])
            if cached and cached[0] == chunk_sha(chunk["text"]):
                facts += write(fh, cached[1])
                chunks_cached += 1
                chunks_done += 1
            else:
                todo.append(chunk)
        log.info("plan", total=len(chunks), cached=chunks_cached, todo=len(todo), workers=args.workers)

        client: Optional[OpenAI] = client_from_env() if todo and PROVIDER != "claude-cli" else None
        if todo:
            log.info("provider", provider=PROVIDER,
                     model=os.environ.get("CLAUDE_MODEL", "sonnet") if PROVIDER == "claude-cli" else model_name(),
                     compact=COMPACT)

        def work(chunk: dict):
            try:
                return chunk, extract_chunk(client, chunk), None
            except Exception as err:
                return chunk, None, err

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(work, chunk) for chunk in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                chunk, records, err = fut.result()
                if err is not None:
                    chunks_failed += 1
                    log.error("chunk_failed", chunk_id=chunk["chunk_id"], error=str(err))
                else:
                    chunks_done += 1
                    facts += write(fh, records)
                    if not records:
                        empty_fh.write(f"{chunk['chunk_id']}\t{chunk_sha(chunk['text'])}\n")
                        empty_fh.flush()
                if i % 50 == 0 or i == len(todo):
                    elapsed = time.time() - started
                    rate = i / elapsed * 60 if elapsed else 0.0
                    log.info("progress", done=i, todo=len(todo), per_min=round(rate, 1),
                             eta_min=round((len(todo) - i) / rate, 1) if rate else None,
                             failed=chunks_failed, facts=facts, **USAGE)

    os.replace(tmp_path, out_path)
    elapsed = round(time.time() - started, 1)
    log.info("done", chunks=chunks_done, cached=chunks_cached, failed=chunks_failed,
             facts=facts, output=str(out_path), seconds=elapsed, **USAGE)
    if USAGE["calls"]:
        out_path.with_suffix(out_path.suffix + ".usage.json").write_text(
            json.dumps({**USAGE, "seconds": elapsed, "chunks_extracted": USAGE["calls"],
                        "model": os.environ.get("LLM_MODEL", ""), "workers": args.workers}, indent=2),
            encoding="utf-8",
        )
    if chunks_failed:
        log.warning("rerun_to_retry_failed_chunks", failed=chunks_failed)


if __name__ == "__main__":
    main()
