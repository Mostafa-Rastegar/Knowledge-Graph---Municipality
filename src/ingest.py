"""Ticket 1: ingest municipality docs with audit trail.

CLI:
  python -m src.ingest <file-or-dir>

Outputs:
  data/processed/raw_extracts.jsonl
  data/processed/cleaned_pages.jsonl
  data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import regex as re
import structlog
from hazm import Normalizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, field_validator

log = structlog.get_logger()

TEHRAN_TZ = timezone(timedelta(hours=3, minutes=30))
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SUPPORTED = {".pdf", ".docx", ".txt"} | IMAGE_EXTS
_normalizer = Normalizer()
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", "؛", ".", "،", " ", ""],
)

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b\u200e\u200f\u202a-\u202e\ufeff]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINL = re.compile(r"\n{3,}")


class RawPage(BaseModel):
    document_id: str
    source_path: str
    file_name: str
    file_type: str
    page_number: int
    raw_text: str
    metadata: dict[str, str]


class CleanedPage(BaseModel):
    document_id: str
    source_path: str
    file_name: str
    file_type: str
    page_number: int
    raw_text: str
    cleaned_text: str
    cleaning_steps: list[str]
    metadata: dict[str, str]


class ChunkMeta(BaseModel):
    detected_doc_type: str
    section_title: Optional[str] = None
    created_at: str


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    source_path: str
    file_name: str
    file_type: str
    page_start: int
    page_end: int
    chunk_index: int
    language: str = "fa"
    text: str
    metadata: ChunkMeta

    @field_validator("text")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("empty chunk text")
        return value


def doc_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"doc_{digest}"


def detect_doc_type(path: Path) -> str:
    name = path.stem.lower()
    mapping = {
        "contract": "contract",
        "قرارداد": "contract",
        "complaint": "complaint",
        "شکایت": "complaint",
        "budget": "budget",
        "بودجه": "budget",
        "report": "report",
        "گزارش": "report",
        "supervision": "supervision",
        "نظارت": "supervision",
    }
    for marker, doc_type in mapping.items():
        if marker in name:
            return doc_type
    return "unknown"


def cleaning_steps() -> list[str]:
    return [
        "removed_control_and_bidi_marks",
        "normalized_persian_characters_with_hazm",
        "collapsed_repeated_spaces",
        "collapsed_repeated_blank_lines",
        "trimmed_outer_whitespace",
    ]


def clean(text: str) -> str:
    text = _CTRL.sub("", text)
    text = _normalizer.normalize(text)
    text = _MULTISPACE.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    return text.strip()


def transcribe_image(path: Path) -> str:
    """OCR a Persian document image via the vision LLM (reuses Ticket-2 client)."""
    import base64

    from src.extract import client_from_env

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    client = client_from_env()
    resp = client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini"),
        temperature=0,
        max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2000")),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "متن این تصویر سند شهرداری را دقیق و کامل به فارسی بازنویسی کن. فقط خودِ متن را برگردان، بدون توضیح."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_no, raw_text), ...]. Non-paged formats use page 1."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        import fitz

        with fitz.open(path) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    if ext == ".docx":
        import docx

        paras = [p.text for p in docx.Document(str(path)).paragraphs]
        return [(1, "\n".join(paras))]
    if ext == ".txt":
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]
    if ext in IMAGE_EXTS:
        return [(1, transcribe_image(path))]
    raise ValueError(f"unsupported file type: {ext}")


def iter_files(target: Path) -> Iterator[Path]:
    if target.is_file():
        if target.suffix.lower() in SUPPORTED:
            yield target
        return
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            yield path


def raw_pages_for_file(path: Path) -> Iterator[RawPage]:
    did = doc_id(path)
    now = datetime.now(TEHRAN_TZ).isoformat(timespec="seconds")
    metadata = {"detected_doc_type": detect_doc_type(path), "created_at": now}
    for page_no, raw_text in extract_pages(path):
        yield RawPage(
            document_id=did,
            source_path=str(path).replace("\\", "/"),
            file_name=path.name,
            file_type=path.suffix.lower().lstrip("."),
            page_number=page_no,
            raw_text=raw_text,
            metadata=metadata,
        )


def cleaned_pages_for_file(path: Path) -> Iterator[CleanedPage]:
    for raw_page in raw_pages_for_file(path):
        yield CleanedPage(
            document_id=raw_page.document_id,
            source_path=raw_page.source_path,
            file_name=raw_page.file_name,
            file_type=raw_page.file_type,
            page_number=raw_page.page_number,
            raw_text=raw_page.raw_text,
            cleaned_text=clean(raw_page.raw_text),
            cleaning_steps=cleaning_steps(),
            metadata=raw_page.metadata,
        )


def chunk_cleaned_pages(path: Path, pages: Iterator[CleanedPage]) -> Iterator[Chunk]:
    did = doc_id(path)
    digest = did[len("doc_"):]
    doc_type = detect_doc_type(path)
    now = datetime.now(TEHRAN_TZ).isoformat(timespec="seconds")
    chunk_index = 0
    for page in pages:
        if not page.cleaned_text:
            continue
        for piece in _splitter.split_text(page.cleaned_text):
            piece = piece.strip()
            if not piece:
                continue
            yield Chunk(
                document_id=did,
                chunk_id=f"chunk_{digest}_{chunk_index:04d}",
                source_path=str(path).replace("\\", "/"),
                file_name=path.name,
                file_type=path.suffix.lower().lstrip("."),
                page_start=page.page_number,
                page_end=page.page_number,
                chunk_index=chunk_index,
                text=piece,
                metadata=ChunkMeta(detected_doc_type=doc_type, created_at=now),
            )
            chunk_index += 1


def process_file(path: Path) -> Iterator[Chunk]:
    yield from chunk_cleaned_pages(path, cleaned_pages_for_file(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs with raw/cleaned/chunk audit trail")
    parser.add_argument("path", help="file or directory")
    parser.add_argument("--out", default="data/processed/chunks.jsonl")
    parser.add_argument("--raw-out", default="data/processed/raw_extracts.jsonl")
    parser.add_argument("--cleaned-out", default="data/processed/cleaned_pages.jsonl")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        parser.error(f"path not found: {target}")

    chunks_out = Path(args.out)
    raw_out = Path(args.raw_out)
    cleaned_out = Path(args.cleaned_out)
    chunks_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    cleaned_out.parent.mkdir(parents=True, exist_ok=True)

    files = 0
    pages = 0
    chunks = 0
    with (
        raw_out.open("w", encoding="utf-8") as raw_fh,
        cleaned_out.open("w", encoding="utf-8") as cleaned_fh,
        chunks_out.open("w", encoding="utf-8") as chunks_fh,
    ):
        for path in iter_files(target):
            files += 1
            cleaned_pages = list(cleaned_pages_for_file(path))
            for page in cleaned_pages:
                raw_fh.write(
                    RawPage(
                        document_id=page.document_id,
                        source_path=page.source_path,
                        file_name=page.file_name,
                        file_type=page.file_type,
                        page_number=page.page_number,
                        raw_text=page.raw_text,
                        metadata=page.metadata,
                    ).model_dump_json()
                    + "\n"
                )
                cleaned_fh.write(page.model_dump_json() + "\n")
            pages += len(cleaned_pages)

            file_chunks = 0
            for chunk in chunk_cleaned_pages(path, iter(cleaned_pages)):
                chunks_fh.write(chunk.model_dump_json() + "\n")
                file_chunks += 1
            chunks += file_chunks
            log.info("processed", file=path.name, pages=len(cleaned_pages), chunks=file_chunks)

    log.info(
        "done",
        files=files,
        pages=pages,
        chunks=chunks,
        raw_output=str(raw_out),
        cleaned_output=str(cleaned_out),
        chunks_output=str(chunks_out),
    )


if __name__ == "__main__":
    main()
