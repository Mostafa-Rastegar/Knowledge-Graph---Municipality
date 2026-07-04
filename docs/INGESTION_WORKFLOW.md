# Municipality Document Ingestion Workflow

## Goal

Keep every stage visible:

1. Original file stays unchanged in `data/raw/`.
2. Extracted raw text is saved in `data/processed/raw_extracts.jsonl`.
3. Cleaned text is saved in `data/processed/cleaned_pages.jsonl`.
4. LLM-ready chunks are saved in `data/processed/chunks.jsonl`.

## Add New File

Put new municipality document here:

```text
data/raw/
```

Supported formats:

```text
.pdf
.docx
.txt
```

Run ingestion:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -m src.ingest data/raw
```

## Output Files

### 1. Original Documents

Path:

```text
data/raw/
```

Meaning:

```text
Before processing. Source documents stay as received.
```

### 2. Raw Extracted Text

Path:

```text
data/processed/raw_extracts.jsonl
```

Meaning:

```text
Text extracted from PDF/DOCX/TXT before cleaning.
```

Use this when checking extraction quality.

### 3. Cleaned Text

Path:

```text
data/processed/cleaned_pages.jsonl
```

Meaning:

```text
Same text after Persian normalization and cleanup.
```

Each row includes:

```json
{
  "document_id": "doc_...",
  "source_path": "data/raw/example.txt",
  "file_name": "example.txt",
  "page_number": 1,
  "raw_text": "original extracted text",
  "cleaned_text": "cleaned text",
  "cleaning_steps": [
    "removed_control_and_bidi_marks",
    "normalized_persian_characters_with_hazm",
    "collapsed_repeated_spaces",
    "collapsed_repeated_blank_lines",
    "trimmed_outer_whitespace"
  ]
}
```

### 4. LLM Chunks

Path:

```text
data/processed/chunks.jsonl
```

Meaning:

```text
Final text units sent to LLM extraction.
```

## Product Demo Story

Use this explanation:

```text
این فایل خام ورودی بود.
این متن خامی است که از فایل بیرون کشیدیم.
این نسخه پاک‌سازی‌شده است.
این نسخه قطعه‌بندی‌شده است که وارد استخراج دانش می‌شود.
```

## Quality Check

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -c "import json; from pathlib import Path; print('raw', len(Path('data/processed/raw_extracts.jsonl').read_text(encoding='utf-8').splitlines())); print('cleaned', len(Path('data/processed/cleaned_pages.jsonl').read_text(encoding='utf-8').splitlines())); print('chunks', len(Path('data/processed/chunks.jsonl').read_text(encoding='utf-8').splitlines()))"
```

Expected current sample:

```text
raw 5
cleaned 5
chunks 5
```
