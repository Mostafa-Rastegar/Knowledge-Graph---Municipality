# Rebuild the whole knowledge graph from the raw documents.
# The pipeline is idempotent, so a second run adds no duplicate node or edge.
#
#   .\rebuild.ps1            # municipality graph (Persian)
#   .\rebuild.ps1 -Benchmark # Re-DocRED benchmark run and evaluation
#   .\rebuild.ps1 -Pdf       # also render the reports to PDF

param(
    [switch]$Benchmark,
    [switch]$Full,
    [switch]$Pdf,
    [int]$Limit = 50
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

if ($Full) {
    # The full 500-document Dev run. Both passes resume from their output file,
    # so a stopped run continues here instead of starting again.
    Step "1/4 pass A of 500 documents (no examples in the prompt)"
    python -m src.extract data/benchmark/chunks_dev.jsonl `
        --out data/benchmark/triplets_full_a.jsonl `
        --ontology configs/ontology_redocred.json

    Step "2/4 pass B of 500 documents (two examples in the prompt)"
    python -m src.extract data/benchmark/chunks_dev.jsonl `
        --out data/benchmark/triplets_full_b.jsonl `
        --ontology configs/ontology_redocred.json `
        --fewshot configs/fewshot_redocred.json

    Step "3/4 consensus of both passes, then the closure layer"
    Get-Content data/benchmark/triplets_full_a.jsonl, data/benchmark/triplets_full_b.jsonl |
        Set-Content -Encoding utf8 data/benchmark/_full_union.jsonl
    python -m src.redocred closure data/benchmark/_full_union.jsonl `
        --out data/benchmark/triplets_full.jsonl
    Remove-Item data/benchmark/_full_union.jsonl

    Step "4/4 evaluate against all 500 gold documents"
    python -m src.redocred eval data/benchmark/triplets_full.jsonl `
        --gold data/benchmark/gold_dev.jsonl `
        --report data/benchmark/eval_report_full.json
}
elseif ($Benchmark) {
    Step "1/5 prepare Re-DocRED ($Limit documents)"
    python -m src.redocred prepare --split dev --limit $Limit

    Step "2/5 build few-shot examples from the train split"
    python -m src.redocred fewshot --n 2

    Step "3/5 extract triplets"
    python -m src.extract data/benchmark/chunks_dev.jsonl `
        --out data/benchmark/triplets_dev_v4.jsonl `
        --ontology configs/ontology_redocred.json `
        --fewshot configs/fewshot_redocred.json

    Step "4/5 rule-based closure"
    python -m src.redocred closure data/benchmark/triplets_dev_v4.jsonl `
        --out data/benchmark/triplets_dev_v5.jsonl

    Step "5/5 evaluate"
    python -m src.redocred eval data/benchmark/triplets_dev_v5.jsonl `
        --report data/benchmark/eval_report_v5.json
}
else {
    Step "1/3 ingest and chunk the raw documents"
    python -m src.ingest

    Step "2/3 extract triplets under the municipality ontology"
    python -m src.extract

    Step "3/3 load the graph into Neo4j"
    python -m src.load_neo4j
}

if ($Pdf) {
    Step "render the reports to PDF"
    python -m src.md2pdf docs/FINAL_REPORT.md
    python -m src.md2pdf docs/DATASET_PROPOSAL.md
}

Write-Host "`nDone." -ForegroundColor Green
