param(
    [switch]$Benchmark,
    [switch]$Full,
    [switch]$DocRED,
    [switch]$Distant,
    [switch]$Pdf,
    [int]$Limit = 50,
    [int]$Workers = 8,
    [string]$Tag = "claude"
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }

function Consensus($a, $b, $out) {
    $union = "$out.union"
    Get-Content $a, $b | Set-Content -Encoding utf8 $union
    python -m src.redocred closure $union --out $out
    Remove-Item $union
}

if ($DocRED) {
    $D = "data/benchmark/docred"
    $R = "$D/$Tag"
    $O = "configs/ontology_redocred.json"
    $F = "configs/fewshot_redocred_compact.json"
    New-Item -ItemType Directory -Force $R | Out-Null

    Step "0/5 prepare DocRED splits and the Re-DocRED test split"
    python -m src.docred prepare
    python -m src.redocred fewshot --n 2 --compact --out $F
    python -m src.redocred prepare --split test --limit 500

    foreach ($split in @("validation", "test", "train_annotated")) {
        Step "$split pass A (no examples)"
        python -m src.extract $D/chunks_$split.jsonl --out $R/triplets_${split}_a.jsonl --ontology $O --compact --workers $Workers
        Step "$split pass B (two examples)"
        python -m src.extract $D/chunks_$split.jsonl --out $R/triplets_${split}_b.jsonl --ontology $O --compact --fewshot $F --workers $Workers
        Step "$split consensus + closure"
        Consensus $R/triplets_${split}_a.jsonl $R/triplets_${split}_b.jsonl $R/triplets_$split.jsonl
        if ($split -ne "test") {
            Step "$split evaluate"
            python -m src.redocred eval $R/triplets_$split.jsonl --gold $D/gold_$split.jsonl --report $R/eval_report_$split.json
        }
    }

    Step "Re-DocRED test pass A"
    python -m src.extract data/benchmark/chunks_test.jsonl --out $R/triplets_redocred_test_a.jsonl --ontology $O --compact --workers $Workers
    Step "Re-DocRED test pass B"
    python -m src.extract data/benchmark/chunks_test.jsonl --out $R/triplets_redocred_test_b.jsonl --ontology $O --compact --fewshot $F --workers $Workers
    Consensus $R/triplets_redocred_test_a.jsonl $R/triplets_redocred_test_b.jsonl $R/triplets_redocred_test.jsonl
    python -m src.redocred eval $R/triplets_redocred_test.jsonl --gold data/benchmark/gold_test.jsonl --report $R/eval_report_redocred_test.json

    Step "train_distant pass A (101,873 documents)"
    python -m src.extract $D/chunks_train_distant.jsonl --out $R/triplets_train_distant_a.jsonl --ontology $O --compact --workers $Workers
    if ($Distant) {
        Step "train_distant pass B"
        python -m src.extract $D/chunks_train_distant.jsonl --out $R/triplets_train_distant_b.jsonl --ontology $O --compact --fewshot $F --workers $Workers
        Consensus $R/triplets_train_distant_a.jsonl $R/triplets_train_distant_b.jsonl $R/triplets_train_distant.jsonl
    } else {
        python -m src.redocred closure $R/triplets_train_distant_a.jsonl --out $R/triplets_train_distant.jsonl
    }
    Step "train_distant agreement with the distant labels"
    python -m src.redocred eval $R/triplets_train_distant.jsonl --gold $D/gold_train_distant.jsonl --report $R/eval_report_train_distant.json
}
elseif ($Full) {
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
    Consensus data/benchmark/triplets_full_a.jsonl data/benchmark/triplets_full_b.jsonl data/benchmark/triplets_full.jsonl

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
