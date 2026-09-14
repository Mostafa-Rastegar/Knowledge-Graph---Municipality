# Build the report locally with XeLaTeX. Run from this folder:
#     .\build.ps1
# Intermediates go to .\build\ ; the finished PDF lands next to this script.
#
#     .\build.ps1 -Quick     skip BibTeX and the extra passes (fast draft)
#     .\build.ps1 -Clean     delete .\build\ first

param(
    [switch]$Quick,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Job  = 'AUTthesis'
$Out  = Join-Path $Root 'build'

# --- locate the toolchain -------------------------------------------------
$xelatex = (Get-Command xelatex -ErrorAction SilentlyContinue)
if ($null -eq $xelatex) {
    Write-Host "xelatex not found on PATH." -ForegroundColor Red
    Write-Host "If MiKTeX was just installed, open a NEW terminal so PATH refreshes."
    exit 1
}
$bibtex = (Get-Command bibtex -ErrorAction SilentlyContinue)

if ($Clean -and (Test-Path $Out)) { Remove-Item -Recurse -Force $Out }
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out | Out-Null }

# Let MiKTeX pull missing packages without asking.
$common = @(
    '-interaction=nonstopmode',
    '-output-directory=build',
    '-aux-directory=build',
    '--enable-installer'
)

function Invoke-Pass([string]$label) {
    Write-Host ">> $label" -ForegroundColor Cyan
    & xelatex @common "$Job.tex" | Out-Null
    # xelatex returns non-zero on error, but nonstopmode still produces a log
    return $LASTEXITCODE
}

$rc = Invoke-Pass 'xelatex pass 1'

if (-not $Quick) {
    if ($null -ne $bibtex) {
        Write-Host ">> bibtex" -ForegroundColor Cyan
        Push-Location $Out
        # so BibTeX can find ..\references.bib
        $env:BIBINPUTS = "$Root;"
        & bibtex $Job | Out-Null
        Pop-Location
    } else {
        Write-Host "bibtex not found - citations will stay as [?]" -ForegroundColor Yellow
    }
    Invoke-Pass 'xelatex pass 2' | Out-Null
    Invoke-Pass 'xelatex pass 3' | Out-Null
}

# --- collect the PDF ------------------------------------------------------
$pdf = Join-Path $Out "$Job.pdf"
if (Test-Path $pdf) {
    $size = [math]::Round((Get-Item $pdf).Length / 1KB)
    $dest = Join-Path $Root "$Job.pdf"
    try {
        Copy-Item $pdf $dest -Force -ErrorAction Stop
        Write-Host ""
        Write-Host "PDF written: $dest  ($size KB)" -ForegroundColor Green
    } catch [System.IO.IOException] {
        # Almost always: the PDF is open in a viewer, which holds a write lock.
        # The build itself is fine - say so, and point at the copy that is.
        Write-Host ""
        Write-Host "Build OK, but $Job.pdf could not be replaced - it is open in another program." -ForegroundColor Yellow
        Write-Host "Close your PDF viewer and re-run, or just open the fresh copy:" -ForegroundColor Yellow
        Write-Host "    $pdf  ($size KB)" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "No PDF produced." -ForegroundColor Red
}

# --- summarise the log ----------------------------------------------------
$log = Join-Path $Out "$Job.log"
if (Test-Path $log) {
    Write-Host ""
    Write-Host "--- log summary ---" -ForegroundColor Cyan
    python (Join-Path $Root 'checklog.py') $log
} else {
    Write-Host "No log file at $log" -ForegroundColor Red
}
