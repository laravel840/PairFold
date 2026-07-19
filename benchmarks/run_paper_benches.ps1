# Sequential paper benchmark pipeline (Domains100 -> ablation -> Long100 -> plots/tables)
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "=== Domains100 ===" -ForegroundColor Cyan
python -u benchmarks/benchmark_expanded.py --domains-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Ablation (20 domains x 5 variants) ===" -ForegroundColor Cyan
python -u benchmarks/benchmark_ablation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Long100 (10k-50k) ===" -ForegroundColor Cyan
python -u benchmarks/benchmark_expanded.py --long-only --long-cases 100
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Plots + paper tables ===" -ForegroundColor Cyan
python -u benchmarks/plot_expanded.py
python -u benchmarks/write_paper_tables.py
Write-Host "Done." -ForegroundColor Green
