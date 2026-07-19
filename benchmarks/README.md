# Benchmarks

Expanded paper evaluation for PairFold (Domains100, Long100, ablation).

```text
benchmarks/
├── sets/domains_100.json          # 100+ domain IDs + structural class
├── pdbs/                          # Cached RCSB downloads (gitignored)
├── results/
│   ├── benchmark_domains100.csv
│   ├── benchmark_long100.csv
│   ├── benchmark_ablation_summary.json
│   └── benchmark_expanded_summary.json
├── benchmark_expanded.py          # Domains100 + Long100 (10k–50k)
├── benchmark_ablation.py          # Base / look-ahead / lever / SS / full
├── plot_expanded.py
├── write_paper_tables.py          # → paper/macros_*.tex, table_*.tex
├── run_paper_benches.ps1          # Full paper pipeline
├── benchmark.py                   # Small short-domain smoke bench
└── benchmark_long.py              # Legacy tiled long stress test
```

## Paper suite (from repo root)

```bash
powershell -File benchmarks/run_paper_benches.ps1
```

Or stepwise:

```bash
python -u benchmarks/benchmark_expanded.py --domains-only
python -u benchmarks/benchmark_ablation.py
python -u benchmarks/benchmark_expanded.py --long-only --long-cases 100
python -u benchmarks/plot_expanded.py
python -u benchmarks/write_paper_tables.py
```

Requires `pip install -e ".[bench]"` (pandas, matplotlib, seaborn, BioPython, …).

## Headline numbers (current)

| Panel | n | Mean RMSD | Mean time |
|---|---:|---:|---:|
| Domains100 | 103 | 25.88 Å | 2.67 s |
| Ablation (full local) | 20 | 13.23 Å | 15.29 s |
| Long100 (10k–50k) | 100 | 26.34 Å (tile) | 0.36 s |
