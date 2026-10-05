# Reproduction guide

Run all commands from the artifact root. The examples use PowerShell; on another platform, replace the virtual-environment interpreter path with the local equivalent.

## Checked environment

- Python 3.14.6
- NumPy 2.4.4
- Matplotlib 3.11.0
- Pillow 12.2.0
- pytest 9.1.1
- `taskorder-spectral` 0.2.0

The package supports Python 3.10 or newer. Exact checked analysis versions are in `requirements-analysis.txt`.

## Install

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

No network access is needed after installation.

## Verify package integrity and claims

```powershell
.\.venv\Scripts\python.exe verify_package.py
.\.venv\Scripts\python.exe verify_claims.py
.\.venv\Scripts\python.exe -m pytest -q tests/core
```

`verify_package.py` checks the deterministic release manifest, the scalar-only input contract, complete permutation coverage, every audit-output digest, implementation binding, benchmark hashes, the curated public release surface, and the absence of Python comment tokens. `verify_claims.py` independently derives the manuscript's displayed K=5, K=6, and exploratory K=7 values from `claim_data/`, checks spectral identities and citations, and verifies both figure assets.

## Regenerate the primary K=5 and K=6 audits

The primary results use complete permutation grids and ten whole-seed replicate landscapes. Each command validates and canonicalizes the grid, runs the exact decomposition and seed-aware inference, and writes its manifest last.

```powershell
.\.venv\Scripts\python.exe -m taskorder_spectral `
  --artifact inputs/k5_landscape.npz `
  --k 5 `
  --permutations-key permutations `
  --values-key values `
  --seeds-key seeds `
  --values-layout group-by-seed `
  --output-dir reproduced/spectral_audit/k5 `
  --high-order-min 3 `
  --allocation-target 2,2,1

.\.venv\Scripts\python.exe -m taskorder_spectral `
  --artifact inputs/k6_landscape.npz `
  --k 6 `
  --permutations-key permutations `
  --values-key values `
  --seeds-key seeds `
  --values-layout group-by-seed `
  --output-dir reproduced/spectral_audit/k6 `
  --high-order-min 3 `
  --allocation-target 3,2,1
```

Canonical input digests:

| Role | File | SHA-256 |
|---|---|---|
| Primary | `inputs/k5_landscape.npz` | `953ef62c93e13ebb17e7c69f0237e2adf9a25bbd9b39dc48822a068b0c95c5f5` |
| Primary | `inputs/k6_landscape.npz` | `0ec6cb422a16d26747846178cdd5b5b66dc5d027522a4507285ce6ee4cf12f51` |
| Exploratory only | `inputs/k7_landscape.npz` | `a99b82f818095c3fc7e533de72bb3c572c1445215a34f35148627245f9ee1f7e` |

## Regenerate the exploratory K=7 allocation audit

This command reproduces only the paper's post-selected near-hook allocation illustration. The K=7 metadata schema was reconstructed retrospectively without retraining, and its seed labels differ from K=5 and K=6. Do not interpret this bundle as a third primary support experiment, an independently selected discovery, or evidence of scaling with K.

```powershell
.\.venv\Scripts\python.exe -m taskorder_spectral `
  --artifact inputs/k7_landscape.npz `
  --k 7 `
  --permutations-key permutations `
  --values-key values `
  --seeds-key seeds `
  --values-layout group-by-seed `
  --output-dir reproduced/spectral_audit/k7 `
  --high-order-min 3 `
  --allocation-target 4,2,1
```

## Rerun the deterministic software benchmark

```powershell
.\.venv\Scripts\python.exe benchmarks/benchmark_audit_scaling.py `
  --ks 4 5 6 7 `
  --seeds 10 `
  --high-order-min 3 `
  --warmups 1 `
  --repeats 3 `
  --output-dir reproduced/benchmarks
```

The benchmark uses synthetic deterministic landscapes. It measures the in-memory audit, not continual-learning training, and its timing and peak-memory values are machine-specific.

## Compile the manuscript

With Tectonic 0.16.9:

```powershell
tectonic -X compile paper.tex --keep-logs --keep-intermediates
```

With a conventional PDFLaTeX toolchain:

```powershell
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Evidence boundary

The supplied scalar landscapes and manifests reproduce the spectral analyses in the paper. They do not reproduce model training, image-data preprocessing, or numerical-library behavior during the original experiments. The inferential unit is a whole seed landscape; permutation rows form the fixed complete design and are not treated as independent observations.
