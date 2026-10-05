# Task-order spectral audit

Public research artifact for Szymon Hubert Duchowicz's paper *Beyond Pairwise Task-Order Models: A Seed-Aware Spectral Audit of Continual-Learning Landscapes*, accepted at the NeurIPS 2026 CL4FMAgents workshop. The [paper PDF](paper.pdf) and [LaTeX source](paper.tex) are the named camera-ready version. This repository also contains the supported Python package, complete scalar response landscapes, manifest-bound audit outputs, derived figures, claim checks, focused tests, and deterministic software benchmarks.

The scientific roles of the three empirical bundles are deliberately different. The complete ten-seed $K=5$ and $K=6$ landscapes provide the paper's primary evidence about approximation strength and exact degree-at-most-two support. The complete ten-seed $K=7$ landscape is included only for the post-selected exploratory within-degree allocation illustration. Its metadata schema was reconstructed retrospectively without retraining, so it is not used to extend the primary support claim or to assert a scaling law.

Start with `REPRODUCE.md`. The shortest read-only verification path is:

```text
python verify_package.py
python verify_claims.py
python -m pytest -q tests/core
```

The package contains only scalar arrays with the keys `permutations`, `values`, and `seeds`. It contains no image datasets, model checkpoints, model-training code, training logs, or raw examples. Reproducing the audit does not retrain the continual-learning models.

The manuscript is compile-ready from `paper.tex`, `references.bib`, `neurips_2026.sty`, and `figures/`. Its artifact citation points to this public repository. The reference audit bundles live under `reference_outputs/spectral_audit/`; `claim_data/` is the manuscript-facing snapshot checked by `verify_claims.py`. Every release member other than `SUPPLEMENT_MANIFEST.sha256` is bound by that deterministic top-level manifest.

The BSD 3-Clause License in `LICENSE` applies to the original software and package documentation. It does not relicense the manuscript, generated research artifacts, third-party dependencies, or upstream datasets.
