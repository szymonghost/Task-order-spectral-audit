# Public release contents

- `src/taskorder_spectral/`: supported BSD-3-Clause Python package.
- `tests/core/`: focused package regression suite.
- `inputs/`: scalar-only complete K=5, K=6, and K=7 landscapes. K=5 and K=6 are primary; K=7 is exploratory only and carries the retrospective metadata qualification stated in the paper and reproduction guide.
- `reference_outputs/spectral_audit/`: manifest-bound audit bundles for the three supplied scalar inputs.
- `claim_data/`: manuscript-facing audit snapshots, descriptive near-hook SVD summary, and benchmark summary consumed by `verify_claims.py`.
- `paper.pdf`: named camera-ready workshop manuscript.
- `paper.tex`, `references.bib`, and `neurips_2026.sty`: compile-ready named manuscript source.
- `figures/`: the two derived figure assets used by the manuscript.
- `verify_claims.py`: independent displayed-claim, spectral-identity, citation, and figure check.
- `verify_package.py`: release membership, hash-chain, scalar-input, implementation, benchmark, source-surface, and Python-comment check.
- `benchmarks/`: deterministic synthetic K=4 through K=7 audit benchmark and recorded results.
- `REPRODUCE.md`: exact verification, audit, benchmark, test, and manuscript-build commands.
- `SUPPLEMENT_MANIFEST.sha256`: deterministic SHA-256 digest of every other artifact member.
- `.gitignore`: excludes local environments, test caches, and LaTeX build byproducts from future commits.

The release excludes raw image datasets, model checkpoints, training code, training logs, local filesystem paths, and earlier manuscript versions.
