# `taskorder_spectral`

This package audits a scalar response measured for every ordering of `K`
tasks. A single complete landscape supports exact descriptive decomposition;
three or more independent landscapes additionally support seed-aware
population-signal inference. The package has no knowledge of a particular
experiment directory or NPZ field name.

## Input contract

An input NPZ supplies three numeric arrays, with their keys named explicitly
on the command line:

- permutations: integer shape `(K!, K)`. Every row is a permutation of
  `0, ..., K-1`, and every permutation occurs once. Row order is arbitrary;
  validation changes it to lexicographic order before analysis.
- values: finite shape `(K!, M)` for `group-by-seed`, or `(M, K!)` for
  `seed-by-group`. Each column in canonical form is one complete landscape.
  At least one landscape is required. Seed-aware inference requires `M >= 3`;
  use `--descriptive-only` otherwise.
- seeds: optional one-dimensional array of `M` unique IDs. Use
  `--implicit-seeds` only when column indices are the intended IDs.

Rows of the permutation grid are a complete fixed design. Statistical
resampling, jackknife intervals, and sign randomization all operate on whole
seed landscapes.

## Command line

After an editable install (`python -m pip install -e .`), run:

```text
taskorder-spectral \
  --artifact PATH/landscape.npz \
  --k 5 \
  --permutations-key order_grid \
  --values-key response_by_seed \
  --seeds-key seed_ids \
  --values-layout group-by-seed \
  --output-dir PATH/audit \
  --high-order-min 3
```

For one realized landscape, store values with shape `(K!, 1)` and run:

```text
taskorder-spectral \
  --artifact PATH/one_landscape.npz \
  --k 5 \
  --permutations-key order_grid \
  --values-key response \
  --implicit-seeds \
  --values-layout group-by-seed \
  --output-dir PATH/descriptive-audit \
  --high-order-min 3 \
  --descriptive-only
```

This mode reports plug-in partition and order energies, within-order shares,
and cumulative full-grid projection fractions. Signal U-statistics,
confidence intervals, support decisions, and seed-randomization p-values are
explicitly unavailable rather than being estimated from permutation rows.

The command publishes `partition_spectrum.csv`, `order_spectrum.csv`, and
`high_order_summary.csv`; requested targets additionally produce
`within_order_allocation.csv`. It then writes `manifest.json` last. The
manifest binds the tables to the input artifact, explicit NPZ schema,
configuration, package implementation hashes, runtime versions, and
Parseval/Burnside checks.
The partition table reports the isotropic within-order baseline using
dimension-squared coordinate shares, alongside observed plug-in and
cross-seed signal-energy shares.

## Advanced diagnostics: within-order allocation

Add a repeatable `--allocation-target` option only when a particular partition
was chosen independently of the observed landscape. For example,
`--allocation-target 2,2,1` requests an order-three contrast at K=5 and adds
`within_order_allocation.csv` to the output bundle.

Each requested partition is contrasted against its squared-dimension
coordinate share within the same interaction order. When multiple targets are
requested, the output includes Holm-adjusted p-values across exactly that
requested family. This adjustment does not correct an earlier data-dependent
choice of targets.

Allocation inference uses the unbiased linear contrast
`E_target - q * E_order`, where `q` is the target's squared-dimension share.
The target and full same-order denominator are recomputed in every
delete-one-seed replicate. Its jackknife-t calibration is approximate. The
coefficient-vector sign-flip test is not reused for this composite allocation
null.

## Python API

```python
from taskorder_spectral import (
    AuditConfig,
    analyze_landscape,
    load_npz_landscape,
    publish_audit,
)

landscape = load_npz_landscape(
    "landscape.npz",
    K=5,
    permutations_key="order_grid",
    values_key="response_by_seed",
    seeds_key="seed_ids",
    values_layout="group-by-seed",
)
result = analyze_landscape(
    landscape,
    AuditConfig(high_order_min=3),
)
manifest = publish_audit(result, "audit-output")
```

Set `AuditConfig(descriptive_only=True)` for one landscape or whenever only
finite-landscape decomposition is desired. Advanced allocation contrasts use
`AuditConfig(allocation_targets=((2, 2, 1),))` at K=5.
