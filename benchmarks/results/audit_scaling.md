# Synthetic software scaling of `taskorder_spectral`

Generated: 2026-08-07T23:38:09.497976+00:00

## Benchmark results

| K | Complete orders (K!) | Irreps | Largest irrep d | Median wall time (s) | Min-max (s) | Median peak RSS (MiB) | Max peak RSS (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 24 | 5 | 3 | 0.017 | 0.016-0.017 | 36.0 | 36.2 |
| 5 | 120 | 7 | 6 | 0.029 | 0.028-0.030 | 36.4 | 36.5 |
| 6 | 720 | 11 | 16 | 0.126 | 0.119-0.126 | 40.9 | 41.8 |
| 7 | 5040 | 15 | 35 | 1.406 | 1.401-1.432 | 154.9 | 154.9 |

Wall time is the median of isolated measured processes; peak RSS is the operating-system process peak, so it includes the interpreter, imported package, synthetic input, and retained audit result.

## Protocol

- Configuration: M=10 deterministic synthetic landscapes with complete permutation grids, `high_order_min=3`, seed-aware inference, and 512 exact whole-seed sign patterns.
- Allocation targets: none. This benchmarks the common core audit at every K and does not incorporate a data-selected exploratory target.
- Timed region: `CompleteLandscape.from_arrays` plus `analyze_landscape`. Python/NumPy import, deterministic input construction, and CSV/manifest publication are excluded.
- Repetition: 1 unreported system warm-up(s) and 3 measured repeat(s) per K; every repeat starts in a fresh Python process.
- Numerical threading: `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` were set to 1 before each worker started; `PYTHONHASHSEED=0`.
- Memory counter: Windows GetProcessMemoryInfo PeakWorkingSetSize.
- Validation: every measured run passed Burnside and Parseval checks. Input and full scientific-result SHA-256 digests agreed across repeats within each K.
- Benchmark script SHA-256: `a08d6b9170f387363235f736b1f94a661d33449a536f23e82dade9a3b1450c8c`.

## Environment

- Platform: `Windows-11-10.0.26200-SP0`
- Processor: `AMD Ryzen 5 6600H with Radeon Graphics`
- Logical CPUs: 12
- Python: 3.14.6
- NumPy: 2.4.4
- `taskorder-spectral`: 0.2.0

These are machine-specific engineering measurements, not asymptotic complexity claims. The raw repeat-level records are in `audit_scaling_runs.csv`; the exact paper table is in `audit_scaling_summary.csv`.
