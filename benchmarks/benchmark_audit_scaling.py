"""Reproducible synthetic scaling benchmark for the released spectral audit.

The coordinator starts every warm-up and measured repeat in a fresh Python
process.  The timed region covers public input validation plus
``analyze_landscape``; imports, deterministic input construction, and output
publication are deliberately outside the timer.  Peak process resident memory
is read from the operating system after the audit.  On Windows this is the
process lifetime ``PeakWorkingSetSize`` reported by ``GetProcessMemoryInfo``.

Run from the repository root with::

    py -3 benchmarks/benchmark_audit_scaling.py

The default configuration benchmarks K=4,5,6,7 with ten synthetic complete
landscapes, one system warm-up and three isolated measured repeats per K.  It
forces the common numerical thread-count environment variables to one before
each child process starts.  The benchmark intentionally requests no optional
allocation target, so every K executes the same core seed-aware audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
RESULT_PREFIX = "TASKORDER_BENCHMARK_RESULT="
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--high-order-min", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--worker-k", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeat", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-phase", default="measured", help=argparse.SUPPRESS)
    return parser


def _validate_options(args: argparse.Namespace) -> None:
    if args.seeds < 3:
        raise ValueError("--seeds must be at least 3 for seed-aware inference")
    if args.high_order_min < 1:
        raise ValueError("--high-order-min must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be nonnegative")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if not args.ks or len(set(args.ks)) != len(args.ks):
        raise ValueError("--ks must contain distinct values")
    for K in args.ks:
        if K < 2:
            raise ValueError("every K must be at least 2")
        if args.high_order_min > K - 1:
            raise ValueError(
                f"--high-order-min={args.high_order_min} is invalid for K={K}"
            )


def _synthetic_arrays(K: int, n_seeds: int):
    """Return a deterministic, nondegenerate complete replicated landscape."""

    import itertools

    import numpy as np

    grid = np.asarray(list(itertools.permutations(range(K))), dtype=np.int64)
    tasks = grid.astype(np.float64) + 1.0
    positions = np.arange(1.0, K + 1.0)

    position_component = np.sum(
        (positions - positions.mean()) * (tasks**2 + 0.3 * tasks), axis=1
    )
    transition_component = np.sum(
        (tasks[:, :-1] + 0.5) * (tasks[:, 1:] + 1.5), axis=1
    )
    triple_component = np.sum(
        tasks[:, :-2]
        * (tasks[:, 1:-1] + 0.25)
        * (tasks[:, 2:] + 0.75),
        axis=1,
    )

    def standardized(vector):
        centered = vector - vector.mean()
        return centered / np.sqrt(np.mean(centered**2))

    first = standardized(position_component)
    second = standardized(transition_component)
    third = standardized(triple_component)
    rank = np.arange(1.0, len(grid) + 1.0)
    values = np.empty((len(grid), n_seeds), dtype=np.float64)
    for seed in range(n_seeds):
        phase = float(seed + 1)
        deterministic_seed_variation = (
            0.075 * np.sin(rank * phase * math.sqrt(2.0))
            + 0.035 * np.cos(rank * (phase + 0.5) * math.sqrt(3.0))
        )
        values[:, seed] = (
            first
            + 0.45 * second
            + 0.20 * third
            + 0.025 * (phase - (n_seeds + 1.0) / 2.0) * first
            + deterministic_seed_variation
        )
    return grid, values


def _array_digest(K: int, n_seeds: int, grid, values) -> str:
    digest = hashlib.sha256()
    digest.update(f"K={K};M={n_seeds};".encode("ascii"))
    digest.update(grid.dtype.str.encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(grid.tobytes(order="C"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Normalize NumPy scalars and nonfinite floats for a strict digest."""

    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _result_digest(result) -> str:
    payload = {
        "partition_rows": result.partition_rows,
        "order_rows": result.order_rows,
        "high_order_rows": result.high_order_rows,
        "allocation_rows": result.allocation_rows,
        "diagnostics": result.diagnostics,
    }
    encoded = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _windows_process_memory() -> tuple[int, int]:
    """Return current and lifetime-peak process working set in bytes."""

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _process_memory() -> tuple[int | None, int, str]:
    if os.name == "nt":
        current, peak = _windows_process_memory()
        return current, peak, "Windows GetProcessMemoryInfo PeakWorkingSetSize"

    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak_bytes = maximum
    else:
        peak_bytes = maximum * 1024
    return None, peak_bytes, "resource.getrusage(RUSAGE_SELF).ru_maxrss"


def _processor_name() -> str:
    """Return a human-readable CPU model without adding a dependency."""

    if os.name == "nt":
        try:
            import winreg

            registry_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            cleaned = " ".join(str(name).split())
            if cleaned:
                return cleaned
        except OSError:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())


def _worker_record(args: argparse.Namespace) -> dict[str, Any]:
    import gc
    import time

    import numpy as np

    source_root = REPOSITORY_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import taskorder_spectral
    from taskorder_spectral import AuditConfig, CompleteLandscape, analyze_landscape
    from taskorder_spectral.irreps import partition_catalog

    K = int(args.worker_k)
    grid, values = _synthetic_arrays(K, args.seeds)
    input_sha256 = _array_digest(K, args.seeds, grid, values)
    config = AuditConfig(
        high_order_min=args.high_order_min,
        allocation_targets=(),
    )
    gc.collect()
    pre_current, pre_peak, memory_source = _process_memory()

    start = time.perf_counter()
    landscape = CompleteLandscape.from_arrays(
        K=K,
        permutations=grid,
        values=values,
        seed_ids=tuple(range(args.seeds)),
        values_layout="group-by-seed",
    )
    result = analyze_landscape(landscape, config)
    wall_seconds = time.perf_counter() - start

    post_current, peak_bytes, post_memory_source = _process_memory()
    if post_memory_source != memory_source:
        raise RuntimeError("memory counter source changed within a worker")
    catalog = partition_catalog(K)
    diagnostics = result.diagnostics
    parseval_errors = (
        float(diagnostics["plugin_parseval_absolute_error"]),
        float(diagnostics["nontrivial_parseval_absolute_error"]),
        float(diagnostics["per_seed_parseval_max_absolute_error"]),
    )
    high_row = result.high_order_rows[0]
    return {
        "phase": args.worker_phase,
        "repeat": int(args.worker_repeat),
        "K": K,
        "group_order": math.factorial(K),
        "n_seeds": args.seeds,
        "high_order_min": args.high_order_min,
        "allocation_target_count": 0,
        "partition_count": len(catalog),
        "largest_irrep_dimension": max(
            int(component["dimension"]) for component in catalog
        ),
        "exact_sign_patterns": int(high_row["n_sign_patterns"]),
        "sign_patterns_exact": bool(high_row["seed_signflip_exact"]),
        "wall_seconds": wall_seconds,
        "pre_audit_rss_mib": (
            pre_current / 2**20 if pre_current is not None else None
        ),
        "pre_audit_lifetime_peak_rss_mib": pre_peak / 2**20,
        "post_audit_rss_mib": (
            post_current / 2**20 if post_current is not None else None
        ),
        "peak_rss_mib": peak_bytes / 2**20,
        "peak_rss_source": memory_source,
        "input_sha256": input_sha256,
        "result_sha256": _result_digest(result),
        "benchmark_script_sha256": _file_digest(Path(__file__).resolve()),
        "parseval_passed": bool(diagnostics["parseval_passed"]),
        "burnside_passed": bool(
            diagnostics["burnside_sum_d2"] == diagnostics["burnside_expected"]
        ),
        "parseval_max_absolute_error": max(parseval_errors),
        "parseval_tolerance": float(diagnostics["parseval_tolerance"]),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "package_version": taskorder_spectral.__version__,
        "platform": platform.platform(),
        "processor": _processor_name(),
        "logical_cpus": os.cpu_count(),
        "python_executable": Path(sys.executable).name,
        **{name.lower(): os.environ.get(name, "unset") for name in THREAD_ENVIRONMENT},
    }


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in THREAD_ENVIRONMENT:
        environment[name] = "1"
    environment["PYTHONHASHSEED"] = "0"
    source_root = str(REPOSITORY_ROOT / "src")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root + os.pathsep + prior_pythonpath
        if prior_pythonpath
        else source_root
    )
    return environment


def _run_child(
    *,
    K: int,
    n_seeds: int,
    high_order_min: int,
    phase: str,
    repeat: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-k",
        str(K),
        "--worker-phase",
        phase,
        "--worker-repeat",
        str(repeat),
        "--seeds",
        str(n_seeds),
        "--high-order-min",
        str(high_order_min),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=_child_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    result_lines = [
        line[len(RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    ]
    if len(result_lines) != 1:
        raise RuntimeError(
            "worker did not emit exactly one result record\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(result_lines[0])


def _validate_records(records: list[dict[str, Any]], repeats: int) -> None:
    if not records:
        raise RuntimeError("benchmark produced no measured records")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["K"]), []).append(record)
        if record["phase"] != "measured":
            raise RuntimeError("measured output contains a non-measured record")
        if not record["parseval_passed"] or not record["burnside_passed"]:
            raise RuntimeError(f"algebraic validation failed for K={record['K']}")
        if not (0.0 < float(record["wall_seconds"])):
            raise RuntimeError("worker reported a nonpositive runtime")
        if not (float(record["peak_rss_mib"]) > 0.0):
            raise RuntimeError("worker reported a nonpositive peak RSS")
        if float(record["parseval_max_absolute_error"]) > float(
            record["parseval_tolerance"]
        ):
            raise RuntimeError("Parseval error exceeds its recorded tolerance")
    for K, rows in grouped.items():
        if len(rows) != repeats:
            raise RuntimeError(f"K={K} has {len(rows)} records, expected {repeats}")
        if len({row["input_sha256"] for row in rows}) != 1:
            raise RuntimeError(f"K={K} input digest differs across repeats")
        if len({row["result_sha256"] for row in rows}) != 1:
            raise RuntimeError(f"K={K} scientific result differs across repeats")
        if len({row["peak_rss_source"] for row in rows}) != 1:
            raise RuntimeError(f"K={K} memory-counter source differs across repeats")
        if len({row["benchmark_script_sha256"] for row in rows}) != 1:
            raise RuntimeError(f"K={K} benchmark script changed across repeats")


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["K"]), []).append(record)
    summaries: list[dict[str, Any]] = []
    for K in sorted(grouped):
        rows = grouped[K]
        runtimes = [float(row["wall_seconds"]) for row in rows]
        memories = [float(row["peak_rss_mib"]) for row in rows]
        template = rows[0]
        summaries.append(
            {
                "K": K,
                "group_order": int(template["group_order"]),
                "partition_count": int(template["partition_count"]),
                "largest_irrep_dimension": int(template["largest_irrep_dimension"]),
                "n_seeds": int(template["n_seeds"]),
                "high_order_min": int(template["high_order_min"]),
                "allocation_target_count": int(template["allocation_target_count"]),
                "exact_sign_patterns": int(template["exact_sign_patterns"]),
                "sign_patterns_exact": bool(template["sign_patterns_exact"]),
                "measured_repeats": len(rows),
                "wall_seconds_median": statistics.median(runtimes),
                "wall_seconds_min": min(runtimes),
                "wall_seconds_max": max(runtimes),
                "peak_rss_mib_median": statistics.median(memories),
                "peak_rss_mib_max": max(memories),
                "parseval_max_absolute_error": max(
                    float(row["parseval_max_absolute_error"]) for row in rows
                ),
                "parseval_passed_all": all(row["parseval_passed"] for row in rows),
                "burnside_passed_all": all(row["burnside_passed"] for row in rows),
                "input_sha256": template["input_sha256"],
                "result_sha256": template["result_sha256"],
                "benchmark_script_sha256": template["benchmark_script_sha256"],
                "peak_rss_source": template["peak_rss_source"],
                "python_version": template["python_version"],
                "numpy_version": template["numpy_version"],
                "package_version": template["package_version"],
                "platform": template["platform"],
                "processor": template["processor"],
                "logical_cpus": template["logical_cpus"],
            }
        )
    return summaries


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".9g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _markdown_report(
    summaries: list[dict[str, Any]],
    *,
    warmups: int,
    generated_at: str,
) -> str:
    first = summaries[0]
    sign_method = "exact" if first["sign_patterns_exact"] else "Monte Carlo"
    lines = [
        "# Synthetic software scaling of `taskorder_spectral`",
        "",
        f"Generated: {generated_at}",
        "",
        "## Benchmark results",
        "",
        "| K | Complete orders (K!) | Irreps | Largest irrep d | Median wall time (s) | Min-max (s) | Median peak RSS (MiB) | Max peak RSS (MiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {K} | {group_order} | {partition_count} | "
            "{largest_irrep_dimension} | {wall_seconds_median:.3f} | "
            "{wall_seconds_min:.3f}-{wall_seconds_max:.3f} | "
            "{peak_rss_mib_median:.1f} | {peak_rss_mib_max:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Wall time is the median of isolated measured processes; peak RSS is the "
            "operating-system process peak, so it includes the interpreter, imported "
            "package, synthetic input, and retained audit result.",
            "",
            "## Protocol",
            "",
            f"- Configuration: M={first['n_seeds']} deterministic synthetic landscapes "
            "with complete permutation grids, "
            f"`high_order_min={first['high_order_min']}`, seed-aware inference, and "
            f"{first['exact_sign_patterns']} {sign_method} whole-seed sign patterns.",
            "- Allocation targets: none. This benchmarks the common core audit at every "
            "K and does not incorporate a data-selected exploratory target.",
            "- Timed region: `CompleteLandscape.from_arrays` plus "
            "`analyze_landscape`. Python/NumPy import, deterministic input construction, "
            "and CSV/manifest publication are excluded.",
            f"- Repetition: {warmups} unreported system warm-up(s) and "
            f"{first['measured_repeats']} measured repeat(s) per K; every repeat starts "
            "in a fresh Python process.",
            "- Numerical threading: `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, "
            "`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and "
            "`VECLIB_MAXIMUM_THREADS` were set to 1 before each worker started; "
            "`PYTHONHASHSEED=0`.",
            f"- Memory counter: {first['peak_rss_source']}.",
            "- Validation: every measured run passed Burnside and Parseval checks. "
            "Input and full scientific-result SHA-256 digests agreed across repeats "
            "within each K.",
            f"- Benchmark script SHA-256: `{first['benchmark_script_sha256']}`.",
            "",
            "## Environment",
            "",
            f"- Platform: `{first['platform']}`",
            f"- Processor: `{first['processor']}`",
            f"- Logical CPUs: {first['logical_cpus']}",
            f"- Python: {first['python_version']}",
            f"- NumPy: {first['numpy_version']}",
            f"- `taskorder-spectral`: {first['package_version']}",
            "",
            "These are machine-specific engineering measurements, not asymptotic "
            "complexity claims. The raw repeat-level records are in "
            "`audit_scaling_runs.csv`; the exact paper table is in "
            "`audit_scaling_summary.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _coordinator(args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    for K in args.ks:
        for warmup in range(args.warmups):
            print(f"K={K}: warm-up {warmup + 1}/{args.warmups}", flush=True)
            _run_child(
                K=K,
                n_seeds=args.seeds,
                high_order_min=args.high_order_min,
                phase="warmup",
                repeat=warmup,
            )
        for repeat in range(args.repeats):
            print(f"K={K}: measured {repeat + 1}/{args.repeats}", flush=True)
            records.append(
                _run_child(
                    K=K,
                    n_seeds=args.seeds,
                    high_order_min=args.high_order_min,
                    phase="measured",
                    repeat=repeat,
                )
            )

    _validate_records(records, args.repeats)
    summaries = _summary_rows(records)
    output_dir = args.output_dir.resolve()
    generated_at = datetime.now(timezone.utc).isoformat()
    _atomic_csv(output_dir / "audit_scaling_runs.csv", records)
    _atomic_csv(output_dir / "audit_scaling_summary.csv", summaries)
    _atomic_text(
        output_dir / "audit_scaling.md",
        _markdown_report(
            summaries,
            warmups=args.warmups,
            generated_at=generated_at,
        ),
    )
    print(f"Validated {len(records)} measured runs; wrote outputs to {output_dir}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_options(args)
    if args.worker_k is not None:
        record = _worker_record(args)
        print(RESULT_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False))
        return 0
    return _coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
