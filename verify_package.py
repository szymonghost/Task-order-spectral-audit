"""Verify the public CL4FMAgents release and all published hash chains."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import statistics
import tokenize
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
AUDITS = ROOT / "reference_outputs" / "spectral_audit"
CLAIMS = ROOT / "claim_data"
EMPIRICAL_KS = (5, 6, 7)
EXPECTED_KEYS = {"permutations", "values", "seeds"}
CORE_OUTPUTS = {
    "high_order_summary.csv",
    "order_spectrum.csv",
    "partition_spectrum.csv",
    "within_order_allocation.csv",
}
ALLOCATION_TARGETS = {
    5: [2, 2, 1],
    6: [3, 2, 1],
    7: [4, 2, 1],
}
GENERATED_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "build", "dist"}
GENERATED_FILES = {
    "paper.aux",
    "paper.bbl",
    "paper.blg",
    "paper.fdb_latexmk",
    "paper.fls",
    "paper.log",
    "paper.out",
    "paper.toc",
}


def is_generated(relative: Path) -> bool:
    return (
        any(part in GENERATED_DIRS or part.endswith(".egg-info") for part in relative.parts)
        or relative.name in GENERATED_FILES
        or relative.name.endswith((".pyc", ".pyo", ".synctex.gz"))
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_input(k: int) -> None:
    path = INPUTS / f"k{k}_landscape.npz"
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_KEYS:
            raise RuntimeError(f"K={k} input is not scalar-only: {archive.files}")
        permutations = archive["permutations"]
        values = archive["values"]
        seeds = archive["seeds"]
    group_order = math.factorial(k)
    if permutations.shape != (group_order, k):
        raise RuntimeError(f"K={k} permutation shape is invalid: {permutations.shape}")
    canonical = np.asarray(list(itertools.permutations(range(k))), dtype=permutations.dtype)
    if not np.array_equal(permutations, canonical):
        raise RuntimeError(f"K={k} grid is not the complete lexicographic permutation grid")
    if values.shape != (group_order, len(seeds)):
        raise RuntimeError(f"K={k} response layout is invalid: {values.shape}")
    if len(seeds) != 10 or len(np.unique(seeds)) != len(seeds):
        raise RuntimeError(f"K={k} seed identifiers are invalid")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise RuntimeError(f"K={k} responses are not finite scalars")


def verify_bundle(k: int) -> None:
    bundle = AUDITS / f"k{k}"
    members = {path.name for path in bundle.iterdir() if path.is_file()}
    if members != CORE_OUTPUTS | {"manifest.json"}:
        raise RuntimeError(f"K={k} audit membership is invalid: {sorted(members)}")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    input_path = INPUTS / f"k{k}_landscape.npz"
    input_record = manifest.get("input", {})
    if input_record.get("path") != input_path.name:
        raise RuntimeError(f"K={k} manifest input path is not the canonical basename")
    if input_record.get("path_scope") != "basename_only":
        raise RuntimeError(f"K={k} manifest input path scope is not basename-only")
    if input_record.get("sha256") != sha256(input_path):
        raise RuntimeError(f"K={k} input hash mismatch")
    expected_input_fields = {
        "K": k,
        "group_order": math.factorial(k),
        "permutations_key": "permutations",
        "values_key": "values",
        "seeds_key": "seeds",
        "n_seeds": 10,
    }
    for name, expected in expected_input_fields.items():
        if input_record.get(name) != expected:
            raise RuntimeError(f"K={k} manifest input field mismatch: {name}")
    if manifest.get("config", {}).get("allocation_targets") != [ALLOCATION_TARGETS[k]]:
        raise RuntimeError(f"K={k} allocation target does not match the documented diagnostic")
    diagnostics = manifest.get("diagnostics", {})
    if diagnostics.get("parseval_passed") is not True:
        raise RuntimeError(f"K={k} manifest lacks a Parseval pass")
    if diagnostics.get("burnside_sum_d2") != math.factorial(k):
        raise RuntimeError(f"K={k} Burnside dimension check failed")
    outputs = manifest.get("outputs", {})
    if set(outputs) != CORE_OUTPUTS:
        raise RuntimeError(f"K={k} manifest output membership is invalid")
    for name, metadata in outputs.items():
        path = bundle / name
        if path.stat().st_size != metadata.get("bytes") or sha256(path) != metadata.get("sha256"):
            raise RuntimeError(f"K={k} output hash mismatch: {name}")
        if len(csv_rows(path)) != metadata.get("rows"):
            raise RuntimeError(f"K={k} output row count mismatch: {name}")
    implementation = ROOT / "src" / "taskorder_spectral"
    implementation_hashes = manifest.get("implementation_sha256", {})
    expected_implementation_names = {path.name for path in implementation.glob("*.py")}
    if set(implementation_hashes) != expected_implementation_names:
        raise RuntimeError(f"K={k} implementation membership mismatch")
    for name, expected in implementation_hashes.items():
        if sha256(implementation / name) != expected:
            raise RuntimeError(f"K={k} implementation hash mismatch: {name}")
    claim_bundle = CLAIMS / f"k{k}"
    claim_members = {path.name for path in claim_bundle.iterdir() if path.is_file()}
    if claim_members != members:
        raise RuntimeError(f"K={k} claim-data membership differs from the reference bundle")
    for name in members:
        if sha256(claim_bundle / name) != sha256(bundle / name):
            raise RuntimeError(f"K={k} claim-data snapshot mismatch: {name}")


def verify_release_scope() -> None:
    input_names = {path.name for path in INPUTS.iterdir() if path.is_file()}
    expected_inputs = {f"k{k}_landscape.npz" for k in EMPIRICAL_KS}
    if input_names != expected_inputs:
        raise RuntimeError(f"unexpected empirical inputs: {sorted(input_names)}")
    audit_names = {path.name for path in AUDITS.iterdir() if path.is_dir()}
    expected_audits = {f"k{k}" for k in EMPIRICAL_KS}
    if audit_names != expected_audits:
        raise RuntimeError(f"unexpected empirical audit bundles: {sorted(audit_names)}")
    required_claim_extras = {"hookshape_svd_summary.csv", "audit_scaling_summary.csv"}
    claim_extras = {path.name for path in CLAIMS.iterdir() if path.is_file()}
    if claim_extras != required_claim_extras:
        raise RuntimeError(f"unexpected top-level claim data: {sorted(claim_extras)}")


def verify_benchmark() -> None:
    results = ROOT / "benchmarks" / "results"
    summary_path = results / "audit_scaling_summary.csv"
    runs_path = results / "audit_scaling_runs.csv"
    summary = {int(row["K"]): row for row in csv_rows(summary_path)}
    if set(summary) != {4, 5, 6, 7}:
        raise RuntimeError(f"unexpected benchmark K values: {sorted(summary)}")
    script_hash = sha256(ROOT / "benchmarks" / "benchmark_audit_scaling.py")
    runs = csv_rows(runs_path)
    if len(runs) != 12:
        raise RuntimeError(f"unexpected measured benchmark row count: {len(runs)}")
    for k, row in summary.items():
        if int(row["group_order"]) != math.factorial(k):
            raise RuntimeError(f"K={k} benchmark group size mismatch")
        if row["benchmark_script_sha256"] != script_hash:
            raise RuntimeError(f"K={k} benchmark script hash mismatch")
        if row["parseval_passed_all"] != "true" or row["burnside_passed_all"] != "true":
            raise RuntimeError(f"K={k} benchmark diagnostics failed")
        selected = [entry for entry in runs if int(entry["K"]) == k]
        if len(selected) != int(row["measured_repeats"]):
            raise RuntimeError(f"K={k} benchmark repeat count mismatch")
        if any(entry["phase"] != "measured" for entry in selected):
            raise RuntimeError(f"K={k} benchmark includes a nonmeasured row")
        if any(entry["benchmark_script_sha256"] != script_hash for entry in selected):
            raise RuntimeError(f"K={k} benchmark run script hash mismatch")
        if {entry["input_sha256"] for entry in selected} != {row["input_sha256"]}:
            raise RuntimeError(f"K={k} benchmark input hash mismatch")
        if {entry["result_sha256"] for entry in selected} != {row["result_sha256"]}:
            raise RuntimeError(f"K={k} benchmark result hash mismatch")
        if any(entry["parseval_passed"] != "true" or entry["burnside_passed"] != "true" for entry in selected):
            raise RuntimeError(f"K={k} benchmark run diagnostics failed")
        walls = [float(entry["wall_seconds"]) for entry in selected]
        peaks = [float(entry["peak_rss_mib"]) for entry in selected]
        expected_numbers = {
            "wall_seconds_median": statistics.median(walls),
            "wall_seconds_min": min(walls),
            "wall_seconds_max": max(walls),
            "peak_rss_mib_median": statistics.median(peaks),
            "peak_rss_mib_max": max(peaks),
        }
        for name, expected in expected_numbers.items():
            if not math.isclose(float(row[name]), expected, rel_tol=0.0, abs_tol=5e-8):
                raise RuntimeError(f"K={k} benchmark summary mismatch: {name}")
    if sha256(CLAIMS / "audit_scaling_summary.csv") != sha256(summary_path):
        raise RuntimeError("manuscript benchmark snapshot differs from the packaged summary")


def verify_python_sources() -> None:
    roots = (ROOT / "src", ROOT / "tests", ROOT / "benchmarks", ROOT)
    paths = set()
    for base in roots:
        if base == ROOT:
            paths.update(base.glob("*.py"))
        else:
            paths.update(base.rglob("*.py"))
    for path in sorted(paths):
        with tokenize.open(path) as handle:
            tokens = tokenize.generate_tokens(handle.readline)
            comments = [token for token in tokens if token.type == tokenize.COMMENT]
        if comments:
            locations = ", ".join(str(token.start[0]) for token in comments[:5])
            raise RuntimeError(f"Python comment token found in {path.relative_to(ROOT)} at line(s) {locations}")


def verify_release_surface() -> None:
    forbidden_parts = {
        "__pycache__",
        ".git",
        "training",
        "train",
        "logs",
        "log",
        "cache",
        "backup",
        "checkpoints",
        "raw_data",
    }
    text_suffixes = {".py", ".md", ".tex", ".bib", ".sty", ".toml", ".txt", ".json", ".csv"}
    forbidden_tokens = (
        "c:" + "\\users\\",
        "/" + "users/",
        "one" + "drive",
        "file" + "://",
        "pre" + "register",
        "pre-" + "register",
        "pre" + "specified",
        "pre-" + "specified",
    )
    allowed_binary = {
        "inputs/k5_landscape.npz",
        "inputs/k6_landscape.npz",
        "inputs/k7_landscape.npz",
        "figures/figure1_population_signal.png",
        "figures/figure_2_spectrum_k6.pdf",
        "paper.pdf",
    }
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if is_generated(relative):
            continue
        lowered_parts = {part.lower() for part in relative.parts}
        if forbidden_parts & lowered_parts:
            raise RuntimeError(f"forbidden artifact path: {relative.as_posix()}")
        if not path.is_file():
            continue
        relative_text = relative.as_posix()
        if path.suffix.lower() in {".npz", ".png", ".pdf"} and relative_text not in allowed_binary:
            raise RuntimeError(f"unexpected binary artifact: {relative_text}")
        if path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        found = [token for token in forbidden_tokens if token in text]
        if found:
            raise RuntimeError(f"forbidden path or protocol text in {relative_text}")


def verify_artifact_manifest() -> None:
    manifest_path = ROOT / "SUPPLEMENT_MANIFEST.sha256"
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected:
            raise RuntimeError(f"duplicate artifact-manifest member: {relative}")
        expected[relative] = digest
    actual = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not is_generated(path.relative_to(ROOT))
    }
    if expected.keys() != actual.keys():
        missing = sorted(actual.keys() - expected.keys())
        extra = sorted(expected.keys() - actual.keys())
        raise RuntimeError(f"artifact manifest membership mismatch; missing={missing}, extra={extra}")
    mismatched = [name for name in expected if expected[name] != actual[name]]
    if mismatched:
        raise RuntimeError(f"artifact manifest hash mismatch: {mismatched}")


def main() -> None:
    verify_release_scope()
    for k in EMPIRICAL_KS:
        verify_input(k)
        verify_bundle(k)
    verify_benchmark()
    verify_python_sources()
    verify_release_surface()
    verify_artifact_manifest()
    print(
        "Verified scalar-only K=5/K=6 primary inputs, the exploratory K=7 input, "
        "all audit and implementation hashes, benchmark chains, public release surface, "
        "comment-free Python sources, and every artifact member."
    )


if __name__ == "__main__":
    main()
