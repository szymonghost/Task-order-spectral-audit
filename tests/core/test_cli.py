from __future__ import annotations

import csv
import itertools
import json

import numpy as np
import pytest

from taskorder_spectral.cli import main
from taskorder_spectral.io import sha256_file


def test_cli_publishes_bound_csv_bundle_from_explicit_schema(tmp_path) -> None:
    grid = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
    rng = np.random.default_rng(44)
    scores = rng.normal(size=(5, 6))
    artifact = tmp_path / "unconventional_fields.npz"
    np.savez(
        artifact,
        order_grid=grid,
        losses=scores,
        trials=np.array([3, 5, 8, 13, 21]),
    )
    output = tmp_path / "audit"
    return_code = main(
        [
            "--artifact",
            str(artifact),
            "--k",
            "3",
            "--permutations-key",
            "order_grid",
            "--values-key",
            "losses",
            "--seeds-key",
            "trials",
            "--values-layout",
            "seed-by-group",
            "--output-dir",
            str(output),
            "--high-order-min",
            "1",
        ]
    )
    assert return_code == 0
    expected_counts = {
        "partition_spectrum.csv": 3,
        "order_spectrum.csv": 3,
        "high_order_summary.csv": 1,
    }
    for name, expected_count in expected_counts.items():
        with (output / name).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected_count

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["input"]["permutations_key"] == "order_grid"
    assert manifest["input"]["values_key"] == "losses"
    assert manifest["input"]["input_values_layout"] == "seed-by-group"
    assert manifest["input"]["sha256"] == sha256_file(artifact)
    assert manifest["input"]["path"] == artifact.name
    assert manifest["input"]["path_scope"] == "basename_only"
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["diagnostics"]["parseval_passed"] is True
    for name in expected_counts:
        assert manifest["outputs"][name]["sha256"] == sha256_file(output / name)


def test_cli_publishes_single_landscape_descriptive_bundle(
    tmp_path, capsys
) -> None:
    grid = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
    artifact = tmp_path / "single.npz"
    np.savez(artifact, grid=grid, score=np.arange(6, dtype=float)[:, None])
    output = tmp_path / "descriptive"
    assert main(
        [
            "--artifact", str(artifact),
            "--k", "3",
            "--permutations-key", "grid",
            "--values-key", "score",
            "--implicit-seeds",
            "--values-layout", "group-by-seed",
            "--output-dir", str(output),
            "--high-order-min", "1",
            "--descriptive-only",
        ]
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["diagnostics"]["analysis_mode"] == "descriptive_only"
    assert manifest["diagnostics"]["seed_inference_available"] is False
    assert "seed inference=not run" in capsys.readouterr().out
    with (output / "high_order_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["signal_energy"] == "nan"
    assert row["inference_available"] == "false"


def test_cli_publishes_requested_allocation_table(tmp_path) -> None:
    grid = np.array(list(itertools.permutations(range(4))), dtype=np.int64)
    rng = np.random.default_rng(91)
    artifact = tmp_path / "allocation.npz"
    np.savez(
        artifact,
        grid=grid,
        values=rng.normal(size=(len(grid), 6)),
    )
    output = tmp_path / "allocation"
    assert main(
        [
            "--artifact", str(artifact),
            "--k", "4",
            "--permutations-key", "grid",
            "--values-key", "values",
            "--implicit-seeds",
            "--values-layout", "group-by-seed",
            "--output-dir", str(output),
            "--high-order-min", "2",
            "--allocation-target", "2,2",
        ]
    ) == 0
    with (output / "within_order_allocation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["target_partition"] == "(2, 2)"
    assert float(rows[0]["coordinate_share_within_order"]) == pytest.approx(
        4 / 13
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    record = manifest["outputs"]["within_order_allocation.csv"]
    assert record["rows"] == 1
    assert record["sha256"] == sha256_file(
        output / "within_order_allocation.csv"
    )
