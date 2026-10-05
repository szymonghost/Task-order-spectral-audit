from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import pytest

from taskorder_spectral import AuditConfig, CompleteLandscape, analyze_landscape
from taskorder_spectral.irreps import build_irrep


ROOT = Path(__file__).resolve().parents[2]


def permutation_grid(K: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(K))), dtype=np.int64)


def test_pure_irrep_signal_is_assigned_to_the_correct_partition() -> None:
    grid = permutation_grid(3)
    target = build_irrep((2, 1), grid)[:, 0, 0]
    values = np.repeat(target[:, None], 6, axis=1)
    landscape = CompleteLandscape.from_arrays(
        K=3,
        permutations=grid,
        values=values,
    )
    result = analyze_landscape(
        landscape,
        AuditConfig(high_order_min=1),
    )
    rows = {row["partition"]: row for row in result.partition_rows}
    assert rows["(2, 1)"]["plugin_energy"] == pytest.approx(0.5, abs=1e-14)
    assert rows["(2, 1)"]["signal_energy"] == pytest.approx(0.5, abs=1e-14)
    assert rows["(2, 1)"]["seed_signflip_p_raw"] == pytest.approx(1 / 32)
    assert rows["(3,)"]["plugin_energy"] == pytest.approx(0.0, abs=1e-28)
    assert rows["(1, 1, 1)"]["plugin_energy"] == pytest.approx(0.0, abs=1e-28)
    assert result.high_order_rows[0]["signal_fraction_nontrivial"] == pytest.approx(
        1.0, abs=1e-14
    )
    assert result.diagnostics["burnside_sum_d2"] == 6
    assert result.diagnostics["parseval_passed"] is True


def test_within_order_baseline_uses_dimension_squared() -> None:
    grid = permutation_grid(5)
    values = np.random.default_rng(2).normal(size=(len(grid), 4))
    result = analyze_landscape(
        CompleteLandscape.from_arrays(
            K=5,
            permutations=grid,
            values=values,
        ),
        AuditConfig(high_order_min=3),
    )
    rows = {row["partition"]: row for row in result.partition_rows}
    assert rows["(2, 2, 1)"]["coordinate_share_within_order"] == pytest.approx(
        25 / 41
    )
    assert rows["(2, 1, 1, 1)"]["coordinate_share_within_order"] == pytest.approx(
        16 / 41
    )


def test_k5_partition_results_match_packaged_reference_bundle() -> None:
    from taskorder_spectral import load_npz_landscape

    artifact = ROOT / "inputs/k5_landscape.npz"
    landscape = load_npz_landscape(
        artifact,
        K=5,
        permutations_key="permutations",
        values_key="values",
        seeds_key="seeds",
    )
    result = analyze_landscape(landscape, AuditConfig(high_order_min=3))
    actual = {row["partition"]: row for row in result.partition_rows}
    with (ROOT / "reference_outputs/spectral_audit/k5/partition_spectrum.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        expected_rows = list(csv.DictReader(handle))
    for expected in expected_rows:
        row = actual[expected["partition"]]
        assert row["plugin_energy"] == pytest.approx(
            float(expected["plugin_energy"]), abs=5.1e-15
        )
        assert row["signal_energy"] == pytest.approx(
            float(expected["signal_energy"]), abs=5.1e-15
        )
        assert row["jackknife_se"] == pytest.approx(
            float(expected["jackknife_se"]), abs=5.1e-15
        )
        assert row["seed_signflip_p_raw"] == pytest.approx(
            float(expected["seed_signflip_p_raw"]), abs=5.1e-15
        )


def test_single_landscape_requires_explicit_descriptive_mode() -> None:
    grid = permutation_grid(3)
    values = np.arange(6, dtype=float)[:, None]
    landscape = CompleteLandscape.from_arrays(
        K=3,
        permutations=grid,
        values=values,
    )
    with pytest.raises(ValueError, match="descriptive_only=True"):
        analyze_landscape(landscape, AuditConfig(high_order_min=1))


def test_descriptive_single_landscape_reports_projection_without_inference() -> None:
    grid = permutation_grid(3)
    target = build_irrep((2, 1), grid)[:, 0, 0]
    result = analyze_landscape(
        CompleteLandscape.from_arrays(
            K=3,
            permutations=grid,
            values=target[:, None],
        ),
        AuditConfig(high_order_min=1, descriptive_only=True),
    )
    rows = {row["partition"]: row for row in result.partition_rows}
    assert rows["(2, 1)"]["plugin_energy"] == pytest.approx(0.5, abs=1e-14)
    assert np.isnan(rows["(2, 1)"]["signal_energy"])
    assert rows["(2, 1)"]["inference_available"] is False
    assert rows["(2, 1)"]["supported_nontrivial_family"] is False
    assert result.order_rows[-1][
        "cumulative_plugin_fraction_nontrivial"
    ] == pytest.approx(1.0, abs=1e-14)
    assert result.high_order_rows[0][
        "plugin_fraction_nontrivial"
    ] == pytest.approx(1.0, abs=1e-14)
    assert result.high_order_rows[0]["fraction_available"] is False
    assert result.diagnostics["analysis_mode"] == "descriptive_only"
    assert result.diagnostics["parseval_passed"] is True


def test_descriptive_constant_landscape_marks_projection_fraction_unavailable() -> None:
    result = analyze_landscape(
        CompleteLandscape.from_arrays(
            K=3,
            permutations=permutation_grid(3),
            values=np.ones((6, 1)),
        ),
        AuditConfig(high_order_min=1, descriptive_only=True),
    )
    assert np.isnan(result.high_order_rows[0]["plugin_fraction_nontrivial"])
    assert (
        result.high_order_rows[0]["plugin_fraction_status"]
        == "zero_nontrivial_plugin_energy"
    )
