from __future__ import annotations

import itertools

import numpy as np
import pytest

from taskorder_spectral import AuditConfig, CompleteLandscape, analyze_landscape
from taskorder_spectral.allocation import infer_allocation
from taskorder_spectral.inference import (
    _student_t_survival,
    unbiased_signal_energy,
)


def permutation_grid(K: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(K))), dtype=np.int64)


def jackknife_standard_error(values: np.ndarray) -> float:
    centered = values - values.mean()
    return float(np.sqrt((len(values) - 1) / len(values) * centered @ centered))


def test_student_t_tail_matches_reference_quantile() -> None:
    assert _student_t_survival(2.2621571627409915, 9) == pytest.approx(
        0.025, abs=3e-12
    )


def test_allocation_recomputes_full_contrast_in_every_jackknife_replicate() -> None:
    rng = np.random.default_rng(6001)
    target = np.array([2.0, 1.0, 0.5]) + rng.normal(scale=0.08, size=(7, 3))
    other = np.array([0.6, 0.3]) + rng.normal(scale=0.08, size=(7, 2))
    baseline = 0.4
    result = infer_allocation(
        target,
        other,
        baseline_share=baseline,
        confidence=0.95,
    )
    target_energy = unbiased_signal_energy(target)
    other_energy = unbiased_signal_energy(other)
    expected_contrast = target_energy - baseline * (target_energy + other_energy)
    contrast_loo = []
    for index in range(len(target)):
        target_loo = unbiased_signal_energy(np.delete(target, index, axis=0))
        other_loo = unbiased_signal_energy(np.delete(other, index, axis=0))
        contrast_loo.append(
            target_loo - baseline * (target_loo + other_loo)
        )
    expected_se = jackknife_standard_error(np.asarray(contrast_loo))
    assert result["allocation_contrast_ustat"] == pytest.approx(
        expected_contrast, abs=1e-14
    )
    assert result["allocation_contrast_jackknife_se"] == pytest.approx(
        expected_se, abs=1e-14
    )
    assert result["signal_order_total_energy_ustat"] == pytest.approx(
        target_energy + other_energy, abs=1e-14
    )
    assert result["signal_share_ci_available"] is True
    assert 0.0 <= result["signal_share_contrast_inversion_ci_lower"]
    assert result["signal_share_contrast_inversion_ci_upper"] <= 1.0


def test_allocation_with_weak_signal_returns_explicit_share_status() -> None:
    target = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    other = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    result = infer_allocation(target, other, baseline_share=0.5)
    assert result["signal_share_ci_available"] is False
    assert np.isnan(result["signal_share_contrast_inversion_ci_lower"])
    assert result["signal_share_ci_status"] != "ok"


def test_requested_target_uses_dimension_squared_baseline_and_holm_family() -> None:
    grid = permutation_grid(5)
    values = np.random.default_rng(12).normal(size=(len(grid), 6))
    result = analyze_landscape(
        CompleteLandscape.from_arrays(K=5, permutations=grid, values=values),
        AuditConfig(
            high_order_min=3,
            allocation_targets=((2, 2, 1), (2, 1, 1, 1)),
        ),
    )
    rows = {row["target_partition"]: row for row in result.allocation_rows}
    assert rows["(2, 2, 1)"]["coordinate_share_within_order"] == pytest.approx(
        25 / 41
    )
    assert rows["(2, 1, 1, 1)"][
        "coordinate_share_within_order"
    ] == pytest.approx(16 / 41)
    for row in rows.values():
        assert row["allocation_p_holm_requested_targets_two_sided"] >= row[
            "allocation_p_raw_two_sided"
        ]
        assert row["multiplicity_scope"] == (
            "Holm_across_targets_requested_in_this_audit"
        )


@pytest.mark.parametrize(
    "target,match",
    [
        ((5,), "no same-order complement"),
        ((4, 1), "no same-order complement"),
        ((2, 3), "nonincreasing"),
        ((2, 2), "not a partition of K=5"),
    ],
)
def test_invalid_allocation_targets_fail_closed(target, match: str) -> None:
    grid = permutation_grid(5)
    values = np.zeros((len(grid), 3))
    if target == (2, 3):
        with pytest.raises(ValueError, match=match):
            AuditConfig(allocation_targets=(target,))
        return
    with pytest.raises(ValueError, match=match):
        analyze_landscape(
            CompleteLandscape.from_arrays(K=5, permutations=grid, values=values),
            AuditConfig(high_order_min=3, allocation_targets=(target,)),
        )
