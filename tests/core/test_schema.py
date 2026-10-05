from __future__ import annotations

import itertools

import numpy as np
import pytest

from taskorder_spectral.schema import CompleteLandscape, load_npz_landscape


def permutation_grid(K: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(K))), dtype=np.int64)


def test_arbitrary_grid_order_and_seed_by_group_are_canonicalized() -> None:
    grid = permutation_grid(3)
    values = np.arange(24, dtype=float).reshape(6, 4)
    order = np.array([4, 0, 5, 2, 1, 3])
    result = CompleteLandscape.from_arrays(
        K=3,
        permutations=grid[order],
        values=values[order].T,
        values_layout="seed-by-group",
        seed_ids=np.array([11, 13, 17, 19]),
    )
    np.testing.assert_array_equal(result.permutations, grid)
    np.testing.assert_array_equal(result.values, values)
    assert result.seed_ids == (11, 13, 17, 19)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate", "duplicate or missing"),
        ("bad_row", "contain 0 through K-1"),
        ("nonfinite", "finite"),
    ],
)
def test_malformed_landscapes_fail_closed(mutation: str, match: str) -> None:
    grid = permutation_grid(3)
    values = np.ones((6, 3))
    if mutation == "duplicate":
        grid[-1] = grid[0]
    elif mutation == "bad_row":
        grid[-1, -1] = grid[-1, 0]
    else:
        values[-1, -1] = np.nan
    with pytest.raises(ValueError, match=match):
        CompleteLandscape.from_arrays(K=3, permutations=grid, values=values)


def test_npz_loader_uses_explicit_keys_and_records_digest(tmp_path) -> None:
    grid = permutation_grid(3)
    values = np.arange(18, dtype=float).reshape(6, 3)
    artifact = tmp_path / "landscape.npz"
    np.savez(
        artifact,
        my_grid=grid,
        my_scores=values,
        my_replicates=np.array([101, 102, 103]),
    )
    result = load_npz_landscape(
        artifact,
        K=3,
        permutations_key="my_grid",
        values_key="my_scores",
        seeds_key="my_replicates",
    )
    assert result.source_path == artifact.resolve()
    assert result.source_sha256 is not None
    assert len(result.source_sha256) == 64
    assert result.permutations_key == "my_grid"
    assert result.values_key == "my_scores"
    assert result.seeds_key == "my_replicates"


def test_npz_loader_does_not_guess_missing_keys(tmp_path) -> None:
    artifact = tmp_path / "landscape.npz"
    np.savez(
        artifact,
        permutations=permutation_grid(3),
        values=np.ones((6, 3)),
    )
    with pytest.raises(ValueError, match="missing NPZ keys"):
        load_npz_landscape(
            artifact,
            K=3,
            permutations_key="permutations",
            values_key="not_values",
            seeds_key=None,
        )


def test_nonfinite_seed_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="seed IDs must be finite"):
        CompleteLandscape.from_arrays(
            K=3,
            permutations=permutation_grid(3),
            values=np.ones((6, 3)),
            seed_ids=np.array([0.0, 1.0, np.nan]),
        )


def test_single_complete_landscape_is_accepted_and_canonicalized() -> None:
    grid = permutation_grid(3)
    values = np.arange(6, dtype=float)[:, None]
    order = np.array([4, 0, 5, 2, 1, 3])
    result = CompleteLandscape.from_arrays(
        K=3,
        permutations=grid[order],
        values=values[order],
        seed_ids=np.array(["observed"]),
    )
    np.testing.assert_array_equal(result.permutations, grid)
    np.testing.assert_array_equal(result.values, values)
    assert result.n_seeds == 1
    assert result.seed_ids == ("observed",)


def test_empty_landscape_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one complete landscape"):
        CompleteLandscape.from_arrays(
            K=3,
            permutations=permutation_grid(3),
            values=np.empty((6, 0)),
        )
