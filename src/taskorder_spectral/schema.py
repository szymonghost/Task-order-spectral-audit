"""Input schema for one or more complete task-order landscapes.

The canonical in-memory layout is:

* ``permutations``: integer array with shape ``(K!, K)``;
* ``values``: finite float array with shape ``(K!, M)``;
* ``seed_ids``: one-dimensional array of length ``M``.

Every row of ``permutations`` must contain ``0, ..., K-1`` exactly once and
the grid must contain every element of :math:`S_K` exactly once.  Input rows
may arrive in any order; validation reorders them lexicographically before
analysis.  Seeds are the inferential units.  Permutation rows are a fixed,
complete design and are never bootstrapped as if they were iid observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CompleteLandscape:
    """Validated scalar landscapes on a complete ``S_K`` grid."""

    K: int
    permutations: np.ndarray
    values: np.ndarray
    seed_ids: tuple[Any, ...]
    source_path: Path | None = None
    source_sha256: str | None = None
    permutations_key: str | None = None
    values_key: str | None = None
    seeds_key: str | None = None
    input_layout: str = "group-by-seed"

    @property
    def group_order(self) -> int:
        return factorial(self.K)

    @property
    def n_seeds(self) -> int:
        return self.values.shape[1]

    @classmethod
    def from_arrays(
        cls,
        *,
        K: int,
        permutations: Any,
        values: Any,
        seed_ids: Any | None = None,
        values_layout: str = "group-by-seed",
        source_path: str | Path | None = None,
        source_sha256: str | None = None,
        permutations_key: str | None = None,
        values_key: str | None = None,
        seeds_key: str | None = None,
    ) -> "CompleteLandscape":
        """Validate arrays and return them in canonical lexicographic layout."""

        if not isinstance(K, (int, np.integer)) or int(K) < 2:
            raise ValueError("K must be an integer at least 2")
        K = int(K)
        group_order = factorial(K)

        grid = np.asarray(permutations)
        if grid.shape != (group_order, K):
            raise ValueError(
                "permutations must have shape "
                f"({group_order}, {K}), got {grid.shape}"
            )
        if not np.issubdtype(grid.dtype, np.integer):
            raise ValueError("permutations must use an integer dtype")
        grid = np.asarray(grid, dtype=np.int64)
        target_row = np.arange(K, dtype=np.int64)
        if not np.all(np.sort(grid, axis=1) == target_row):
            raise ValueError("every permutation row must contain 0 through K-1 once")

        tuples = [tuple(int(value) for value in row) for row in grid]
        if len(set(tuples)) != group_order:
            raise ValueError("permutation grid contains duplicate or missing elements")
        order = np.array(sorted(range(group_order), key=tuples.__getitem__), dtype=int)
        grid = np.ascontiguousarray(grid[order])

        matrix = np.asarray(values, dtype=np.float64)
        if values_layout == "seed-by-group":
            matrix = matrix.T
        elif values_layout != "group-by-seed":
            raise ValueError(
                "values_layout must be 'group-by-seed' or 'seed-by-group'"
            )
        if matrix.ndim != 2 or matrix.shape[0] != group_order:
            raise ValueError(
                "values must have canonical shape "
                f"({group_order}, M), got {matrix.shape} after layout conversion"
            )
        if matrix.shape[1] < 1:
            raise ValueError("values must contain at least one complete landscape")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("values must contain only finite numbers")
        matrix = np.ascontiguousarray(matrix[order])

        if seed_ids is None:
            seeds = tuple(range(matrix.shape[1]))
        else:
            seed_array = np.asarray(seed_ids)
            if seed_array.ndim != 1 or len(seed_array) != matrix.shape[1]:
                raise ValueError(
                    f"seed_ids must be one-dimensional with length {matrix.shape[1]}"
                )
            seeds = tuple(_json_scalar(value) for value in seed_array)
            try:
                unique_count = len(set(seeds))
            except TypeError as error:
                raise ValueError("seed IDs must be scalar, hashable values") from error
            if any(
                isinstance(seed, (float, np.floating))
                and not np.isfinite(float(seed))
                for seed in seeds
            ):
                raise ValueError("numeric seed IDs must be finite")
            if unique_count != len(seeds):
                raise ValueError("seed IDs must be unique")

        resolved = Path(source_path).resolve() if source_path is not None else None
        return cls(
            K=K,
            permutations=grid,
            values=matrix,
            seed_ids=seeds,
            source_path=resolved,
            source_sha256=source_sha256,
            permutations_key=permutations_key,
            values_key=values_key,
            seeds_key=seeds_key,
            input_layout=values_layout,
        )


def load_npz_landscape(
    path: str | Path,
    *,
    K: int,
    permutations_key: str = "permutations",
    values_key: str = "F_per_seed",
    seeds_key: str | None = "seeds",
    values_layout: str = "group-by-seed",
) -> CompleteLandscape:
    """Load a generic numeric NPZ using explicit field and layout choices."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"landscape artifact does not exist: {source}")
    digest_before = sha256_file(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {permutations_key, values_key}
            if seeds_key is not None:
                required.add(seeds_key)
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{source}: missing NPZ keys {sorted(missing)}")
            permutations = np.array(archive[permutations_key], copy=True)
            values = np.array(archive[values_key], copy=True)
            seeds = (
                np.array(archive[seeds_key], copy=True)
                if seeds_key is not None
                else None
            )
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(source) in str(error):
            raise
        raise ValueError(f"{source}: cannot load numeric NPZ safely: {error}") from error
    digest_after = sha256_file(source)
    if digest_after != digest_before:
        raise RuntimeError(f"landscape artifact changed while it was being loaded: {source}")

    return CompleteLandscape.from_arrays(
        K=K,
        permutations=permutations,
        values=values,
        seed_ids=seeds,
        values_layout=values_layout,
        source_path=source,
        source_sha256=digest_after,
        permutations_key=permutations_key,
        values_key=values_key,
        seeds_key=seeds_key,
    )
