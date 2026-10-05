"""Young-orthogonal irreducible representations of symmetric groups.

This implementation is extracted from the repository's validated
``predictive_irrep_features.py`` machinery.  Representations are built one
partition at a time so a full audit need not retain the entire regular
representation in memory.
"""

from __future__ import annotations

from collections import deque
from math import factorial
from typing import Iterable

import numpy as np


def partitions(n: int, maximum: int | None = None) -> Iterable[tuple[int, ...]]:
    """Yield integer partitions in reverse lexicographic order."""

    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    if maximum is None:
        maximum = n
    if n == 0:
        yield ()
        return
    for first in range(min(n, maximum), 0, -1):
        for remainder in partitions(n - first, first):
            yield (first,) + remainder


def hook_length(partition: tuple[int, ...], row: int, column: int) -> int:
    arm = partition[row] - column - 1
    leg = sum(
        1
        for lower_row in range(row + 1, len(partition))
        if column < partition[lower_row]
    )
    return arm + leg + 1


def irrep_dimension(partition: tuple[int, ...]) -> int:
    """Dimension from the hook-length formula."""

    if not partition or any(value <= 0 for value in partition):
        raise ValueError("partition must contain positive row lengths")
    if any(partition[i] < partition[i + 1] for i in range(len(partition) - 1)):
        raise ValueError("partition row lengths must be nonincreasing")
    hook_product = 1
    for row, length in enumerate(partition):
        for column in range(length):
            hook_product *= hook_length(partition, row, column)
    return factorial(sum(partition)) // hook_product


def standard_young_tableaux(shape: tuple[int, ...]) -> list[list[list[int]]]:
    """Enumerate standard Young tableaux of ``shape``."""

    n_cells = sum(shape)
    tableau = [[0] * length for length in shape]
    result: list[list[list[int]]] = []

    def allowed(row: int, column: int, value: int) -> bool:
        if tableau[row][column]:
            return False
        if column > 0 and (
            tableau[row][column - 1] == 0
            or tableau[row][column - 1] >= value
        ):
            return False
        if row > 0 and column < len(tableau[row - 1]) and (
            tableau[row - 1][column] == 0
            or tableau[row - 1][column] >= value
        ):
            return False
        return True

    def fill(value: int) -> None:
        if value > n_cells:
            result.append([values[:] for values in tableau])
            return
        for row, length in enumerate(shape):
            for column in range(length):
                if allowed(row, column, value):
                    tableau[row][column] = value
                    fill(value + 1)
                    tableau[row][column] = 0

    fill(1)
    return result


def adjacent_transpositions(K: int) -> tuple[np.ndarray, ...]:
    result = []
    for generator in range(K - 1):
        permutation = np.arange(K, dtype=np.int64)
        permutation[generator], permutation[generator + 1] = (
            generator + 1,
            generator,
        )
        result.append(permutation)
    return tuple(result)


def build_irrep(
    shape: tuple[int, ...],
    permutations: np.ndarray,
) -> np.ndarray:
    """Build one Young-orthogonal irrep in the supplied group-element order."""

    grid = np.asarray(permutations, dtype=np.int64)
    if grid.ndim != 2 or grid.shape[1] != sum(shape):
        raise ValueError("permutations and partition size disagree")
    K = grid.shape[1]
    group_order = factorial(K)
    if grid.shape[0] != group_order:
        raise ValueError("build_irrep requires the complete symmetric group")
    permutation_to_index = {
        tuple(int(value) for value in permutation): index
        for index, permutation in enumerate(grid)
    }
    if len(permutation_to_index) != group_order:
        raise ValueError("permutation grid is not unique")

    tableaux = standard_young_tableaux(shape)
    dimension = len(tableaux)
    expected_dimension = irrep_dimension(shape)
    if dimension != expected_dimension:
        raise RuntimeError(
            f"tableau count {dimension} disagrees with hook dimension "
            f"{expected_dimension} for {shape}"
        )
    flattened = [
        tuple(value for row in tableau for value in row)
        for tableau in tableaux
    ]
    tableau_index = {tableau: index for index, tableau in enumerate(flattened)}

    def swapped(tableau: tuple[int, ...], first: int, second: int):
        values = list(tableau)
        first_index, second_index = values.index(first), values.index(second)
        values[first_index], values[second_index] = (
            values[second_index],
            values[first_index],
        )
        return tuple(values)

    def find_entry(tableau: list[list[int]], value: int) -> tuple[int, int]:
        for row, values in enumerate(tableau):
            for column, entry in enumerate(values):
                if entry == value:
                    return row, column
        raise RuntimeError("tableau entry is missing")

    generator_matrices = []
    for generator in range(K - 1):
        first, second = generator + 1, generator + 2
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for index, tableau in enumerate(tableaux):
            row1, column1 = find_entry(tableau, first)
            row2, column2 = find_entry(tableau, second)
            axial_distance = (column2 - row2) - (column1 - row1)
            matrix[index, index] = 1.0 / axial_distance
            partner = tableau_index.get(swapped(flattened[index], first, second))
            if partner is not None and partner != index:
                value = np.sqrt(1.0 - 1.0 / axial_distance**2)
                matrix[index, partner] = value
                matrix[partner, index] = value
        generator_matrices.append(matrix)

    representations: list[np.ndarray | None] = [None] * group_order
    identity = np.arange(K, dtype=np.int64)
    identity_key = tuple(int(value) for value in identity)
    if identity_key not in permutation_to_index:
        raise ValueError("permutation grid omits the identity")
    representations[permutation_to_index[identity_key]] = np.eye(dimension)
    queue = deque([identity])
    visited = {identity_key}
    adjacent = adjacent_transpositions(K)
    while queue:
        permutation = queue.popleft()
        permutation_index = permutation_to_index[
            tuple(int(value) for value in permutation)
        ]
        current = representations[permutation_index]
        if current is None:
            raise RuntimeError("representation traversal reached an empty state")
        for generator in range(K - 1):
            next_permutation = adjacent[generator][permutation]
            next_key = tuple(int(value) for value in next_permutation)
            if next_key not in visited:
                visited.add(next_key)
                representations[permutation_to_index[next_key]] = (
                    generator_matrices[generator] @ current
                )
                queue.append(next_permutation)
    if len(visited) != group_order or any(value is None for value in representations):
        raise RuntimeError("irrep traversal did not cover the complete group")
    return np.asarray(representations, dtype=np.float64)


def partition_catalog(K: int) -> list[dict[str, object]]:
    """Return partitions, dimensions, and interaction orders with Burnside check."""

    shapes = list(partitions(K))
    dimensions = [irrep_dimension(shape) for shape in shapes]
    if sum(dimension**2 for dimension in dimensions) != factorial(K):
        raise RuntimeError("Burnside dimension identity failed")
    return [
        {
            "partition": shape,
            "dimension": dimension,
            "interaction_order": K - shape[0],
        }
        for shape, dimension in zip(shapes, dimensions)
    ]
