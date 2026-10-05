"""End-to-end spectral audit and reproducible CSV publication.

The audit works with a complete, seed-replicated scalar landscape on ``S_K``.
It computes one Young-orthogonal representation at a time, so the peak
representation memory is set by the largest irrep rather than by the entire
regular representation.
"""

from __future__ import annotations

import csv
import json
import math
import platform
from dataclasses import asdict, dataclass
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .allocation import describe_allocation, infer_allocation
from .inference import (
    describe_vectors,
    holm_adjust,
    infer_vectors,
    jackknife_signal_fraction,
    per_seed_fourier,
    sign_flip_patterns,
    unbiased_signal_energy,
)
from .io import atomic_text_writer, sha256_file
from .irreps import build_irrep, partition_catalog
from .schema import CompleteLandscape


ANALYSIS_VERSION = "taskorder_spectral_audit_v2"


@dataclass(frozen=True)
class AuditConfig:
    """Inference controls for :func:`analyze_landscape`."""

    high_order_min: int = 3
    confidence: float = 0.95
    max_exact_seeds: int = 16
    monte_carlo_signs: int = 20_000
    random_seed: int = 20_260_524
    descriptive_only: bool = False
    allocation_targets: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.high_order_min, int) or self.high_order_min < 1:
            raise ValueError("high_order_min must be a positive integer")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if not isinstance(self.max_exact_seeds, int) or self.max_exact_seeds < 1:
            raise ValueError("max_exact_seeds must be a positive integer")
        if not isinstance(self.monte_carlo_signs, int) or self.monte_carlo_signs < 1:
            raise ValueError("monte_carlo_signs must be a positive integer")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if not isinstance(self.descriptive_only, bool):
            raise ValueError("descriptive_only must be boolean")
        try:
            targets = tuple(tuple(target) for target in self.allocation_targets)
        except TypeError as error:
            raise ValueError(
                "allocation_targets must be an iterable of integer partitions"
            ) from error
        for target in targets:
            if not target or any(
                not isinstance(value, (int, np.integer)) or int(value) <= 0
                for value in target
            ):
                raise ValueError(
                    "each allocation target must contain positive integers"
                )
            if any(int(target[index]) < int(target[index + 1])
                   for index in range(len(target) - 1)):
                raise ValueError(
                    "allocation target parts must be nonincreasing"
                )
        normalized = tuple(
            tuple(int(value) for value in target) for target in targets
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("allocation targets must be unique")
        object.__setattr__(self, "allocation_targets", normalized)


@dataclass(frozen=True)
class AuditResult:
    """Serializable audit tables and algebraic diagnostics."""

    landscape: CompleteLandscape
    config: AuditConfig
    partition_rows: tuple[dict[str, Any], ...]
    order_rows: tuple[dict[str, Any], ...]
    high_order_rows: tuple[dict[str, Any], ...]
    allocation_rows: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def _partition_label(shape: tuple[int, ...]) -> str:
    return str(tuple(int(value) for value in shape))


def _public_inference(inference: dict[str, object]) -> dict[str, object]:
    """Drop internal leave-one-out vectors before rows are serialized."""

    return {
        key: value
        for key, value in inference.items()
        if key != "leave_one_out"
    }


def _supported(row: dict[str, Any], alpha: float, *, p_key: str) -> bool:
    p_value = float(row[p_key])
    lower = float(row["signal_ci_lower"])
    return bool(p_value < alpha and lower > 0.0)


def _aggregate_vectors(
    blocks: list[np.ndarray],
    indices: list[int],
    *,
    n_seeds: int,
) -> np.ndarray:
    if not indices:
        return np.empty((n_seeds, 0), dtype=np.float64)
    return np.concatenate([blocks[index] for index in indices], axis=1)


def _summarize_vectors(
    vectors: np.ndarray,
    *,
    seed_inference: bool,
    signs: np.ndarray | None,
    signs_exact: bool,
    confidence: float,
) -> dict[str, object]:
    if not seed_inference:
        return _public_inference(describe_vectors(vectors))
    if signs is None:
        raise RuntimeError("seed-aware inference requires sign patterns")
    return _public_inference(
        infer_vectors(
            vectors,
            signs,
            signs_exact=signs_exact,
            confidence=confidence,
        )
    )


def _inference_labels(seed_inference: bool) -> tuple[str, str]:
    if seed_inference:
        return (
            "cross_seed_u_jackknife_t_whole_seed_sign_flip",
            "adjusted_p_below_alpha_and_signal_ci_lower_above_zero",
        )
    return "not_available_descriptive_only", "not_applicable_descriptive_only"


def analyze_landscape(
    landscape: CompleteLandscape,
    config: AuditConfig | None = None,
) -> AuditResult:
    """Decompose a complete landscape and optionally run seed-aware inference.

    Energies use the Plancherel normalization ``d_lambda / (K!)**2``.
    In seed-aware mode, signal energies are all-cross-seed U-statistics,
    confidence intervals delete one whole seed at a time, and randomization
    tests flip one sign per complete seed landscape.  Descriptive-only mode
    reports exact plug-in decompositions and projection ceilings without
    population-signal quantities.
    """

    if not isinstance(landscape, CompleteLandscape):
        raise TypeError("landscape must be a validated CompleteLandscape")
    if config is None:
        config = AuditConfig()
    if config.high_order_min > landscape.K - 1:
        raise ValueError(
            f"high_order_min={config.high_order_min} leaves no nontrivial "
            f"components for K={landscape.K}"
        )
    if landscape.n_seeds < 3 and not config.descriptive_only:
        raise ValueError(
            "seed-aware inference requires at least three independent "
            "landscapes; set descriptive_only=True (CLI: --descriptive-only) "
            "to publish decomposition and projection summaries without "
            "inferential claims"
        )

    K = landscape.K
    group_order = factorial(K)
    n_seeds = landscape.n_seeds
    seed_inference = not config.descriptive_only
    alpha = 1.0 - config.confidence
    catalog = partition_catalog(K)
    shapes = {tuple(component["partition"]) for component in catalog}
    for target in config.allocation_targets:
        if sum(target) != K or target not in shapes:
            raise ValueError(
                f"allocation target {target} is not a partition of K={K}"
            )
        target_order = K - target[0]
        same_order = [
            component
            for component in catalog
            if int(component["interaction_order"]) == target_order
        ]
        if len(same_order) < 2:
            raise ValueError(
                f"allocation target {target} has no same-order complement"
            )
    if seed_inference:
        rng = np.random.default_rng(config.random_seed)
        signs, signs_exact = sign_flip_patterns(
            n_seeds,
            max_exact_seeds=config.max_exact_seeds,
            n_draws=config.monte_carlo_signs,
            rng=rng,
        )
    else:
        signs, signs_exact = None, False

    partition_rows: list[dict[str, Any]] = []
    blocks: list[np.ndarray] = []
    per_seed_partition_energies: list[np.ndarray] = []

    for component in catalog:
        shape = component["partition"]
        dimension = int(component["dimension"])
        interaction_order = int(component["interaction_order"])
        representation = build_irrep(shape, landscape.permutations)
        hats = per_seed_fourier(landscape.values, representation)
        scaled = np.ascontiguousarray(
            (math.sqrt(dimension) / group_order) * hats.reshape(n_seeds, -1)
        )
        blocks.append(scaled)
        per_seed_partition_energies.append(
            np.einsum("mi,mi->m", scaled, scaled, optimize=True)
        )

        row: dict[str, Any] = {
            "K": K,
            "group_order": group_order,
            "n_seeds": n_seeds,
            "partition": _partition_label(shape),
            "dimension": dimension,
            "interaction_order": interaction_order,
            "coordinates_d2": dimension**2,
            "coordinate_share_all": dimension**2 / group_order,
        }
        inference = _summarize_vectors(
            scaled,
            seed_inference=seed_inference,
            signs=signs,
            signs_exact=signs_exact,
            confidence=config.confidence,
        )
        row.update(inference)
        partition_rows.append(row)

    nontrivial_indices = [
        index
        for index, row in enumerate(partition_rows)
        if int(row["interaction_order"]) >= 1
    ]
    high_indices = [
        index
        for index, row in enumerate(partition_rows)
        if int(row["interaction_order"]) >= config.high_order_min
    ]

    for row in partition_rows:
        interaction_order = int(row["interaction_order"])
        same_order = [
            candidate
            for candidate in partition_rows
            if int(candidate["interaction_order"]) == interaction_order
        ]
        coordinate_denominator = sum(
            int(candidate["coordinates_d2"]) for candidate in same_order
        )
        plugin_denominator = sum(
            float(candidate["plugin_energy"]) for candidate in same_order
        )
        signal_denominator = sum(
            float(candidate["signal_energy"]) for candidate in same_order
        )
        row["coordinate_share_within_order"] = (
            int(row["coordinates_d2"]) / coordinate_denominator
        )
        row["plugin_energy_share_within_order"] = (
            float(row["plugin_energy"]) / plugin_denominator
            if plugin_denominator > 0.0
            else float("nan")
        )
        row["signal_energy_share_within_order"] = (
            float(row["signal_energy"]) / signal_denominator
            if signal_denominator > 0.0
            else float("nan")
        )

    if seed_inference:
        nontrivial_adjusted = holm_adjust(
            [float(partition_rows[index]["seed_signflip_p_raw"])
             for index in nontrivial_indices]
        )
        high_adjusted = holm_adjust(
            [float(partition_rows[index]["seed_signflip_p_raw"])
             for index in high_indices]
        )
    else:
        nontrivial_adjusted = np.full(len(nontrivial_indices), np.nan)
        high_adjusted = np.full(len(high_indices), np.nan)
    inference_method, partition_support_rule = _inference_labels(seed_inference)
    for row in partition_rows:
        row["seed_signflip_p_holm_nontrivial"] = float("nan")
        row["seed_signflip_p_holm_high_order"] = float("nan")
        row["supported_nontrivial_family"] = False
        row["supported_high_order_family"] = False
        row["alpha"] = alpha
        row["support_rule"] = partition_support_rule
        row["inference_method"] = inference_method
    for index, adjusted in zip(nontrivial_indices, nontrivial_adjusted):
        row = partition_rows[index]
        row["seed_signflip_p_holm_nontrivial"] = float(adjusted)
        if seed_inference:
            row["supported_nontrivial_family"] = _supported(
                row, alpha, p_key="seed_signflip_p_holm_nontrivial"
            )
    for index, adjusted in zip(high_indices, high_adjusted):
        row = partition_rows[index]
        row["seed_signflip_p_holm_high_order"] = float(adjusted)
        if seed_inference:
            row["supported_high_order_family"] = _supported(
                row, alpha, p_key="seed_signflip_p_holm_high_order"
            )

    order_rows: list[dict[str, Any]] = []
    for interaction_order in sorted(
        {int(row["interaction_order"]) for row in partition_rows}
    ):
        indices = [
            index
            for index, row in enumerate(partition_rows)
            if int(row["interaction_order"]) == interaction_order
        ]
        combined = _aggregate_vectors(blocks, indices, n_seeds=n_seeds)
        inference = _summarize_vectors(
            combined,
            seed_inference=seed_inference,
            signs=signs,
            signs_exact=signs_exact,
            confidence=config.confidence,
        )
        row = {
            "K": K,
            "group_order": group_order,
            "n_seeds": n_seeds,
            "interaction_order": interaction_order,
            "partitions": ";".join(
                str(partition_rows[index]["partition"]) for index in indices
            ),
            "partition_count": len(indices),
            "coordinates_d2": sum(
                int(partition_rows[index]["coordinates_d2"])
                for index in indices
            ),
        }
        row["coordinate_share_all"] = row["coordinates_d2"] / group_order
        row.update(inference)
        row["supported"] = (
            _supported(row, alpha, p_key="seed_signflip_p_raw")
            if seed_inference
            else False
        )
        row["alpha"] = alpha
        row["support_rule"] = (
            "raw_p_below_alpha_and_signal_ci_lower_above_zero"
            if seed_inference
            else "not_applicable_descriptive_only"
        )
        row["inference_method"] = inference_method
        order_rows.append(row)

    nontrivial_plugin_total = sum(
        float(partition_rows[index]["plugin_energy"])
        for index in nontrivial_indices
    )
    cumulative_nontrivial_plugin = 0.0
    for row in order_rows:
        interaction_order = int(row["interaction_order"])
        if interaction_order >= 1:
            cumulative_nontrivial_plugin += float(row["plugin_energy"])
        if nontrivial_plugin_total > 0.0:
            row["plugin_fraction_nontrivial"] = (
                float(row["plugin_energy"]) / nontrivial_plugin_total
                if interaction_order >= 1
                else 0.0
            )
            row["cumulative_plugin_fraction_nontrivial"] = (
                cumulative_nontrivial_plugin / nontrivial_plugin_total
            )
            row["projection_fraction_status"] = "ok"
        else:
            row["plugin_fraction_nontrivial"] = float("nan")
            row["cumulative_plugin_fraction_nontrivial"] = float("nan")
            row["projection_fraction_status"] = "zero_nontrivial_plugin_energy"

    high_vectors = _aggregate_vectors(blocks, high_indices, n_seeds=n_seeds)
    nontrivial_vectors = _aggregate_vectors(
        blocks, nontrivial_indices, n_seeds=n_seeds
    )
    high_inference = _summarize_vectors(
        high_vectors,
        seed_inference=seed_inference,
        signs=signs,
        signs_exact=signs_exact,
        confidence=config.confidence,
    )
    if seed_inference:
        fraction = jackknife_signal_fraction(
            high_vectors,
            nontrivial_vectors,
            confidence=config.confidence,
        )
        all_nontrivial_signal_energy = unbiased_signal_energy(
            nontrivial_vectors
        )
    else:
        fraction = {
            "available": False,
            "status": "descriptive_only_no_seed_inference",
            "estimate": float("nan"),
            "standard_error": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
        }
        all_nontrivial_signal_energy = float("nan")
    high_row: dict[str, Any] = {
        "K": K,
        "group_order": group_order,
        "n_seeds": n_seeds,
        "minimum_interaction_order": config.high_order_min,
        "orders_included": ";".join(
            str(order)
            for order in sorted(
                {int(partition_rows[index]["interaction_order"])
                 for index in high_indices}
            )
        ),
        "partitions": ";".join(
            str(partition_rows[index]["partition"]) for index in high_indices
        ),
        "partition_count": len(high_indices),
        "coordinates_d2": sum(
            int(partition_rows[index]["coordinates_d2"])
            for index in high_indices
        ),
    }
    high_row["coordinate_share_all"] = high_row["coordinates_d2"] / group_order
    high_row.update(high_inference)
    high_row["all_nontrivial_plugin_energy"] = nontrivial_plugin_total
    high_row["plugin_fraction_nontrivial"] = (
        float(high_row["plugin_energy"]) / nontrivial_plugin_total
        if nontrivial_plugin_total > 0.0
        else float("nan")
    )
    high_row["plugin_fraction_status"] = (
        "ok" if nontrivial_plugin_total > 0.0
        else "zero_nontrivial_plugin_energy"
    )
    high_row["all_nontrivial_signal_energy"] = all_nontrivial_signal_energy
    high_row["signal_fraction_nontrivial"] = fraction["estimate"]
    high_row["fraction_jackknife_se"] = fraction["standard_error"]
    high_row["fraction_ci_lower"] = fraction["lower"]
    high_row["fraction_ci_upper"] = fraction["upper"]
    high_row["fraction_available"] = fraction["available"]
    high_row["fraction_status"] = fraction["status"]
    high_row["supported"] = (
        _supported(high_row, alpha, p_key="seed_signflip_p_raw")
        if seed_inference
        else False
    )
    high_row["alpha"] = alpha
    high_row["support_rule"] = (
        "raw_joint_p_below_alpha_and_signal_ci_lower_above_zero"
        if seed_inference
        else "not_applicable_descriptive_only"
    )
    high_row["inference_method"] = inference_method

    shape_to_index = {
        tuple(component["partition"]): index
        for index, component in enumerate(catalog)
    }
    allocation_rows: list[dict[str, Any]] = []
    for target in config.allocation_targets:
        target_index = shape_to_index[target]
        target_order = int(partition_rows[target_index]["interaction_order"])
        same_order_indices = [
            index
            for index, row in enumerate(partition_rows)
            if int(row["interaction_order"]) == target_order
        ]
        complement_indices = [
            index for index in same_order_indices if index != target_index
        ]
        target_coordinates = int(
            partition_rows[target_index]["coordinates_d2"]
        )
        complement_coordinates = sum(
            int(partition_rows[index]["coordinates_d2"])
            for index in complement_indices
        )
        order_coordinates = target_coordinates + complement_coordinates
        baseline_share = target_coordinates / order_coordinates
        target_vectors = blocks[target_index]
        complement_vectors = _aggregate_vectors(
            blocks, complement_indices, n_seeds=n_seeds
        )
        if seed_inference:
            allocation = infer_allocation(
                target_vectors,
                complement_vectors,
                baseline_share=baseline_share,
                confidence=config.confidence,
            )
        else:
            allocation = describe_allocation(
                target_vectors,
                complement_vectors,
                baseline_share=baseline_share,
            )
        allocation_row: dict[str, Any] = {
            "K": K,
            "group_order": group_order,
            "n_seeds": n_seeds,
            "target_partition": _partition_label(target),
            "interaction_order": target_order,
            "order_partitions": ";".join(
                str(partition_rows[index]["partition"])
                for index in same_order_indices
            ),
            "complement_partitions": ";".join(
                str(partition_rows[index]["partition"])
                for index in complement_indices
            ),
            "target_dimension": int(partition_rows[target_index]["dimension"]),
            "target_coordinates_d2": target_coordinates,
            "complement_coordinates_d2": complement_coordinates,
            "order_coordinates_d2": order_coordinates,
            "coordinate_share_within_order": baseline_share,
        }
        allocation_row.update(allocation)
        allocation_row["allocation_p_holm_requested_targets_one_sided"] = (
            float("nan")
        )
        allocation_row["allocation_p_holm_requested_targets_two_sided"] = (
            float("nan")
        )
        allocation_row["confidence"] = config.confidence
        allocation_row["alpha"] = alpha
        allocation_row["allocation_null"] = (
            "population_signal_target_share_equals_dimension_squared_share"
        )
        allocation_row["allocation_direction"] = "target_above_baseline"
        allocation_row["multiplicity_scope"] = (
            "Holm_across_targets_requested_in_this_audit"
            if len(config.allocation_targets) > 1
            else "single_requested_target"
        )
        allocation_row["target_selection_scope"] = (
            "requested_targets_only_no_adjustment_for_prior_data_dependent_selection"
        )
        allocation_rows.append(allocation_row)

    if seed_inference and allocation_rows:
        one_sided_adjusted = holm_adjust(
            [float(row["allocation_p_raw_one_sided"])
             for row in allocation_rows]
        )
        two_sided_adjusted = holm_adjust(
            [float(row["allocation_p_raw_two_sided"])
             for row in allocation_rows]
        )
        for row, one_sided, two_sided in zip(
            allocation_rows, one_sided_adjusted, two_sided_adjusted
        ):
            row["allocation_p_holm_requested_targets_one_sided"] = float(
                one_sided
            )
            row["allocation_p_holm_requested_targets_two_sided"] = float(
                two_sided
            )

    mean_landscape = landscape.values.mean(axis=1)
    direct_plugin_total = float(np.mean(mean_landscape**2))
    direct_nontrivial_plugin = float(
        np.mean((mean_landscape - mean_landscape.mean()) ** 2)
    )
    spectral_plugin_total = float(
        sum(float(row["plugin_energy"]) for row in partition_rows)
    )
    spectral_nontrivial_plugin = float(
        sum(float(partition_rows[index]["plugin_energy"])
            for index in nontrivial_indices)
    )
    spectral_seed_energy = np.sum(
        np.stack(per_seed_partition_energies, axis=0), axis=0
    )
    direct_seed_energy = np.mean(landscape.values**2, axis=0)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(direct_seed_energy))))
    total_error = abs(spectral_plugin_total - direct_plugin_total)
    nontrivial_error = abs(
        spectral_nontrivial_plugin - direct_nontrivial_plugin
    )
    per_seed_error = float(
        np.max(np.abs(spectral_seed_energy - direct_seed_energy))
    )
    diagnostics = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_mode": (
            "seed_aware_inference" if seed_inference else "descriptive_only"
        ),
        "seed_inference_available": seed_inference,
        "seed_inference_status": (
            "available" if seed_inference
            else "disabled_by_descriptive_only_configuration"
        ),
        "burnside_sum_d2": sum(
            int(row["coordinates_d2"]) for row in partition_rows
        ),
        "burnside_expected": group_order,
        "direct_plugin_total": direct_plugin_total,
        "spectral_plugin_total": spectral_plugin_total,
        "plugin_parseval_absolute_error": total_error,
        "direct_nontrivial_plugin": direct_nontrivial_plugin,
        "spectral_nontrivial_plugin": spectral_nontrivial_plugin,
        "nontrivial_parseval_absolute_error": nontrivial_error,
        "per_seed_parseval_max_absolute_error": per_seed_error,
        "parseval_tolerance": tolerance,
        "parseval_passed": bool(
            total_error <= tolerance
            and nontrivial_error <= tolerance
            and per_seed_error <= tolerance
        ),
    }
    if not diagnostics["parseval_passed"]:
        raise RuntimeError(f"spectral Parseval validation failed: {diagnostics}")

    return AuditResult(
        landscape=landscape,
        config=config,
        partition_rows=tuple(partition_rows),
        order_rows=tuple(order_rows),
        high_order_rows=(high_row,),
        allocation_rows=tuple(allocation_rows),
        diagnostics=diagnostics,
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return format(number, ".17g")
    return value


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        raise ValueError(f"cannot publish an empty table: {path.name}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"rows for {path.name} do not share an ordered schema")
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _implementation_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {
        path.name: sha256_file(path)
        for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name)
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def publish_audit(result: AuditResult, output_dir: str | Path) -> Path:
    """Atomically publish core CSVs, optional contrasts, and a manifest.

    The manifest is written last.  A present manifest therefore identifies a
    complete bundle and records a digest for every input and generated table.
    """

    if not isinstance(result, AuditResult):
        raise TypeError("result must be an AuditResult")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    tables = {
        "partition_spectrum.csv": result.partition_rows,
        "order_spectrum.csv": result.order_rows,
        "high_order_summary.csv": result.high_order_rows,
    }
    optional_allocation_path = destination / "within_order_allocation.csv"
    if result.allocation_rows:
        tables[optional_allocation_path.name] = result.allocation_rows
    elif optional_allocation_path.exists():
        optional_allocation_path.unlink()
    for name, rows in tables.items():
        _write_csv(destination / name, rows)

    outputs = {
        name: {
            "sha256": sha256_file(destination / name),
            "bytes": (destination / name).stat().st_size,
            "rows": len(rows),
        }
        for name, rows in tables.items()
    }
    landscape = result.landscape
    manifest = {
        "schema_version": 2,
        "analysis_version": ANALYSIS_VERSION,
        "package_version": __version__,
        "publication_protocol": "manifest_written_after_atomic_csv_bundle_v1",
        "input": {
            "path": landscape.source_path.name if landscape.source_path else None,
            "path_scope": "basename_only",
            "sha256": landscape.source_sha256,
            "permutations_key": landscape.permutations_key,
            "values_key": landscape.values_key,
            "seeds_key": landscape.seeds_key,
            "input_values_layout": landscape.input_layout,
            "canonical_values_layout": "group-by-seed",
            "canonical_permutation_order": "lexicographic",
            "K": landscape.K,
            "group_order": landscape.group_order,
            "n_seeds": landscape.n_seeds,
            "seed_ids": landscape.seed_ids,
        },
        "config": asdict(result.config),
        "diagnostics": result.diagnostics,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "implementation_sha256": _implementation_hashes(),
        "outputs": outputs,
    }
    with atomic_text_writer(manifest_path, newline="\n") as handle:
        json.dump(
            _json_safe(manifest),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return manifest_path
