"""Dimension-normalized allocation contrasts within an interaction order."""

from __future__ import annotations

import numpy as np

from .inference import (
    _finite,
    _jackknife_standard_error,
    _student_t_quantile_exact,
    _student_t_survival,
    plugin_energy,
    unbiased_signal_energy,
)


def _student_statistic(estimate: float, standard_error: float) -> float:
    if not np.isfinite(estimate):
        raise ValueError("estimate must be finite")
    if not np.isfinite(standard_error) or standard_error < 0.0:
        raise ValueError("standard_error must be finite and nonnegative")
    if standard_error > 0.0:
        return float(estimate / standard_error)
    if estimate > 0.0:
        return float("inf")
    if estimate < 0.0:
        return float("-inf")
    return 0.0


def _quadratic_acceptance_intervals(
    quadratic: float,
    linear: float,
    constant: float,
) -> list[tuple[float, float]]:
    """Return the subset of ``[0, 1]`` satisfying ``a q^2+b q+c <= 0``."""

    coefficients = np.asarray([quadratic, linear, constant], dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("confidence-set polynomial is nonfinite")
    scale = max(1.0, float(np.max(np.abs(coefficients))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    roots: list[float] = []
    if abs(quadratic) <= tolerance:
        if abs(linear) > tolerance:
            roots = [float(-constant / linear)]
        elif constant <= tolerance:
            return [(0.0, 1.0)]
        else:
            return []
    else:
        discriminant = linear**2 - 4.0 * quadratic * constant
        if discriminant >= -tolerance:
            square_root = float(np.sqrt(max(0.0, discriminant)))
            signed = square_root if linear >= 0.0 else -square_root
            first = (-linear - signed) / (2.0 * quadratic)
            second = (
                constant / (quadratic * first)
                if abs(first) > tolerance
                else (-linear + signed) / (2.0 * quadratic)
            )
            roots = [float(first), float(second)]

    breakpoints = [0.0, 1.0]
    breakpoints.extend(root for root in roots if 0.0 < root < 1.0)
    breakpoints = sorted(set(breakpoints))
    intervals: list[tuple[float, float]] = []
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        midpoint = (lower + upper) / 2.0
        value = (quadratic * midpoint + linear) * midpoint + constant
        if value <= tolerance:
            if intervals and abs(intervals[-1][1] - lower) <= tolerance:
                intervals[-1] = (intervals[-1][0], upper)
            else:
                intervals.append((lower, upper))
    return intervals


def _share_confidence_set(
    target_energy: float,
    total_energy: float,
    target_leave_one_out: np.ndarray,
    total_leave_one_out: np.ndarray,
    critical: float,
) -> dict[str, object]:
    """Invert the same jackknife-t contrast over shares in ``[0, 1]``."""

    target_loo = _finite(target_leave_one_out, "target_leave_one_out")
    total_loo = _finite(total_leave_one_out, "total_leave_one_out")
    if target_loo.ndim != 1 or total_loo.shape != target_loo.shape:
        raise ValueError("leave-one-out share inputs must be matching vectors")
    if len(target_loo) < 3 or not np.isfinite(critical) or critical <= 0.0:
        raise ValueError("invalid share confidence-set inputs")
    if (
        target_energy <= 0.0
        or total_energy <= 0.0
        or target_energy >= total_energy
        or np.any(target_loo <= 0.0)
        or np.any(total_loo <= 0.0)
        or np.any(target_loo >= total_loo)
    ):
        return {
            "available": False,
            "status": "nonpositive_or_out_of_range_signal_energy",
            "lower": float("nan"),
            "upper": float("nan"),
        }

    multiplier = (len(target_loo) - 1) / len(target_loo)
    centered_target = target_loo - target_loo.mean()
    centered_total = total_loo - total_loo.mean()
    variance_target = float(multiplier * centered_target @ centered_target)
    variance_total = float(multiplier * centered_total @ centered_total)
    covariance = float(multiplier * centered_target @ centered_total)
    critical_squared = critical**2
    intervals = _quadratic_acceptance_intervals(
        total_energy**2 - critical_squared * variance_total,
        -2.0 * target_energy * total_energy
        + 2.0 * critical_squared * covariance,
        target_energy**2 - critical_squared * variance_target,
    )
    if len(intervals) != 1:
        return {
            "available": False,
            "status": (
                "empty_confidence_set" if not intervals
                else "disconnected_confidence_set"
            ),
            "lower": float("nan"),
            "upper": float("nan"),
        }
    return {
        "available": True,
        "status": "ok",
        "lower": float(intervals[0][0]),
        "upper": float(intervals[0][1]),
    }


def describe_allocation(
    target_vectors: np.ndarray,
    other_vectors: np.ndarray,
    *,
    baseline_share: float,
) -> dict[str, object]:
    """Describe target allocation without making seed-level claims."""

    target = _finite(target_vectors, "target_vectors", min_ndim=2)
    other = _finite(other_vectors, "other_vectors", min_ndim=2)
    if target.shape[0] != other.shape[0]:
        raise ValueError("allocation inputs must use identical landscapes")
    if target.shape[1] == 0 or other.shape[1] == 0:
        raise ValueError("target and comparison blocks must both be nonempty")
    if not 0.0 < baseline_share < 1.0:
        raise ValueError("baseline_share must lie strictly between zero and one")
    target_plugin = plugin_energy(target)
    other_plugin = plugin_energy(other)
    total_plugin = target_plugin + other_plugin
    return {
        "plugin_target_energy": target_plugin,
        "plugin_other_same_order_energy": other_plugin,
        "plugin_order_total_energy": total_plugin,
        "plugin_target_share": (
            target_plugin / total_plugin if total_plugin > 0.0 else float("nan")
        ),
        "signal_target_energy_ustat": float("nan"),
        "signal_other_same_order_energy_ustat": float("nan"),
        "signal_order_total_energy_ustat": float("nan"),
        "signal_target_share_ustat": float("nan"),
        "signal_share_contrast_inversion_ci_lower": float("nan"),
        "signal_share_contrast_inversion_ci_upper": float("nan"),
        "signal_share_ci_available": False,
        "signal_share_ci_status": "descriptive_only_no_seed_inference",
        "allocation_contrast_ustat": float("nan"),
        "allocation_contrast_jackknife_se": float("nan"),
        "allocation_contrast_ci_lower": float("nan"),
        "allocation_contrast_ci_upper": float("nan"),
        "allocation_contrast_t": float("nan"),
        "allocation_contrast_df": 0,
        "allocation_p_raw_one_sided": float("nan"),
        "allocation_p_raw_two_sided": float("nan"),
        "inference_available": False,
        "inference_status": "descriptive_only_no_seed_inference",
        "inference_method": "not_available_descriptive_only",
    }


def infer_allocation(
    target_vectors: np.ndarray,
    other_vectors: np.ndarray,
    *,
    baseline_share: float,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Infer whether a target exceeds its coordinate share within an order.

    The primary statistic is the unbiased linear contrast
    ``E_target - q * E_order``.  Both terms are recomputed in every
    delete-one-landscape replicate.  Student-t calibration is approximate;
    no exact coefficient sign-flip claim is made for this composite null.
    """

    target = _finite(target_vectors, "target_vectors", min_ndim=2)
    other = _finite(other_vectors, "other_vectors", min_ndim=2)
    if target.shape[0] != other.shape[0]:
        raise ValueError("allocation inputs must use identical seeds")
    if target.shape[1] == 0 or other.shape[1] == 0:
        raise ValueError("target and comparison blocks must both be nonempty")
    if target.shape[0] < 3:
        raise ValueError("allocation inference requires at least three seeds")
    if not 0.0 < baseline_share < 1.0:
        raise ValueError("baseline_share must lie strictly between zero and one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    result = describe_allocation(target, other, baseline_share=baseline_share)
    target_energy = unbiased_signal_energy(target)
    other_energy = unbiased_signal_energy(other)
    total_energy = target_energy + other_energy
    n_seeds = target.shape[0]
    target_loo = np.empty(n_seeds, dtype=np.float64)
    other_loo = np.empty(n_seeds, dtype=np.float64)
    for index in range(n_seeds):
        target_loo[index] = unbiased_signal_energy(
            np.delete(target, index, axis=0)
        )
        other_loo[index] = unbiased_signal_energy(
            np.delete(other, index, axis=0)
        )
    total_loo = target_loo + other_loo
    contrast = target_energy - baseline_share * total_energy
    contrast_loo = target_loo - baseline_share * total_loo
    standard_error = _jackknife_standard_error(contrast_loo)
    statistic = _student_statistic(contrast, standard_error)
    degrees_freedom = n_seeds - 1
    critical = _student_t_quantile_exact(
        0.5 + confidence / 2.0, degrees_freedom
    )
    share_set = _share_confidence_set(
        target_energy,
        total_energy,
        target_loo,
        total_loo,
        critical,
    )
    result.update(
        {
            "signal_target_energy_ustat": target_energy,
            "signal_other_same_order_energy_ustat": other_energy,
            "signal_order_total_energy_ustat": total_energy,
            "signal_target_share_ustat": (
                target_energy / total_energy
                if target_energy > 0.0
                and other_energy > 0.0
                and total_energy > 0.0
                else float("nan")
            ),
            "signal_share_contrast_inversion_ci_lower": share_set["lower"],
            "signal_share_contrast_inversion_ci_upper": share_set["upper"],
            "signal_share_ci_available": share_set["available"],
            "signal_share_ci_status": share_set["status"],
            "allocation_contrast_ustat": contrast,
            "allocation_contrast_jackknife_se": standard_error,
            "allocation_contrast_ci_lower": contrast - critical * standard_error,
            "allocation_contrast_ci_upper": contrast + critical * standard_error,
            "allocation_contrast_t": statistic,
            "allocation_contrast_df": degrees_freedom,
            "allocation_p_raw_one_sided": _student_t_survival(
                statistic, degrees_freedom
            ),
            "allocation_p_raw_two_sided": min(
                1.0,
                2.0 * _student_t_survival(abs(statistic), degrees_freedom),
            ),
            "inference_available": True,
            "inference_status": "seed_aware_inference_available",
            "inference_method": "cross_seed_u_delete_one_jackknife_t",
        }
    )
    return result
