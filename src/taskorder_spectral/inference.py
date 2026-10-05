"""Seed-level energy estimators used by the generic spectral audit."""

from __future__ import annotations

from itertools import product
from statistics import NormalDist

import numpy as np


def _finite(values, name: str, *, min_ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if min_ndim is not None and array.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def per_seed_fourier(values: np.ndarray, representation: np.ndarray) -> np.ndarray:
    """Return Fourier matrices with shape ``(M, d, d)``."""

    matrix = _finite(values, "values")
    rho = _finite(representation, "representation")
    if matrix.ndim != 2:
        raise ValueError("values must have shape (G, M)")
    if rho.ndim != 3 or rho.shape[0] != matrix.shape[0] or rho.shape[1] != rho.shape[2]:
        raise ValueError("representation must have shape (G, d, d) with matching G")
    return np.einsum("gm,gij->mij", matrix, rho, optimize=True)


def plugin_energy(vectors: np.ndarray) -> float:
    values = _finite(vectors, "vectors", min_ndim=2)
    mean = values.mean(axis=0)
    return float(np.vdot(mean, mean).real)


def unbiased_signal_energy(vectors: np.ndarray) -> float:
    """All-cross-seed U-statistic for squared population-mean norm."""

    values = _finite(vectors, "vectors", min_ndim=2)
    if values.shape[0] < 2:
        raise ValueError("signal energy requires at least two seeds")
    n_seeds = values.shape[0]
    summed = values.sum(axis=0)
    cross = np.vdot(summed, summed).real - np.vdot(values, values).real
    return float(cross / (n_seeds * (n_seeds - 1)))


def _student_t_quantile(probability: float, degrees_freedom: int) -> float:
    """Validated Cornish-Fisher approximation used by the existing analyses."""

    if not 0.5 < probability < 1.0:
        raise ValueError("probability must lie in (0.5, 1)")
    if degrees_freedom < 1:
        raise ValueError("degrees_freedom must be positive")
    if abs(probability - 0.975) < 1e-12 and degrees_freedom in (1, 2, 3):
        return {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053}[
            degrees_freedom
        ]
    z = NormalDist().inv_cdf(probability)
    v = float(degrees_freedom)
    return float(
        z
        + (z**3 + z) / (4 * v)
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3)
    )


def jackknife_signal_interval(
    vectors: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, object]:
    values = _finite(vectors, "vectors", min_ndim=2)
    n_seeds = values.shape[0]
    if n_seeds < 3:
        raise ValueError("jackknife inference requires at least three seeds")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    estimate = unbiased_signal_energy(values)
    leave_one_out = np.array(
        [unbiased_signal_energy(np.delete(values, index, axis=0))
         for index in range(n_seeds)]
    )
    center = float(leave_one_out.mean())
    standard_error = float(
        np.sqrt(
            (n_seeds - 1) / n_seeds
            * np.sum((leave_one_out - center) ** 2)
        )
    )
    critical = _student_t_quantile(
        0.5 + confidence / 2.0,
        n_seeds - 1,
    )
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "lower": float(estimate - critical * standard_error),
        "upper": float(estimate + critical * standard_error),
        "critical": critical,
        "leave_one_out": leave_one_out,
    }


def sign_flip_patterns(
    n_seeds: int,
    *,
    max_exact_seeds: int = 16,
    n_draws: int = 20_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool]:
    """Return whole-seed sign patterns, modulo global-sign duplicates."""

    if n_seeds < 2:
        raise ValueError("sign-flip inference requires at least two seeds")
    if max_exact_seeds < 1 or n_draws < 1:
        raise ValueError("sign-pattern budgets must be positive")
    if n_seeds <= max_exact_seeds:
        tail = np.array(list(product((-1.0, 1.0), repeat=n_seeds - 1)))
        return np.column_stack([np.ones(len(tail)), tail]), True
    if rng is None:
        rng = np.random.default_rng(20260524)
    signs = rng.choice((-1.0, 1.0), size=(n_draws, n_seeds))
    signs[:, 0] = 1.0
    return signs, False


def sign_flip_pvalue(
    vectors: np.ndarray,
    signs: np.ndarray,
    *,
    exact: bool,
) -> dict[str, object]:
    values = _finite(vectors, "vectors", min_ndim=2)
    patterns = _finite(signs, "signs")
    if patterns.ndim != 2 or patterns.shape[1] != values.shape[0]:
        raise ValueError("signs must have shape (n_patterns, M)")
    if not np.all((patterns == -1.0) | (patterns == 1.0)):
        raise ValueError("every sign must equal -1 or +1")
    observed = plugin_energy(values)
    flipped_means = np.tensordot(patterns, values, axes=(1, 0)) / values.shape[0]
    null = np.einsum("s...,s...->s", flipped_means, flipped_means, optimize=True)
    tied = np.isclose(null, observed, rtol=1e-12, atol=0.0)
    exceedances = int(np.count_nonzero((null > observed) | tied))
    p_value = (
        exceedances / len(null)
        if exact
        else (exceedances + 1) / (len(null) + 1)
    )
    return {
        "p_value": float(p_value),
        "exact": bool(exact),
        "n_patterns": int(len(null)),
    }


def infer_vectors(
    vectors: np.ndarray,
    signs: np.ndarray,
    *,
    signs_exact: bool,
    confidence: float,
) -> dict[str, object]:
    """Plugin energy, signal U-statistic, interval, and whole-seed sign test."""

    interval = jackknife_signal_interval(vectors, confidence=confidence)
    sign_test = sign_flip_pvalue(vectors, signs, exact=signs_exact)
    return {
        "plugin_energy": plugin_energy(vectors),
        "signal_energy": interval["estimate"],
        "jackknife_se": interval["standard_error"],
        "signal_ci_lower": interval["lower"],
        "signal_ci_upper": interval["upper"],
        "seed_signflip_p_raw": sign_test["p_value"],
        "seed_signflip_exact": sign_test["exact"],
        "n_sign_patterns": sign_test["n_patterns"],
        "inference_available": True,
        "inference_status": "seed_aware_inference_available",
        "leave_one_out": interval["leave_one_out"],
    }


def describe_vectors(vectors: np.ndarray) -> dict[str, object]:
    """Return plug-in energy while explicitly suppressing seed inference.

    The inferential keys intentionally match :func:`infer_vectors`.  This
    keeps published CSV schemas stable while ensuring that a single realized
    landscape is never presented as if it supplied population-signal
    uncertainty.
    """

    values = _finite(vectors, "vectors", min_ndim=2)
    return {
        "plugin_energy": plugin_energy(values),
        "signal_energy": float("nan"),
        "jackknife_se": float("nan"),
        "signal_ci_lower": float("nan"),
        "signal_ci_upper": float("nan"),
        "seed_signflip_p_raw": float("nan"),
        "seed_signflip_exact": False,
        "n_sign_patterns": 0,
        "inference_available": False,
        "inference_status": "descriptive_only_no_seed_inference",
        "leave_one_out": np.empty((0,), dtype=np.float64),
    }


def _jackknife_standard_error(leave_one_out: np.ndarray) -> float:
    values = _finite(leave_one_out, "leave_one_out")
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("leave_one_out must be a vector of length at least three")
    centered = values - values.mean()
    return float(
        np.sqrt((len(values) - 1) / len(values) * np.vdot(centered, centered).real)
    )


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta via a stable continued fraction.

    This small implementation avoids making SciPy a runtime dependency solely
    for Student-t tail probabilities.  It follows the standard symmetry
    transform and modified-Lentz evaluation.
    """

    import math

    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("invalid regularized incomplete beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    def continued_fraction(first: float, second: float, value: float) -> float:
        maximum_iterations = 400
        epsilon = 3.0e-14
        tiny = np.finfo(np.float64).tiny / epsilon
        qab = first + second
        qap = first + 1.0
        qam = first - 1.0
        c = 1.0
        d = 1.0 - qab * value / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        result = d
        for iteration in range(1, maximum_iterations + 1):
            twice = 2 * iteration
            numerator = (
                iteration * (second - iteration) * value
                / ((qam + twice) * (first + twice))
            )
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            result *= d * c

            numerator = -(
                (first + iteration) * (qab + iteration) * value
                / ((first + twice) * (qap + twice))
            )
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= epsilon:
                return float(result)
        raise RuntimeError("incomplete-beta continued fraction did not converge")

    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * continued_fraction(a, b, x) / a
    else:
        value = 1.0 - front * continued_fraction(b, a, 1.0 - x) / b
    return float(min(1.0, max(0.0, value)))


def _student_t_survival(statistic: float, degrees_freedom: int) -> float:
    """Upper-tail probability for a Student-t random variable."""

    if degrees_freedom < 1:
        raise ValueError("degrees_freedom must be positive")
    if np.isnan(statistic):
        return float("nan")
    if statistic == float("inf"):
        return 0.0
    if statistic == float("-inf"):
        return 1.0
    x = degrees_freedom / (degrees_freedom + float(statistic) ** 2)
    half_tail = 0.5 * _regularized_incomplete_beta(
        x, degrees_freedom / 2.0, 0.5
    )
    return float(half_tail if statistic >= 0.0 else 1.0 - half_tail)


def _student_t_quantile_exact(
    probability: float,
    degrees_freedom: int,
) -> float:
    """Positive Student-t quantile obtained by monotone tail inversion."""

    if not 0.5 < probability < 1.0:
        raise ValueError("probability must lie in (0.5, 1)")
    if degrees_freedom < 1:
        raise ValueError("degrees_freedom must be positive")
    target_tail = 1.0 - probability
    lower, upper = 0.0, 1.0
    while _student_t_survival(upper, degrees_freedom) > target_tail:
        upper *= 2.0
        if not np.isfinite(upper):
            raise RuntimeError("could not bracket Student-t quantile")
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _student_t_survival(midpoint, degrees_freedom) > target_tail:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0)


def holm_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("p_values must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def jackknife_signal_fraction(
    numerator_vectors: np.ndarray,
    denominator_vectors: np.ndarray,
    *,
    confidence: float,
) -> dict[str, object]:
    """Ratio of cross-seed signal energies with joint delete-one recomputation."""

    numerator = _finite(numerator_vectors, "numerator_vectors", min_ndim=2)
    denominator = _finite(denominator_vectors, "denominator_vectors", min_ndim=2)
    if numerator.shape[0] != denominator.shape[0]:
        raise ValueError("fraction inputs must use identical seeds")
    n_seeds = numerator.shape[0]
    if n_seeds < 3:
        raise ValueError("fraction jackknife requires at least three seeds")

    numerator_energy = unbiased_signal_energy(numerator)
    denominator_energy = unbiased_signal_energy(denominator)
    if denominator_energy <= 0.0:
        return {
            "available": False,
            "status": "nonpositive_full_denominator",
            "estimate": float("nan"),
            "standard_error": float("nan"),
            "lower": float("nan"),
            "upper": float("nan"),
        }
    leave_one_out = np.empty(n_seeds, dtype=np.float64)
    for index in range(n_seeds):
        numerator_loo = unbiased_signal_energy(np.delete(numerator, index, axis=0))
        denominator_loo = unbiased_signal_energy(
            np.delete(denominator, index, axis=0)
        )
        if denominator_loo <= 0.0:
            return {
                "available": False,
                "status": "nonpositive_leave_one_out_denominator",
                "estimate": float("nan"),
                "standard_error": float("nan"),
                "lower": float("nan"),
                "upper": float("nan"),
            }
        leave_one_out[index] = numerator_loo / denominator_loo
    estimate = float(numerator_energy / denominator_energy)
    center = float(leave_one_out.mean())
    standard_error = float(
        np.sqrt(
            (n_seeds - 1) / n_seeds
            * np.sum((leave_one_out - center) ** 2)
        )
    )
    critical = _student_t_quantile(0.5 + confidence / 2.0, n_seeds - 1)
    return {
        "available": True,
        "status": "ok",
        "estimate": estimate,
        "standard_error": standard_error,
        "lower": float(estimate - critical * standard_error),
        "upper": float(estimate + critical * standard_error),
    }
