# (c) Copyright Riverlane 2020-2025.
"""Binomial likelihood intervals for logical error probability estimates.

Adapted from Stim's ``sinter._probability_util`` (Apache-2.0, quantumlib/Stim).
See https://quantumcomputing.stackexchange.com/a/37268/1386 for interpretation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class ProbabilityFit:
    """Low / best / high estimates compatible with observed hit counts.

    Attributes:
        low: Smallest rate whose binomial likelihood is within the configured factor
            of the maximum-likelihood rate.
        best: Maximum-likelihood estimate (``hits / shots``).
        high: Largest rate with likelihood within that factor of the best fit.
    """

    low: float
    best: float
    high: float

    @property
    def lower_margin(self) -> float:
        """Distance from ``best`` down to ``low``."""
        return self.best - self.low

    @property
    def upper_margin(self) -> float:
        """Distance from ``best`` up to ``high``."""
        return self.high - self.best


def log_factorial(n: int) -> float:
    """Natural log of ``n!``.

    Args:
        n: Non-negative integer.

    Returns:
        ``ln(n!)``.
    """
    return math.lgamma(n + 1)


def log_binomial(
    *, p: float | npt.NDArray[np.float64], n: int, hits: int
) -> npt.NDArray[np.float64]:
    r"""``ln P(hits | Binomial(n, p))`` with stable log-space arithmetic.

    Args:
        p: Success probability, scalar or array in ``[0, 1]``.
        n_trials: Number of trials.
        n_successes: Number of observed successes.

    Returns:
        Log-likelihood array with the same shape as ``p``.
    """
    p_clipped = np.clip(p, 0, 1)
    result: np.ndarray = np.zeros(shape=p_clipped.shape, dtype=np.float64)
    misses = n - hits

    if hits != 0:
        result[p_clipped == 0] = -np.inf
    if misses != 0:
        result[p_clipped == 1] = -np.inf

    nonzero = p_clipped != 0
    result[nonzero] += np.log(p_clipped[nonzero]) * float(hits)
    not_one = p_clipped != 1
    result[not_one] += np.log1p(-p_clipped[not_one]) * float(misses)

    log_n_choose_hits = log_factorial(n) - log_factorial(misses) - log_factorial(hits)
    result += log_n_choose_hits
    return result


def binary_search(
    *,
    func: Callable[[int], float],
    min_x: int,
    max_x: int,
    target: float,
) -> int:
    """Binary search on a monotonically ascending integer-valued function.

    Args:
        func: Monotonically ascending function of an integer argument.
        min_x: Lower bound of the search interval (inclusive).
        max_x: Upper bound of the search interval (inclusive).
        target: Value to search for.

    Returns:
        Integer in ``[min_x, max_x]`` where ``func(x)`` is closest to ``target``.
    """
    while max_x > min_x + 1:
        med_x = (min_x + max_x) // 2
        out = func(med_x)
        if out < target:
            min_x = med_x
        elif out > target:
            max_x = med_x
        else:
            return med_x
    fmax = func(max_x)
    fmin = func(min_x)
    dmax = 0 if fmax == target else fmax - target
    dmin = 0 if fmin == target else fmin - target
    return max_x if abs(dmax) < abs(dmin) else min_x


DEFAULT_MAX_LIKELIHOOD_FACTOR: float = 1000.0


def fit_binomial(
    *,
    num_shots: int,
    num_hits: int,
    max_likelihood_factor: float = DEFAULT_MAX_LIKELIHOOD_FACTOR,
) -> ProbabilityFit:
    """Binomial likelihood interval for an error rate given shot outcomes.

    Args:
        num_shots: Number of independent samples.
        num_hits: Number of failures (logical errors) observed.
        max_likelihood_factor: Rates whose likelihood is below ``best_likelihood /
            max_likelihood_factor`` are excluded from the interval. Must be ``>= 1``.

    Returns:
        ``ProbabilityFit`` with ``best = num_hits / num_shots`` and asymmetric ``low`` /
        ``high`` bounds.

    Raises:
        ValueError: If ``max_likelihood_factor < 1``, ``num_hits < 0``,
            ``num_shots < 0``, or ``num_hits > num_shots``.
    """
    if max_likelihood_factor < 1:
        msg = f"max_likelihood_factor={max_likelihood_factor} must be greater or equal than 1."
        raise ValueError(msg)
    if num_hits < 0 or num_shots < 0 or num_hits > num_shots:
        msg = (
            f"need 0 <= num_hits={num_hits} <= num_shots={num_shots}, got invalid range"
        )
        raise ValueError(msg)
    if num_shots == 0:
        return ProbabilityFit(low=0.0, best=0.5, high=1.0)

    best = num_hits / num_shots
    log_max_likelihood = log_binomial(p=best, n=num_shots, hits=num_hits)
    target_log_likelihood = log_max_likelihood - math.log(max_likelihood_factor)
    acc = 100
    low = (
        binary_search(
            func=lambda exp_err: float(
                log_binomial(p=exp_err / (acc * num_shots), n=num_shots, hits=num_hits)
            ),
            target=float(target_log_likelihood),
            min_x=0,
            max_x=num_hits * acc,
        )
        / acc
    )
    high = (
        binary_search(
            func=lambda exp_err: float(
                -log_binomial(p=exp_err / (acc * num_shots), n=num_shots, hits=num_hits)
            ),
            target=float(-target_log_likelihood),
            min_x=num_hits * acc,
            max_x=num_shots * acc,
        )
        / acc
    )
    return ProbabilityFit(
        best=best,
        low=low / num_shots,
        high=high / num_shots,
    )


def fit_binomial_batch(
    *,
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    num_hits: npt.NDArray[np.int_] | Sequence[int],
    max_likelihood_factor: float = DEFAULT_MAX_LIKELIHOOD_FACTOR,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Vectorised wrapper around :func:`fit_binomial`.

    Args:
        num_shots: Number of independent samples per element.
        num_hits: Number of failures observed per element.
        max_likelihood_factor: Passed to each :func:`fit_binomial` call.

    Returns:
        Tuple ``(low, best, high)`` arrays with the same length as the inputs.

    Raises:
        ValueError: When ``num_shots`` and ``num_hits`` have different shapes.
    """
    shots = np.asarray(num_shots, dtype=np.int_)
    hits = np.asarray(num_hits, dtype=np.int_)
    if shots.shape != hits.shape:
        msg = "num_shots and num_hits must have the same shape."
        raise ValueError(msg)
    fits = [
        fit_binomial(
            num_shots=int(s),
            num_hits=int(h),
            max_likelihood_factor=max_likelihood_factor,
        )
        for s, h in zip(shots.ravel(), hits.ravel(), strict=True)
    ]
    low = np.array([f.low for f in fits], dtype=np.float64).reshape(shots.shape)
    best = np.array([f.best for f in fits], dtype=np.float64).reshape(shots.shape)
    high = np.array([f.high for f in fits], dtype=np.float64).reshape(shots.shape)
    return low, best, high


def effective_stddev_from_fit(fit: ProbabilityFit) -> float:
    """Symmetric Gaussian stand-in for WLS from binomial likelihood margins.

    Args:
        fit: Binomial likelihood interval to summarise.

    Returns:
        Average of lower and upper margins as a single sigma value.
    """
    return (fit.lower_margin + fit.upper_margin) / 2


def effective_stddev_from_fits(
    fits: Sequence[ProbabilityFit],
) -> npt.NDArray[np.float64]:
    """Per-point :func:`effective_stddev_from_fit` values.

    Args:
        fits: Sequence of binomial likelihood intervals.

    Returns:
        Array of effective standard deviations, one per input fit.
    """
    return np.asarray([effective_stddev_from_fit(f) for f in fits], dtype=np.float64)


def asymmetric_yerr_from_fits(
    fits: Sequence[ProbabilityFit],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Lower and upper matplotlib ``yerr`` margins relative to each fit's ``best``.

    Args:
        fits: Sequence of binomial likelihood intervals.

    Returns:
        Tuple ``(lower_margin, upper_margin)`` arrays for use as matplotlib ``yerr``.
    """
    low = np.asarray([f.low for f in fits], dtype=np.float64)
    best = np.asarray([f.best for f in fits], dtype=np.float64)
    high = np.asarray([f.high for f in fits], dtype=np.float64)
    return best - low, high - best
