from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from deltakit_core.plotting.colours import RIVERLANE_PLOT_COLOURS
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_core.analysis import ProbabilityFit, asymmetric_yerr_from_fits
from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.plotting.plotting import plot
from deltakit_explorer.plotting.results import interpolate_leppr


def plot_logical_error_probability_per_round(
    leppr_data: LEPPRData,
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    logical_error_probability: npt.NDArray[np.float64] | Sequence[float],
    logical_error_probability_stddev: (
        npt.NDArray[np.float64] | Sequence[float] | None
    ) = None,
    logical_error_probability_fit: Sequence[ProbabilityFit] | None = None,
    *,
    num_sigmas: int = 3,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        leppr_data: Data class containing logical error probability per round
            fit results.
        num_rounds: a sequence of integers representing the number of rounds
            used to get the corresponding results in ``num_failed_shots`` and
            ``num_shots``.
        logical_error_probability: a sequence of floats representing the logical
            error probabilities corresponding to the number of rounds in
            ``num_rounds``.
        logical_error_probability_stddev: a sequence of floats representing the
            standard deviation of the logical error probabilities corresponding
            to the number of rounds in ``num_rounds``. If None, no error bars
            will be plotted unless ``logical_error_probability_fit`` is set.
            Mutually exclusive with ``logical_error_probability_fit``.
        logical_error_probability_fit: binomial likelihood intervals (e.g. from
            :func:`~deltakit_explorer.analysis.calculate_lep_and_lep_fit`). When set,
            asymmetric error bars are drawn and ``num_sigmas`` is ignored.
        num_sigmas: number of sigmas to consider when plotting symmetric error bars.
        fig: a matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: a matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Example:

        >>> from deltakit_explorer.analysis import (
        ...     calculate_lep_and_lep_fit,
        ...     compute_logical_error_per_round_from_counts,
        ... )
        >>> num_failed_shots = [34, 151, 356]
        >>> num_shots = [500000] * 3
        >>> num_rounds = [2, 4, 6]
        >>> res = compute_logical_error_per_round_from_counts(
        ...     num_rounds, num_failed_shots, num_shots
        ... )
        >>> lep, fits = calculate_lep_and_lep_fit(
        ...     fails=num_failed_shots, shots=num_shots
        ... )
        >>> fig, ax = plot_logical_error_probability_per_round(
        ...     res,
        ...     num_rounds=num_rounds,
        ...     logical_error_probability=lep,
        ...     logical_error_probability_fit=fits,
        ... )
    """
    if (fig is None) ^ (ax is None):
        msg = "The 'fig' and 'ax' parameters should either be both None or both set."
        raise ValueError(msg)

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    assert ax is not None
    assert fig is not None

    if logical_error_probability_stddev is not None and (
        logical_error_probability_fit is not None
    ):
        msg = (
            "Provide at most one of 'logical_error_probability_stddev' and "
            "'logical_error_probability_fit'."
        )
        raise ValueError(msg)
    lens = {len(num_rounds), len(logical_error_probability)}
    if logical_error_probability_stddev is not None:
        lens.add(len(logical_error_probability_stddev))
    if logical_error_probability_fit is not None:
        lens.add(len(logical_error_probability_fit))
    if len(lens) > 1:
        msg = (
            "The lengths of 'num_rounds', 'logical_error_probability', and any "
            "uncertainty sequence must match. Got lengths: "
            f"{lens}."
        )
        raise ValueError(msg)

    isort = np.argsort(num_rounds)
    num_rounds = np.asarray(num_rounds)[isort]
    logical_error_probability = np.asarray(logical_error_probability)[isort]
    yerr: npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...] | None = None
    error_label = f"Logical error probabilities (±{num_sigmas}σ)"  # noqa: RUF001
    if logical_error_probability_fit is not None:
        fits = [logical_error_probability_fit[i] for i in isort]
        yerr = asymmetric_yerr_from_fits(fits)
        error_label = "Logical error probabilities (likelihood interval)"
    elif logical_error_probability_stddev is not None:
        yerr = num_sigmas * np.asarray(logical_error_probability_stddev)[isort]

    ax.errorbar(
        num_rounds,
        logical_error_probability,
        yerr=yerr,
        fmt=".",
        color=RIVERLANE_PLOT_COLOURS[0],
        label=error_label,
    )

    leppr_result = interpolate_leppr(leppr_data, num_sigmas=num_sigmas)

    plot(leppr_result, fig=fig, ax=ax)

    return fig, ax
