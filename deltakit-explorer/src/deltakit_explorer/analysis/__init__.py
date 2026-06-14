# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.explorer.analysis`` namespace here."""

from deltakit_core.analysis import (
    DEFAULT_MAX_LIKELIHOOD_FACTOR,
    ProbabilityFit,
    asymmetric_yerr_from_fits,
    effective_stddev_from_fit,
    effective_stddev_from_fits,
    fit_binomial,
    fit_binomial_batch,
)

from deltakit_explorer.analysis._analysis import (
    get_exp_fit,
    get_lambda_fit,
)
from deltakit_explorer.analysis._lambda import (
    LambdaData,
    calculate_lambda_and_lambda_stddev,
)
from deltakit_explorer.analysis._leppr import (
    LogicalErrorProbabilityPerRoundData,
    calculate_lep_and_lep_fit,
    calculate_lep_and_lep_stddev,
    compute_logical_error_per_round,
    compute_logical_error_per_round_from_counts,
    simulate_different_round_numbers_for_lep_per_round_estimation,
)
from deltakit_explorer.analysis._quops import (
    predict_distance_for_quops,
    predict_quops_at_distance,
)

from . import error_budget

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]
