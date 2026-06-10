import pytest

from deltakit_core.analysis import (
    DEFAULT_MAX_LIKELIHOOD_FACTOR,
    ProbabilityFit,
    asymmetric_yerr_from_fits,
    effective_stddev_from_fit,
    fit_binomial,
    fit_binomial_batch,
    log_binomial,
)


class TestProbabilityFit:
    def test_rejects_values_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            ProbabilityFit(low=-0.1, best=0.5, high=0.8)
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            ProbabilityFit(low=0.1, best=0.5, high=1.1)

    def test_rejects_invalid_ordering(self) -> None:
        with pytest.raises(ValueError, match="need low <= best <= high"):
            ProbabilityFit(low=0.6, best=0.5, high=0.8)


class TestLogBinomial:
    def test_rejects_p_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
            log_binomial(p=1.1, n=10, hits=2)

    def test_rejects_array_p_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="p must be in \\[0, 1\\]"):
            log_binomial(p=[0.1, 1.5], n=10, hits=2)


class TestFitBinomial:
    def test_matches_sinter_documented_examples(self) -> None:
        fit_large = fit_binomial(
            num_shots=100_000_000,
            num_hits=2,
            max_likelihood_factor=1000,
        )
        assert fit_large == pytest.approx(
            ProbabilityFit(low=2e-10, best=2e-08, high=1.259e-07), rel=0, abs=1e-12
        )

        fit_small = fit_binomial(num_shots=10, num_hits=5, max_likelihood_factor=9)
        assert fit_small.low == pytest.approx(0.202, abs=1e-3)
        assert fit_small.best == pytest.approx(0.5)
        assert fit_small.high == pytest.approx(0.798, abs=1e-3)

    def test_zero_shots_returns_full_interval(self) -> None:
        assert fit_binomial(num_shots=0, num_hits=0) == ProbabilityFit(
            low=0.0, best=0.5, high=1.0
        )

    def test_asymmetric_when_rate_near_zero(self) -> None:
        fit = fit_binomial(num_shots=1_000_000, num_hits=2, max_likelihood_factor=1000)
        assert fit.upper_margin > fit.lower_margin
        assert fit.best == pytest.approx(2e-6)

    def test_rejects_invalid_max_likelihood_factor(self) -> None:
        with pytest.raises(ValueError, match="max_likelihood_factor"):
            fit_binomial(num_shots=10, num_hits=1, max_likelihood_factor=0.5)

    def test_default_factor(self) -> None:
        fit = fit_binomial(num_shots=1000, num_hits=10)
        fit_default = fit_binomial(
            num_shots=1000,
            num_hits=10,
            max_likelihood_factor=DEFAULT_MAX_LIKELIHOOD_FACTOR,
        )
        assert fit == fit_default

    def test_batch_matches_scalar(self) -> None:
        shots = [10, 100_000_000]
        hits = [5, 2]
        low, best, high = fit_binomial_batch(num_shots=shots, num_hits=hits)
        for s, h, lo, be, hi in zip(shots, hits, low, best, high, strict=True):
            fit = fit_binomial(num_shots=s, num_hits=h, max_likelihood_factor=1000)
            assert lo == pytest.approx(fit.low)
            assert be == pytest.approx(fit.best)
            assert hi == pytest.approx(fit.high)


class TestEffectiveStddev:
    def test_average_of_margins(self) -> None:
        fit = fit_binomial(num_shots=1_000_000, num_hits=2, max_likelihood_factor=1000)
        assert effective_stddev_from_fit(fit) == pytest.approx(
            (fit.lower_margin + fit.upper_margin) / 2
        )


class TestAsymmetricYerr:
    def test_margins(self) -> None:
        fit = fit_binomial(num_shots=10, num_hits=5, max_likelihood_factor=9)
        lower, upper = asymmetric_yerr_from_fits([fit])
        assert lower[0] == pytest.approx(fit.lower_margin)
        assert upper[0] == pytest.approx(fit.upper_margin)
