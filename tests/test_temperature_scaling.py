import numpy as np
import pytest

pytest.importorskip("mmcv")

from pyskl.utils.temperature_scaling import (
    fit_temperature,
    mean_mc_probabilities,
    negative_log_likelihood,
)


def test_temperature_one_reproduces_uncalibrated_probabilities():
    logits = np.asarray(
        [
            [[2.0, 0.0], [1.8, 0.2]],
            [[0.1, 1.2], [0.0, 1.4]],
        ],
        dtype=np.float64,
    )

    raw = mean_mc_probabilities(logits)
    identity = mean_mc_probabilities(logits, temperature=1.0)

    assert np.allclose(raw, identity)


def test_fit_temperature_positive_and_nll_not_worse():
    logits = np.asarray(
        [
            [[4.0, 0.0], [3.5, 0.5]],
            [[0.0, 3.0], [0.2, 2.8]],
            [[2.5, 0.1], [2.2, 0.3]],
            [[0.3, 2.4], [0.1, 2.6]],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)

    before = negative_log_likelihood(mean_mc_probabilities(logits, temperature=1.0), labels)
    result = fit_temperature(logits, labels, num_bins=5)
    after = negative_log_likelihood(
        mean_mc_probabilities(logits, temperature=result["temperature"]),
        labels,
    )

    assert result["temperature"] > 0
    assert after <= before + 1e-8
