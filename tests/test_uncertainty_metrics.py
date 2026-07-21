import numpy as np
import pytest

pytest.importorskip("mmcv")

from pyskl.utils.uncertainty_metrics import (
    predictive_quantities,
    validate_probabilities,
)


def test_probability_validity():
    probs = np.asarray([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64)
    validate_probabilities(probs)


def test_uncertainty_validity():
    prob_passes = np.asarray(
        [
            [[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]],
            [[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]],
        ],
        dtype=np.float64,
    )

    quantities = predictive_quantities(prob_passes)

    assert np.all(quantities["predictive_entropy"] >= 0)
    assert np.all(quantities["expected_entropy"] >= 0)
    assert np.all(quantities["mutual_information"] >= -1e-7)
    assert np.all(
        quantities["mutual_information"]
        <= quantities["predictive_entropy"] + 1e-7
    )
