import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mmcv")
import torch.nn as nn

from pyskl.utils.mc_dropout import enable_mc_dropout


class ToyDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.dropout(self.features(x)))


def test_enable_mc_dropout_modes():
    model = ToyDropoutNet()
    model.train()

    found = enable_mc_dropout(model)

    assert len(found) == 1
    assert model.dropout.training is True
    assert model.features[1].training is False
    assert model.features.training is False
    assert model.head.training is False


def test_dropout_forward_passes_are_stochastic():
    torch.manual_seed(123)
    model = ToyDropoutNet()
    enable_mc_dropout(model)
    x = torch.randn(16, 4)

    y1 = model(x)
    y2 = model(x)

    assert not torch.allclose(y1, y2)
