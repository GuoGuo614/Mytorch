import numpy as np

import mytorch as mt
import mytorch.nn as nn
from mytorch.data import DataLoader
from mytorch.data.datasets import NDArrayDataset


def _mse(prediction, target):
    delta = prediction - target
    return (delta * delta).sum() / prediction.shape[0]


def test_tiny_linear_training_reduces_loss():
    np.random.seed(0)
    features = np.linspace(-1, 1, 32, dtype=np.float32).reshape(-1, 1)
    targets = 3 * features - 0.5
    loader = DataLoader(
        NDArrayDataset(features, targets), batch_size=8, shuffle=False
    )
    model = nn.Linear(1, 1)
    optimizer = mt.optim.SGD(model.parameters(), lr=0.1)

    initial = float(_mse(model(mt.Tensor(features)), mt.Tensor(targets)).numpy())
    for _ in range(20):
        for x, y in loader:
            optimizer.reset_grad()
            loss = _mse(model(x), y)
            loss.backward()
            optimizer.step()
    final = float(_mse(model(mt.Tensor(features)), mt.Tensor(targets)).numpy())
    assert final < initial * 0.05
