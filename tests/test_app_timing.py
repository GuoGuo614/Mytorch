import numpy as np
import pytest

from apps.timing import Stopwatch, format_duration
from mytorch.data import DataLoader
from mytorch.data.datasets import NDArrayDataset


def test_stopwatch_reports_elapsed_time():
    readings = iter((10.0, 12.5, 16.0))
    timer = Stopwatch(clock=lambda: next(readings))
    assert timer.elapsed() == 2.5
    assert timer.elapsed() == 6.0


def test_shuffled_dataloader_length_is_available_before_iteration():
    dataset = NDArrayDataset(np.arange(10), np.arange(10))
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    assert len(loader) == 3


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0.0s"), (12.34, "12.3s"), (65, "00:01:05"), (3661, "01:01:01")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected
