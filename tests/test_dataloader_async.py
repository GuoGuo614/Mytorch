import gc
import pickle
import threading
import time

import numpy as np
import pytest

import mytorch as mt
from mytorch.data import DataLoader, Dataset
from mytorch.data.datasets import CIFAR10Dataset, NDArrayDataset


def _values(loader):
    batches = [features.numpy().reshape(-1) for features, _ in loader]
    return np.concatenate(batches) if batches else np.array([], dtype=np.int64)


@pytest.mark.parametrize("workers", [0, 1, 3])
def test_order_and_drop_last(workers):
    dataset = NDArrayDataset(np.arange(11), np.arange(11))
    loader = DataLoader(
        dataset, batch_size=4, num_workers=workers,
        prefetch_factor=2, drop_last=True,
    )
    np.testing.assert_array_equal(_values(loader), np.arange(8))
    assert len(loader) == 2
    assert loader.active_workers == 0


def test_sync_and_async_shuffle_match_across_epochs():
    dataset = NDArrayDataset(np.arange(29), np.arange(29))
    sync = DataLoader(dataset, batch_size=5, shuffle=True, seed=1729)
    asynchronous = DataLoader(
        dataset, batch_size=5, shuffle=True, seed=1729,
        num_workers=3, prefetch_factor=2,
    )
    sync_epochs = [_values(sync) for _ in range(3)]
    async_epochs = [_values(asynchronous) for _ in range(3)]
    for sync_values, async_values in zip(sync_epochs, async_epochs):
        np.testing.assert_array_equal(sync_values, async_values)
        np.testing.assert_array_equal(np.sort(sync_values), np.arange(29))
    assert not np.array_equal(sync_epochs[0], sync_epochs[1])


class DelayedDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        time.sleep(0.001 * (4 - int(index) % 5))
        return np.int64(index), np.int64(index)


def test_out_of_order_workers_still_yield_sampler_order():
    loader = DataLoader(
        DelayedDataset(), batch_size=2, num_workers=4, prefetch_factor=1
    )
    np.testing.assert_array_equal(_values(loader), np.arange(20))


class FailingDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        if index == 5:
            raise ValueError("synthetic worker failure")
        return np.int64(index), np.int64(index)


def test_worker_exception_is_reraised_and_workers_are_cleaned_up():
    loader = DataLoader(
        FailingDataset(), batch_size=2, num_workers=2, prefetch_factor=2
    )
    iterator = iter(loader)
    with pytest.raises(ValueError, match="synthetic worker failure"):
        list(iterator)
    assert iterator.active_workers == 0
    assert not iterator.state.producer.is_alive()


class InterruptingDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        raise KeyboardInterrupt("synthetic interrupt")


def test_worker_keyboard_interrupt_is_propagated_and_cleaned_up():
    iterator = iter(DataLoader(
        InterruptingDataset(), batch_size=1, num_workers=1
    ))
    with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
        next(iterator)
    assert iterator.active_workers == 0
    assert not iterator.state.producer.is_alive()


def test_early_close_and_implicit_break_release_resources():
    loader = DataLoader(
        DelayedDataset(), batch_size=1, num_workers=2, prefetch_factor=1
    )
    iterator = iter(loader)
    assert iterator.state.tasks.maxsize == loader.queue_capacity == 2
    assert iterator.state.results.maxsize == 2
    next(iterator)
    iterator.close()
    assert iterator.active_workers == 0
    assert not iterator.state.producer.is_alive()

    before = {thread.name for thread in threading.enumerate()
              if thread.name.startswith("mytorch-loader-")}
    for _ in loader:
        break
    gc.collect()
    after = {thread.name for thread in threading.enumerate()
             if thread.name.startswith("mytorch-loader-")}
    assert after == before


def test_starting_next_epoch_closes_partially_consumed_iterator():
    loader = DataLoader(
        DelayedDataset(), batch_size=2, shuffle=True, seed=9,
        num_workers=2, prefetch_factor=1,
    )
    first = iter(loader)
    next(first)
    second = iter(loader)
    assert first.active_workers == 0
    second_values = np.concatenate([
        values.numpy() for values, _ in second
    ])
    np.testing.assert_array_equal(np.sort(second_values), np.arange(20))
    assert second.active_workers == 0


def test_invalid_loader_configuration():
    dataset = NDArrayDataset(np.arange(2), np.arange(2))
    with pytest.raises(ValueError, match="batch_size"):
        DataLoader(dataset, batch_size=0)
    with pytest.raises(ValueError, match="num_workers"):
        DataLoader(dataset, num_workers=-1)
    with pytest.raises(ValueError, match="prefetch_factor"):
        DataLoader(dataset, num_workers=1, prefetch_factor=0)


def test_cifar10_reader_and_transform_contract(tmp_path):
    directory = tmp_path / "cifar-10-batches-py"
    directory.mkdir()
    flat = np.arange(2 * 3 * 32 * 32, dtype=np.int64).reshape(2, -1)
    flat = (flat % 256).astype(np.uint8)
    payload = {"data": flat, "labels": [3, 7]}
    for index in range(1, 6):
        with (directory / f"data_batch_{index}").open("wb") as file:
            pickle.dump(payload, file)
    with (directory / "test_batch").open("wb") as file:
        pickle.dump(payload, file)

    seen_shapes = []

    def transform(image):
        seen_shapes.append(image.shape)
        return image + 1

    dataset = CIFAR10Dataset(
        tmp_path, train=False, transforms=[transform], layout="CHW"
    )
    images, labels = next(iter(DataLoader(dataset, batch_size=2)))
    assert images.shape == (2, 3, 32, 32)
    assert labels.numpy().tolist() == [3, 7]
    assert seen_shapes == [(32, 32, 3), (32, 32, 3)]
    assert images.dtype == "float32"


def test_cifar10_missing_files_explain_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="cifar-10-python.tar.gz"):
        CIFAR10Dataset(tmp_path)


@pytest.mark.skipif(not mt.is_cuda_available(), reason="CUDA is unavailable")
def test_async_pinned_batch_transfer_to_cuda():
    dataset = NDArrayDataset(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.arange(6, dtype=np.int64),
    )
    loader = DataLoader(
        dataset, batch_size=3, device=mt.cuda(0), num_workers=2,
        prefetch_factor=2, pin_memory=True,
    )
    features, labels = next(iter(loader))
    assert features.device == mt.cuda(0)
    assert labels.device == mt.cuda(0)
    np.testing.assert_array_equal(features.numpy(), dataset.arrays[0][:3])
