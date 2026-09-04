"""Dataset and deterministic synchronous/asynchronous data loading."""

from dataclasses import dataclass
import queue
import threading
import weakref
from typing import List, Optional

import numpy as np

from ..autograd import Tensor
from ..backend import Device, cpu


class Dataset:
    """Base class for indexable datasets."""

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            for transform in self.transforms:
                x = transform(x)
        return x

    def get_batch(self, indices):
        """Fetch scalar samples and stack corresponding components."""
        samples = [self[int(index)] for index in indices]
        if not samples:
            raise ValueError("cannot collate an empty batch")
        if isinstance(samples[0], (tuple, list)):
            return tuple(np.stack(component) for component in zip(*samples))
        return np.stack(samples)

    def set_epoch(self, epoch):
        """Optional hook for deterministic epoch-dependent transforms."""


def _fetch(dataset, indices):
    return dataset.get_batch(indices)


def _as_components(samples):
    if isinstance(samples, (tuple, list)):
        return samples
    return (samples,)


def _pinned_copy(array, device):
    """Copy one NumPy array through pinned memory, importing CuPy lazily."""
    host = np.ascontiguousarray(array)
    cp = device.xp
    allocation = cp.cuda.alloc_pinned_memory(host.nbytes)
    pinned = np.frombuffer(allocation, dtype=host.dtype, count=host.size).reshape(
        host.shape
    )
    pinned[...] = host
    # H2D happens here, in the main consumer thread.  The CuPy array owns the
    # device allocation before the temporary pinned host buffer is released.
    return cp.asarray(pinned)


def _to_tensors(samples, device, pin_memory):
    tensors = []
    for sample in _as_components(samples):
        if (pin_memory and device.kind == "cuda"
                and isinstance(sample, np.ndarray)):
            sample = _pinned_copy(sample, device)
        tensors.append(Tensor(sample, device=device))
    return tensors


class DataLoader:
    """Batch an indexable dataset with optional bounded thread prefetching.

    ``num_workers=0`` is the synchronous baseline. With workers enabled, raw
    host batches are produced on background threads and converted to Tensors
    (including any host-to-device copy) on the consuming thread.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        device: Optional[Device] = None,
        *,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        seed: int = 0,
        pin_memory: bool = False,
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if not isinstance(prefetch_factor, int) or prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer")
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if device is not None and not isinstance(device, Device):
            raise TypeError("device must be a mytorch Device or None")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.device = device or cpu()
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        # Accepted as a no-op on CPU so one configuration works on both paths.
        self.pin_memory = bool(pin_memory and self.device.kind == "cuda")
        self.ordering = []
        self._epoch = 0
        self._active_iterator = None

    @property
    def queue_capacity(self):
        return 0 if self.num_workers == 0 else self.num_workers * self.prefetch_factor

    @property
    def active_workers(self):
        iterator = self._active_iterator() if self._active_iterator else None
        return 0 if iterator is None else iterator.active_workers

    def set_epoch(self, epoch):
        """Set the next epoch number used by ordering and dataset transforms."""
        if not isinstance(epoch, (int, np.integer)) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        active = self._active_iterator() if self._active_iterator else None
        if active is not None:
            active.close()
        self._epoch = int(epoch)

    def _make_ordering(self, epoch):
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            # A separate generator prevents unrelated global NumPy randomness
            # and worker scheduling from changing sample order.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))
            indices = rng.permutation(indices)
        stop = len(indices)
        if self.drop_last:
            stop -= stop % self.batch_size
        return [indices[start:min(start + self.batch_size, stop)]
                for start in range(0, stop, self.batch_size)]

    def __iter__(self):
        active = self._active_iterator() if self._active_iterator else None
        if active is not None:
            active.close()
        ordering = self._make_ordering(self._epoch)
        self.dataset.set_epoch(self._epoch)
        self._epoch += 1
        self.ordering = ordering
        iterator = (_SyncDataLoaderIter(self, ordering)
                    if self.num_workers == 0
                    else _AsyncDataLoaderIter(self, ordering))
        self._active_iterator = weakref.ref(iterator)
        return iterator

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class _SyncDataLoaderIter:
    def __init__(self, loader, ordering):
        self.loader = loader
        self.ordering = ordering
        self.index = 0

    @property
    def active_workers(self):
        return 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.ordering):
            raise StopIteration
        indices = self.ordering[self.index]
        self.index += 1
        samples = _fetch(self.loader.dataset, indices)
        return _to_tensors(samples, self.loader.device, self.loader.pin_memory)

    def close(self):
        self.index = len(self.ordering)


@dataclass
class _WorkerResult:
    sequence: int
    samples: object = None
    error: BaseException = None


_STOP = object()


class _AsyncDataLoaderIter:
    def __init__(self, loader, ordering):
        self.state = _AsyncLoaderState(loader, ordering)
        self.next_sequence = 0
        self.buffer = {}
        self.state.start()

    @property
    def active_workers(self):
        return self.state.active_workers

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_sequence >= len(self.state.ordering):
            self.close()
            raise StopIteration
        try:
            while self.next_sequence not in self.buffer:
                try:
                    result = self.state.results.get(timeout=0.05)
                except queue.Empty:
                    if (not self.state.producer.is_alive()
                            and not self.active_workers):
                        self.close()
                        raise RuntimeError("DataLoader workers exited without a result")
                    continue
                if result.sequence == -1:
                    self.close()
                    raise result.error
                self.buffer[result.sequence] = result
            result = self.buffer.pop(self.next_sequence)
            self.next_sequence += 1
            self.state.slots.release()
            if result.error is not None:
                self.close()
                raise result.error
            batch = _to_tensors(
                result.samples,
                self.state.loader.device,
                self.state.loader.pin_memory,
            )
            if self.next_sequence >= len(self.state.ordering):
                self.close()
            return batch
        except BaseException:
            self.close()
            raise

    def close(self):
        self.state.close()
        self.buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def __del__(self):
        self.close()


class _AsyncLoaderState:
    """Thread-owned state kept separate so dropping an iterator can close it."""

    def __init__(self, loader, ordering):
        self.loader = loader
        self.ordering = ordering
        self.capacity = loader.queue_capacity
        self.tasks = queue.Queue(maxsize=self.capacity)
        self.results = queue.Queue(maxsize=self.capacity)
        self.slots = threading.BoundedSemaphore(self.capacity)
        self.stop_event = threading.Event()
        prefix = f"mytorch-loader-{id(self):x}"
        self.workers = [
            threading.Thread(
                target=_worker_loop,
                args=(self,),
                name=f"{prefix}-worker-{index}",
                daemon=True,
            )
            for index in range(loader.num_workers)
        ]
        self.producer = threading.Thread(
            target=_producer_loop,
            args=(self,),
            name=f"{prefix}-producer",
            daemon=True,
        )

    @property
    def active_workers(self):
        return sum(thread.is_alive() for thread in self.workers)

    def start(self):
        for worker in self.workers:
            worker.start()
        self.producer.start()

    def put(self, destination, item):
        while not self.stop_event.is_set():
            try:
                destination.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def close(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        for _ in range(self.capacity):
            try:
                self.slots.release()
            except ValueError:
                break
        current = threading.current_thread()
        for thread in [self.producer, *self.workers]:
            if thread is not current:
                thread.join(timeout=2.0)


def _producer_loop(state):
    try:
        for sequence, indices in enumerate(state.ordering):
            while not state.stop_event.is_set():
                if state.slots.acquire(timeout=0.05):
                    break
            else:
                return
            if not state.put(state.tasks, (sequence, indices)):
                state.slots.release()
                return
        for _ in state.workers:
            if not state.put(state.tasks, _STOP):
                return
    except BaseException as error:
        state.put(state.results, _WorkerResult(-1, error=error))


def _worker_loop(state):
    while not state.stop_event.is_set():
        try:
            task = state.tasks.get(timeout=0.05)
        except queue.Empty:
            continue
        if task is _STOP:
            return
        sequence, indices = task
        try:
            result = _WorkerResult(
                sequence, samples=_fetch(state.loader.dataset, indices)
            )
        except BaseException as error:
            result = _WorkerResult(sequence, error=error)
        if not state.put(state.results, result):
            return
