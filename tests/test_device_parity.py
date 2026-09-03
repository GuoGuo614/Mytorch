import numpy as np
import pytest

import mytorch as mt
import mytorch.nn as nn
from mytorch.data import DataLoader
from mytorch.data import RandomCrop, RandomFlipHorizontal
from mytorch.data.datasets import NDArrayDataset


def _assert_close(left, right, atol=2e-5, rtol=2e-5):
    np.testing.assert_allclose(left, right, atol=atol, rtol=rtol)


def _mlp(device):
    return nn.Sequential(
        nn.Linear(4, 6, device=device),
        nn.BatchNorm1d(6, device=device),
        nn.ReLU(),
        nn.Residual(nn.Sequential(nn.Linear(6, 6, device=device), nn.ReLU())),
        nn.LayerNorm1d(6, device=device),
        nn.Linear(6, 3, device=device),
    )


def _run_step(model, device, inputs, labels, optimizer_type=mt.optim.SGD):
    x = mt.Tensor(inputs, device=device, requires_grad=False)
    y = mt.Tensor(labels, device=device, requires_grad=False)
    optimizer = optimizer_type(model.parameters(), lr=0.01)
    optimizer.reset_grad()
    logits = model(x)
    loss = nn.SoftmaxLoss()(logits, y)
    loss.backward()
    gradients = [parameter.grad.numpy().copy() for parameter in model.parameters()]
    optimizer.step()
    parameters = [parameter.numpy().copy() for parameter in model.parameters()]
    return logits.numpy(), float(loss.numpy()), gradients, parameters


def test_module_to_and_state_dict_preserve_parameter_identity_dtype_and_buffers():
    model = _mlp(mt.cpu())
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    state = model.state_dict()
    assert "modules.1.running_mean" in state
    assert all(value.device == mt.cpu() for value in state.values())

    if mt.is_cuda_available():
        model.to(mt.cuda(0))
        assert parameter_ids == [id(parameter) for parameter in model.parameters()]
        assert all(parameter.device == mt.cuda(0) for parameter in model.parameters())
        assert all(buffer.device == mt.cuda(0) for _, buffer in model.named_buffers())
        assert all(parameter.dtype == np.float32 for parameter in model.parameters())
        model.load_state_dict(state)
        assert all(parameter.device == mt.cuda(0) for parameter in model.parameters())


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_mlp_cpu_cuda_forward_backward_and_optimizer_step_match():
    rng = np.random.default_rng(22)
    inputs = rng.normal(size=(8, 4)).astype(np.float32)
    labels = rng.integers(0, 3, size=8, dtype=np.int32)
    np.random.seed(11)
    initial = _mlp(mt.cpu()).state_dict()

    cpu_model = _mlp(mt.cpu())
    cpu_model.load_state_dict(initial)
    gpu_model = _mlp(mt.cuda(0))
    gpu_model.load_state_dict(initial)
    cpu_result = _run_step(cpu_model, mt.cpu(), inputs, labels)
    gpu_result = _run_step(gpu_model, mt.cuda(0), inputs, labels)

    _assert_close(cpu_result[0], gpu_result[0])
    _assert_close(cpu_result[1], gpu_result[1])
    for cpu_value, gpu_value in zip(cpu_result[2], gpu_result[2]):
        _assert_close(cpu_value, gpu_value)
    for cpu_value, gpu_value in zip(cpu_result[3], gpu_result[3]):
        _assert_close(cpu_value, gpu_value)


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_adam_state_and_updates_stay_on_cuda():
    model = nn.Linear(3, 2, device=mt.cuda(0))
    optimizer = mt.optim.Adam(model.parameters(), lr=0.001)
    x = mt.Tensor(np.ones((2, 3), dtype=np.float32), device=mt.cuda(0))
    model(x).sum().backward()
    optimizer.step()
    assert all(parameter.device == mt.cuda(0) for parameter in model.parameters())
    assert all(mt.device_of(value) == mt.cuda(0) for value in optimizer.m.values())
    assert all(mt.device_of(value) == mt.cuda(0) for value in optimizer.v.values())


def _conv_net(device):
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(8, 3, device=device),
    )


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_small_conv_cpu_cuda_forward_backward_and_sgd_step_match():
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(2, 1, 4, 4)).astype(np.float32)
    labels = np.array([0, 2], dtype=np.int32)
    np.random.seed(3)
    initial = _conv_net(mt.cpu()).state_dict()
    cpu_model = _conv_net(mt.cpu())
    cpu_model.load_state_dict(initial)
    gpu_model = _conv_net(mt.cuda(0))
    gpu_model.load_state_dict(initial)

    cpu_result = _run_step(cpu_model, mt.cpu(), inputs, labels, mt.optim.SGD)
    gpu_result = _run_step(gpu_model, mt.cuda(0), inputs, labels, mt.optim.SGD)
    _assert_close(cpu_result[0], gpu_result[0], atol=5e-5)
    _assert_close(cpu_result[1], gpu_result[1], atol=5e-5)
    for cpu_value, gpu_value in zip(cpu_result[2], gpu_result[2]):
        _assert_close(cpu_value, gpu_value, atol=5e-5)
    for cpu_value, gpu_value in zip(cpu_result[3], gpu_result[3]):
        _assert_close(cpu_value, gpu_value, atol=5e-5)


def test_dataloader_can_place_batches_on_requested_device():
    dataset = NDArrayDataset(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.arange(6, dtype=np.int32),
    )
    device = mt.cuda(0) if mt.is_cuda_available() else mt.cpu()
    features, labels = next(iter(DataLoader(dataset, batch_size=3, device=device)))
    assert features.device == device
    assert labels.device == device


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_data_transforms_keep_cuda_arrays_on_device():
    image = mt.asarray(
        np.arange(4 * 5 * 2, dtype=np.float32).reshape(4, 5, 2),
        device=mt.cuda(0),
    )
    flipped = RandomFlipHorizontal(p=1.0)(image)
    cropped = RandomCrop(padding=2)(image)
    assert mt.device_of(flipped) == mt.cuda(0)
    assert mt.device_of(cropped) == mt.cuda(0)
    assert flipped.shape == cropped.shape == image.shape
    _assert_close(mt.asnumpy(flipped), mt.asnumpy(image)[:, ::-1, :])


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
@pytest.mark.parametrize(
    ("dtype", "tolerance"), [("float32", 2e-5), ("float64", 1e-10)]
)
def test_core_operator_forward_backward_dtype_parity(dtype, tolerance):
    values = np.linspace(0.2, 1.4, 6, dtype=dtype).reshape(2, 3)

    def evaluate(device):
        x = mt.Tensor(values, device=device, requires_grad=True)
        expanded = x.transpose().reshape((3, 2)).broadcast_to((4, 3, 2))
        loss = mt.ops.logsumexp(expanded, axes=(0, 2)).sum()
        loss.backward()
        return float(loss.numpy()), x.grad.numpy(), loss.dtype

    cpu_loss, cpu_grad, cpu_dtype = evaluate(mt.cpu())
    gpu_loss, gpu_grad, gpu_dtype = evaluate(mt.cuda(0))
    _assert_close(cpu_loss, gpu_loss, atol=tolerance, rtol=tolerance)
    _assert_close(cpu_grad, gpu_grad, atol=tolerance, rtol=tolerance)
    assert cpu_dtype == gpu_dtype == np.dtype(dtype)


@pytest.mark.skipif(not mt.is_cuda_available(), reason="working CUDA/CuPy unavailable")
def test_remaining_operator_families_match_on_cpu_and_cuda():
    values = np.linspace(0.2, 1.8, 16, dtype=np.float32).reshape(1, 1, 4, 4)

    def evaluate(device):
        x = mt.Tensor(values, device=device, requires_grad=True)
        pooled = mt.ops.avg_pool2d(x, kernel_size=2, stride=2)
        exponent = mt.Tensor(
            np.full(pooled.shape, 1.5, dtype=np.float32),
            device=device,
            requires_grad=False,
        )
        nonlinear = (
            mt.ops.tanh(pooled)
            + mt.ops.sigmoid(pooled)
            + mt.ops.log(mt.ops.exp(pooled))
            + mt.ops.power(pooled, exponent)
        )
        first, second = mt.ops.fused_add_scalars(nonlinear, 0.2, 0.4)
        log_probs = mt.ops.logsoftmax(first.reshape((1, 4)))
        loss = log_probs.sum() + second.sum()
        loss.backward()
        prediction = mt.ops.argmax(log_probs, axis=1)
        return float(loss.numpy()), x.grad.numpy(), prediction.numpy()

    cpu_result = evaluate(mt.cpu())
    gpu_result = evaluate(mt.cuda(0))
    _assert_close(cpu_result[0], gpu_result[0], atol=5e-5)
    _assert_close(cpu_result[1], gpu_result[1], atol=5e-5)
    np.testing.assert_array_equal(cpu_result[2], gpu_result[2])
