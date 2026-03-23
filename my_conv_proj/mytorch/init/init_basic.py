import math
import mytorch as torch


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random numbers uniform between low and high"""
    device = torch.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random normal with specified mean and std deviation"""
    device = torch.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)




def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = torch.cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)



def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate binary random Tensor"""
    device = torch.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return torch.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """Generate one-hot encoding Tensor"""
    device = torch.cpu() if device is None else device
    return torch.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
