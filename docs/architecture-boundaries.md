# Architecture boundaries

## Canonical framework

`mytorch/` is the maintained framework. It contains the TensorOp computation
graph, automatic differentiation, NumPy/CuPy device layer, operators, neural
network modules, optimizers, and data utilities. Root-level applications import
only this package.

V2 keeps one operator implementation per API and selects NumPy or CuPy from
the input array. Module migration is recursive, optimizer state follows its
parameter device, and only explicit transfers or reporting through
`Tensor.numpy()` may copy CUDA data to the host. The convolution implementation
uses separately testable naive and bounded im2col paths. Triton and custom CUDA
kernels remain outside V3.

## Removed legacy framework

The previous Needle homework implementation and its custom C++/CUDA NDArray
backend were removed after V0. They were useful only as migration references;
their history remains available in Git. MyTorch must keep one computation graph
and must not copy the old Needle autograd system back into the runtime.

## Existing tracked data

The repository already tracks MNIST files under `data/MNIST/raw`. V0 leaves
those files untouched. New datasets and generated artifacts are ignored and
must not be added in later versions.
