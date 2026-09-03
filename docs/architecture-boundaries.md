# Architecture boundaries

## Canonical framework

`mytorch/` is the maintained framework. It contains the TensorOp computation
graph, automatic differentiation, NumPy/CuPy device layer, operators, neural
network modules, optimizers, and data utilities. Root-level applications import
only this package.

## Removed legacy framework

The previous Needle homework implementation and its custom C++/CUDA NDArray
backend were removed after V0. They were useful only as migration references;
their history remains available in Git. MyTorch must keep one computation graph
and must not copy the old Needle autograd system back into the runtime.

## Existing tracked data

The repository already tracks MNIST files under `data/MNIST/raw`. V0 leaves
those files untouched. New datasets and generated artifacts are ignored and
must not be added in later versions.
