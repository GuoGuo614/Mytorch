# Needle / MyTorch

An educational deep-learning framework based on the 10-714 Needle homework.
The repository combines the broader Needle backend and sequence-model support
with the useful CNN additions from the former `my_conv_proj` implementation.

## Layout

- `python/needle`: tensors, automatic differentiation, operators, modules,
  optimizers, data loading, and NumPy/native backends
- `apps`: MNIST, CIFAR-10, and language-model examples
- `tests`: operator, backend, data, convolution, pooling, and sequence tests
- `src`: C++ CPU and CUDA ndarray backends
- `data`: local datasets used by the examples

## Merged CNN features

- PyTorch-style `nn.Conv2d` with configurable integer padding (`nn.Conv` remains valid)
- `nn.MaxPool2d` and `nn.AvgPool2d`
- differentiable `ops.pad` and `ops.logsoftmax`
- `MNISTConvNet`, a pooling-based CNN example in `apps/lenet5.py`

The pure NumPy ndarray backend works without compilation. Run `make` on a
supported Unix-like environment to build the optimized CPU backend and the
optional CUDA backend.
