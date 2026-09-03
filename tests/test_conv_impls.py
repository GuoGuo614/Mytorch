import numpy as np
import pytest

import mytorch as mt
import mytorch.nn as nn


DEVICES = [mt.cpu()]
if mt.is_cuda_available():
    DEVICES.append(mt.cuda(0))


CASES = [
    # batch, in channels, out channels, height, width, kernel, stride, padding
    (1, 1, 1, 5, 7, 3, 1, 0),
    (2, 2, 3, 6, 5, 2, 2, 1),
    (1, 3, 2, 4, 7, 1, 1, 0),
    (2, 1, 2, 7, 6, 3, 2, 1),
    (1, 2, 2, 5, 8, (2, 3), (2, 1), (1, 0)),
]


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _run(case, implementation, device, max_im2col_bytes=64 * 1024 * 1024):
    batch, in_channels, out_channels, height, width, kernel, stride, padding = case
    kernel_h, kernel_w = _pair(kernel)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    rng = np.random.default_rng(100 + sum(case[:5]))
    x_values = rng.normal(size=(batch, in_channels, height, width)).astype(np.float32)
    weight_values = rng.normal(
        size=(out_channels, in_channels, kernel_h, kernel_w)
    ).astype(np.float32)
    bias_values = rng.normal(size=(1, out_channels, 1, 1)).astype(np.float32)
    padded_h, padded_w = height + 2 * pad_h, width + 2 * pad_w
    out_h = (padded_h - kernel_h) // stride_h + 1
    out_w = (padded_w - kernel_w) // stride_w + 1
    upstream_values = rng.normal(
        size=(batch, out_channels, out_h, out_w)
    ).astype(np.float32)

    x = mt.Tensor(x_values, device=device, requires_grad=True)
    weight = mt.Tensor(weight_values, device=device, requires_grad=True)
    bias = mt.Tensor(bias_values, device=device, requires_grad=True)
    padded = (
        mt.ops.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        if (pad_h, pad_w) != (0, 0) else x
    )
    conv_op = mt.ops.Conv2d(
        stride=stride,
        implementation=implementation,
        max_im2col_bytes=max_im2col_bytes,
    )
    conv_output = conv_op(padded, weight)
    output = conv_output + bias.broadcast_to(conv_output.shape)
    upstream = mt.Tensor(upstream_values, device=device, requires_grad=False)
    (output * upstream).sum().backward()
    return {
        "output": output.numpy(),
        "dx": x.grad.numpy(),
        "dw": weight.grad.numpy(),
        "db": bias.grad.numpy(),
        "selected": conv_op.selected_implementation,
        "chunk_rows": conv_op.chunk_rows,
    }


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("case", CASES)
def test_naive_and_im2col_forward_and_gradients_match(case, device):
    naive = _run(case, "naive", device)
    im2col = _run(case, "im2col", device)
    assert naive["selected"] == "naive"
    assert im2col["selected"] == "im2col"
    for name in ("output", "dx", "dw", "db"):
        np.testing.assert_allclose(
            naive[name], im2col[name], rtol=2e-5, atol=3e-5
        )


@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_im2col_chunks_large_column_matrices(device):
    case = CASES[1]
    result = _run(case, "im2col", device, max_im2col_bytes=512)
    batch, _, _, height, width, kernel, stride, padding = case
    kernel_h, kernel_w = _pair(kernel)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    total_rows = batch * (
        (height + 2 * pad_h - kernel_h) // stride_h + 1
    ) * ((width + 2 * pad_w - kernel_w) // stride_w + 1)
    assert 0 < result["chunk_rows"] < total_rows


def test_auto_selection_is_recorded_and_has_memory_fallback():
    x = mt.Tensor(np.ones((1, 1, 5, 7), dtype=np.float32))
    weight = mt.Tensor(np.ones((2, 1, 3, 3), dtype=np.float32))
    fast = mt.ops.Conv2d(implementation="auto")(x, weight)
    assert fast.op.selected_implementation == "im2col"

    limited = mt.ops.Conv2d(
        implementation="auto", max_im2col_bytes=64
    )(x, weight)
    assert limited.op.selected_implementation == "naive"
    with pytest.raises(MemoryError, match="im2col work row"):
        mt.ops.Conv2d(
            implementation="im2col", max_im2col_bytes=64
        )(x, weight)


def test_conv_module_supports_non_square_input_kernel_stride_and_padding():
    layer = nn.Conv2d(
        2, 3, kernel_size=(2, 3), stride=(2, 1), padding=(1, 0),
        implementation="auto",
    )
    output = layer(mt.Tensor(np.ones((1, 2, 5, 8), dtype=np.float32)))
    assert output.shape == (1, 3, 3, 6)
    assert layer.last_implementation == "im2col"
    assert layer.last_chunk_rows > 0


@pytest.mark.parametrize(
    ("x_shape", "weight_shape", "message"),
    [
        ((1, 3, 5), (2, 3, 3, 3), "NCHW"),
        ((1, 3, 5, 5), (2, 4, 3, 3), "channel mismatch"),
        ((1, 3, 2, 2), (2, 3, 3, 3), "kernel must fit"),
    ],
)
def test_conv_layout_and_shape_validation(x_shape, weight_shape, message):
    x = mt.Tensor(np.ones(x_shape, dtype=np.float32))
    weight = mt.Tensor(np.ones(weight_shape, dtype=np.float32))
    with pytest.raises(ValueError, match=message):
        mt.ops.conv2d(x, weight)
