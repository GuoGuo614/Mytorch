"""CuPy RawKernel for one-launch col2im gradient scattering."""

import cupy as cp


_KERNEL_TYPES = {
    "float16": ("half", "#include <cuda_fp16.h>"),
    "float32": ("float", ""),
    "float64": ("double", ""),
}
_KERNEL_CACHE = {}


def _kernel_for(dtype):
    dtype_name = str(dtype)
    if dtype_name not in _KERNEL_TYPES:
        raise TypeError(f"CUDA col2im does not support dtype {dtype}")
    if dtype_name in _KERNEL_CACHE:
        return _KERNEL_CACHE[dtype_name]
    scalar_type, header = _KERNEL_TYPES[dtype_name]
    source = f"""
    {header}
    extern "C" __global__ void col2im_scatter(
        const {scalar_type}* grad_columns,
        {scalar_type}* grad_x,
        long long start,
        long long rows,
        int in_channels,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int output_height,
        int output_width,
        int stride_height,
        int stride_width) {{
      long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
      long long kernel_elements =
          (long long)in_channels * kernel_height * kernel_width;
      long long count = rows * kernel_elements;
      if (index >= count) return;

      long long local_row = index / kernel_elements;
      int kernel_index = (int)(index - local_row * kernel_elements);
      long long position = start + local_row;
      int image = (int)(position / (output_height * output_width));
      int spatial = (int)(position % (output_height * output_width));
      int output_y = spatial / output_width;
      int output_x = spatial % output_width;
      int channel = kernel_index / (kernel_height * kernel_width);
      int kernel_spatial = kernel_index % (kernel_height * kernel_width);
      int input_y = output_y * stride_height + kernel_spatial / kernel_width;
      int input_x = output_x * stride_width + kernel_spatial % kernel_width;
      long long input_index =
          ((long long)image * in_channels + channel) * input_height * input_width
          + input_y * input_width + input_x;
      atomicAdd(grad_x + input_index, grad_columns[index]);
    }}
    """
    kernel = cp.RawKernel(source, "col2im_scatter", options=("--std=c++11",))
    _KERNEL_CACHE[dtype_name] = kernel
    return kernel


def col2im_scatter(grad_columns, grad_x, start, rows, shape):
    """Scatter one im2col chunk into grad_x with a single CUDA launch."""
    (_, in_channels, input_height, input_width, _, kernel_height,
     kernel_width, output_height, output_width, stride_height,
     stride_width) = shape
    count = rows * in_channels * kernel_height * kernel_width
    threads = 256
    blocks = (count + threads - 1) // threads
    _kernel_for(grad_x.dtype)(
        (blocks,), (threads,),
        (
            grad_columns, grad_x, start, rows, in_channels, input_height,
            input_width, kernel_height, kernel_width, output_height,
            output_width, stride_height, stride_width,
        ),
    )
