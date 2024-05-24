#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NUM_THREAD 1024

template <typename scalar_t>
__global__ void _forward_kernel(scalar_t* x, scalar_t* y, const int size) {
    const uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        y[index] = 1 / (1 + exp(-x[index]));
    }
}

template <typename scalar_t>
__global__ void _backward_kernel(scalar_t* y, scalar_t* dy, scalar_t* g, const int size) {
    const uint32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        g[index] = dy[index] * y[index] * (1 - y[index]);
    }
}

__host__ at::Tensor sigmoid_forward_cuda(at::Tensor& x) {
    const int size = x.numel();
    const int nthred = NUM_THREAD;
    const int nblock = (size + nthred - 1) / nthred;

    auto y = at::empty_like(x); // Create an output tensor y with the same shape as x

    // Dispatch the appropriate scalar type at runtime
    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "sigmoid_forward_cuda", 
        [&]() {
            _forward_kernel<scalar_t><<<nblock, nthred>>>(
                x.data<scalar_t>(), y.data<scalar_t>(), size
    );});

    return y;
}

__host__ at::Tensor sigmoid_backward_cuda(const at::Tensor& y, const at::Tensor& dy) {
    const int size = y.numel();
    const int nthred = NUM_THREAD;
    const int nblock = (size + nthred - 1) / nthred;

    auto g = dy.clone(); // Clone dy to create the output tensor g

    // Dispatch the appropriate scalar type at runtime
    AT_DISPATCH_FLOATING_TYPES(
        dy.scalar_type(), "sigmoid_backward_cuda",
        [&]() {
        _backward_kernel<scalar_t><<<nblock, nthred>>>(
            y.data<scalar_t>(), dy.data<scalar_t>(),
            g.data<scalar_t>(), size
    );});

    return g;
}