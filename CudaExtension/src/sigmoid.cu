#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 1024

template <typename scalar_t>
__global__ void sigmoid_forward_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, const int n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        auto expx = expf(-input[index]);
        output[index] = expx / (1.0 + expx);
    }
}

template <typename scalar_t>
__global__ void sigmoid_backward_kernel(const scalar_t* __restrict__ grad_output, const scalar_t* __restrict__ output, scalar_t* __restrict__ grad_input, const int n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        grad_input[index] = grad_output[index] * output[index] * (1.0 - output[index]);
    }
}

__host__ at::Tensor sigmoid_forward_cuda(const at::Tensor& input) {
    const auto n = input.numel();
    auto output = at::empty_like(input);
    const dim3 blocks((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const dim3 threads(THREADS_PER_BLOCK);
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_forward_cuda", ([&] {
        sigmoid_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            n
        );
    }));
    return output;
}

__host__ at::Tensor sigmoid_backward_cuda(const at::Tensor& grad_output, const at::Tensor& output) {
    const auto n = grad_output.numel();
    auto grad_input = at::empty_like(grad_output);
    const dim3 blocks((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    const dim3 threads(THREADS_PER_BLOCK);
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "sigmoid_backward_cuda", ([&] {
        sigmoid_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            n
        );
    }));
    return grad_input;
}