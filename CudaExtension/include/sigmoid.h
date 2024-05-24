// ./CxxExtension/src/sigmoid.h
#include <torch/torch.h>

// Forward propagation
at::Tensor sigmoid_forward(const at::Tensor& x);

// Backward propagation
at::Tensor sigmoid_backward(const at::Tensor& y, const at::Tensor& dy);

// Cuda functions
at::Tensor sigmoid_forward_cuda(at::Tensor& x);
at::Tensor sigmoid_backward_cuda(at::Tensor& y, at::Tensor& dy);
