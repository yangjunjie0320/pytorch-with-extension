// ./CxxExtension/src/sigmoid.h
#include <torch/torch.h>

#define CHECK_IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_CONT(x) AT_ASSERTM(x.is_contiguous(),  #x " must be contiguous")
#define CHECK(x) CHECK_IS_CUDA(x); CHECK_IS_CONT(x)

// Forward propagation
// f(x) = exp(-x) / (1 + exp(-x))
at::Tensor sigmoid_forward(const at::Tensor& x);

// Backward propagation
at::Tensor sigmoid_backward(const at::Tensor& y, const at::Tensor& g);
