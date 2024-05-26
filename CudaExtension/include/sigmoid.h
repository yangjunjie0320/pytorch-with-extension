// ./CxxExtension/src/sigmoid.h
#include <torch/torch.h>

// Cuda functions
at::Tensor sigmoid_forw_cuda(at::Tensor& x);
at::Tensor sigmoid_back_cuda(at::Tensor& y, at::Tensor& dy);