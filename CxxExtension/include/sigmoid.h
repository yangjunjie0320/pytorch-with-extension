// ./CxxExtension/src/sigmoid.h
#include <torch/torch.h>

// Forward propagation
// f(x) = exp(-x) / (1 + exp(-x))
at::Tensor sigmoid_forward(const at::Tensor& x);

// Backward propagation
at::Tensor sigmoid_backward(const at::Tensor& y, const at::Tensor& g);
