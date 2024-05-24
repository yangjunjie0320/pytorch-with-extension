// ./CxxExtension/src/sigmoid.cxx
#include <torch/torch.h>
#include <vector>
// #include "sigmoid.h" // Include the header file

// #define CHECK_IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_IS_CONT(x) AT_ASSERTM(x.is_contiguous(),  #x " must be contiguous")
// #define CHECK(x) CHECK_IS_CUDA(x); CHECK_IS_CONT(x)

at::Tensor sigmoid_forward_cuda(at::Tensor& x);
at::Tensor sigmoid_backward_cuda(at::Tensor& y, at::Tensor& dy);

// Define the forward function
at::Tensor sigmoid_forward(at::Tensor& x) {
    // Call the CUDA forward function
    // CHECK(x);  
    return sigmoid_forward_cuda(x); 
}

// Define the backward function
at::Tensor sigmoid_backward(at::Tensor& y, at::Tensor& dy) {
    // Call the CUDA backward function
    // CHECK(y); CHECK(dy); 
    return sigmoid_backward_cuda(y, dy); 
}

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &sigmoid_forward, "sigmoid forward (CUDA)");
    m.def("backward", &sigmoid_backward, "sigmoid backward (CUDA)");
}
