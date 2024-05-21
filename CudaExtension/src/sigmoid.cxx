// ./CxxExtension/src/sigmoid.cxx
#include <iostream> // Include the iostream library
#include "sigmoid.h" // Include the header file

#define CHECK_IS_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_CONT(x) AT_ASSERTM(x.is_contiguous(),  #x " must be contiguous")
#define CHECK(x) CHECK_IS_CUDA(x); CHECK_IS_CONT(x)

// Define the forward function
at::Tensor sigmoid_forward(const at::Tensor& x) {
    CHECK(x); // Check if the input tensor is a CUDA tensor and contiguous
    return sigmoid_forward_cuda(x); // Call the CUDA forward function
}

// Define the backward function
at::Tensor sigmoid_backward(const at::Tensor& y, const at::Tensor& g) {
    at::Tensor grad = at::ones_like(g);

    CHECK(y); CHECK(g);
    return sigmoid_backward_cuda(y, g); // Call the CUDA backward function
}

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &sigmoid_forward, "sigmoid forward (CUDA)");
    m.def("backward", &sigmoid_backward, "sigmoid backward (CUDA)");
}
