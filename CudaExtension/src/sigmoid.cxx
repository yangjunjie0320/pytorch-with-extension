// ./CxxExtension/src/sigmoid.cxx
#include <iostream> // Include the iostream library
#include "sigmoid.h" // Include the header file

// Define the forward function
at::Tensor sigmoid_forward(const at::Tensor& x) {
    CHECK(x); // Check if the input tensor is a CUDA tensor and contiguous
    return sigmoid_cuda_forward(x); // Call the CUDA forward function
}

// Define the backward function
at::Tensor sigmoid_backward(const at::Tensor& y, const at::Tensor& g) {
    at::Tensor grad = at::ones_like(g);

    CHECK(y); CHECK(g);
    return sigmoid_cuda_backward(y, g); // Call the CUDA backward function
}

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &sigmoid_forward, "sigmoid forward (CUDA)");
    m.def("backward", &sigmoid_backward, "sigmoid backward (CUDA)");
}
