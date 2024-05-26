// ./CxxExtension/src/sigmoid.cxx
#include "sigmoid.h" // Include the header file

#define CHECK_IS_CUDA(x) AT_ASSERTM(x.is_cuda(),        #x " must be a CUDA tensor")
#define CHECK_IS_CONT(x) AT_ASSERTM(x.is_contiguous(),  #x " must be contiguous")

// Define the forward function
at::Tensor sigmoid_forw(at::Tensor& x) {
    // Call the CUDA forward function
    CHECK_IS_CUDA(x); CHECK_IS_CONT(x);
    return sigmoid_forw_cuda(x); 
}

// Define the backward function
at::Tensor sigmoid_back(at::Tensor& y, at::Tensor& dy) {
    // Call the CUDA backward function
    CHECK_IS_CUDA(y);  CHECK_IS_CONT(y);
    CHECK_IS_CUDA(dy); CHECK_IS_CONT(dy);
    return sigmoid_back_cuda(y, dy); 
}

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &sigmoid_forw, "sigmoid forward  (CUDA)");
    m.def("backward", &sigmoid_back, "sigmoid backward (CUDA)");
}
