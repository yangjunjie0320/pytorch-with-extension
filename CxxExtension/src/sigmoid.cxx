// ./CxxExtension/src/sigmoid.cxx
#include <iostream> // Include the iostream library
#include "sigmoid.h" // Include the header file

// Define the forward function
at::Tensor sigmoid_forward(const at::Tensor& x) {
    at::Tensor y = at::zeros_like(x);

    for (auto i = 0; i < x.size(0); i++) {
        for (auto j = 0; j < x.size(1); j++) {
            y[i][j] = at::exp(-x[i][j]) / (1.0 + at::exp(-x[i][j]));
        }
    }

    return y;
}

// Define the backward function
at::Tensor sigmoid_backward(const at::Tensor& y, const at::Tensor& g) {
    at::Tensor grad = at::ones_like(g);

    for (auto i = 0; i < y.size(0); i++) {
        for (auto j = 0; j < y.size(1); j++) {
            grad[i][j] = y[i][j] * (y[i][j] - 1.0) * g[i][j];
        }
    }

    return grad;
}

// Define the Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sigmoid_forward, "sigmoid forward");
    m.def("backward", &sigmoid_backward, "sigmoid backward");
}
