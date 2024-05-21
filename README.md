# pytorch-with-extension

## Build the C++ Extension
Currently, PyTorch has implemented many functions. However, in some scenarios, it is still necessary to use C++ or CUDA to customize some operations. These scenarios mainly include the following two types: (a) operations not yet supported by PyTorch; and (b) implementations in PyTorch are not efficient.

For these scenarios, specific functionalities can be achieved by writing C++ or CUDA extensions, thus achieving higher computational efficiency and accelerating the training of the network.

Compared to Python, C/C++ has inherent advantages in terms of low overhead. Therefore, for some complex operations, these can be implemented in C/C++ and then called through Python. Additionally, when implementing C++ extensions, users can customize the backward propagation functions instead of using PyTorch's autograd to automatically generate them, which is more efficient.
```bash
cd CxxExtension; conda env create -f environment.yml
conda activate $(grep 'name:' environment.yml | awk '{print $2}')
pip install .; python main.py; cd ..;
```


## Build the CUDA Extension

To successfully build CUDA extensions with PyTorch, you need both the PyTorch library with CUDA support and the NVIDIA CUDA Compiler (`nvcc`).

Prerequisites:

- **PyTorch with CUDA Support**: It's recommended to install PyTorch using Conda, which simplifies the installation process and ensures compatibility with CUDA, provided that your system has compatible NVIDIA drivers installed.
- **NVIDIA CUDA Toolkit**: Includes the `nvcc` compiler, which is essential for building CUDA extensions. Note it cannot be installed via `conda` and must be installed separately. You can download the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit) and follow the installation instructions [here](https://docs.nvidia.com/cuda/index.html).

After installing PyTorch, check that it recognizes CUDA, indicating that your installation 
includes CUDA support:

```bash
python -c "from torch.utils.cpp_extension import CUDA_HOME; print('CUDA HOME:', CUDA_HOME)"
```

If this command prints a path, CUDA is configured properly in PyTorch. If it returns `None`, you need to review your setup.

## Reference
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)

