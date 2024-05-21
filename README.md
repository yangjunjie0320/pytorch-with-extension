# pytorch-with-extension

## Build the C++ Extension
```bash
# create a new conda environment and activate it
conda env create -f environment.yml 
conda activate $(grep 'name:' environment.yml | awk '{print $2}')

# Build C++ extension
cd CxxExtension; pip install .; python main.py; cd ..;
```

Absolutely! Here’s an updated version of the guide with the requested adjustments, including emphasis on using Conda for PyTorch installation but not for `nvcc`.

---

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

