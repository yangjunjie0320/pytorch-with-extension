# pytorch-with-extension

## Build
```bash
conda create -n pytorch-with-extension -f requirements.txt -c pytorch
conda activate pytorch-with-extension

# Build C++ extension
cd CxxExtension; pip install .; cd ..;

# Build CUDA extension
# TODO

python main.py
```

## Reference
- [PyTorch C++ Frontend Tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
- [PyTorch CUDA Extension Tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)