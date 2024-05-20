# pytorch-with-extension

## Build
```bash
# create a new conda environment and activate it
conda env create -f environment.yml 
conda activate $(grep 'name:' environment.yml | awk '{print $2}')

# Build C++ extension
cd CxxExtension; pip install .; cd ..;

# Build CUDA extension
# TODO

python main.py
```

## Reference
- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)