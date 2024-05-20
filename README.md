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
