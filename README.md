# Spiking Neural Network on Skyrmion Racetrack Memory (SK-RM)
## Motivation
Training and inferencing neural network are computationally expensive in terms of both computation and memory access. Thus we utilize SK-RM as storage and perform computations on it at the same time based on the beneficial characters of this memory. However, we only implement the inference first due to difficulties of differential of functions in backpropagation on SK-RM.

## Getting Started
### Install Dependencies

1. **Python 3.8** - Follow instructions to install the latest version of python for your platform in the [python docs](https://docs.python.org/3/using/unix.html#getting-and-installing-the-latest-version-of-python)

2. **Virtual Environment** - We recommend working within a virtual environment whenever using Python for projects. This keeps your dependencies for each project separate and organized. Instructions for setting up a virual environment for your platform can be found in the [python docs](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)

3. **PIP Dependencies** - Once your virtual environment is setup and running, install the required dependencies by running:

```bash
pip3 install -r requirements.txt
```
All required packages are included in the requirements file.

4. **LibTorch** - Download ZIP archives containing the latest LibTorch distribution on [PyTorch website](https://pytorch.org/get-started/locally/) and unzip it.

### Set up the `LibTorch` path
Navigate to the src directory and open the CMakeLists.txt file to modify the `CMAKE_PREFIX_PATH`, where should be the absolute path to the unzipped LibTorch distribution.

### Use C++ Extension
We use C++ Extension to integrate our developed opration in PyTorch, since the `sky.h/sky.cpp` files are implemented in C++ and we want to reuse it. Those files simulate operations on SK-RM, including shift, detect, insert, delete, read and write.
Running:
```bash
python3 setup.py install
```
### Run the inference of MNist dataset
```bash
python3 function.py
```
