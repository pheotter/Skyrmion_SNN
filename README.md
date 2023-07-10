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
### Google test functions in leaky.cpp and IEEE754.cpp
First, comment out the pybind11 codes at the bottom of these two files.
Then, executing these commands from the src directory:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Describe the idea
I'll use two methods to implement the idea and take the dataset MNIST for example.
### Method I
#### Architecture
- Each SK-RM represents one neuron
- I'll use 32 bits to represent the values of membrane potential, weights and bias (including the sign bit.)
- The first distance between two access ports stores the value of membrane potential for each neuron.
- The second distance between two access ports stores the rough value of the weight between this neuron and the first neuron of the prvious layer.
- The third distance between two access ports stores the rough value of the weight between this neuron and the second neuron of the prvious layer, and so on.
- The last distance between two access ports stores the rough value of the bias of this neuron.
Please see the picture below:
![image](https://github.com/pheotter/Skyrmion_SNN/blob/master/picture/sky1.png)
#### How to store values of weights and bias
- The sign bit is at the position 31. If the value is positive, then the position 31 will be empty. However, if the value is negative, the position 31 will exist a skyrmion.
- If the absolute value of a weight or bias is less than 1.0 and greater than or equal to (31/31+30/31)*0.5, the position 30 will contain a skyrmion.
- If the absolute value of a weight or bias is less than (31/31+30/31)*0.5 and greater than or equal to (30/31+29/31)*0.5, the position 29 will contain a skyrmion, and so on.
- If the value of a weight or bias is equal to 0, there will be no skyrmions from position 0 to position 31. It will insert a skyrmion on the right-hand-sided access port of the distance.
![image](https://github.com/pheotter/Skyrmion_SNN/blob/master/picture/sky2.png)
