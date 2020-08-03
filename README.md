# SoftGym

### Prerequisite
This codebase is tested with Ubuntu 16.04 LTS, CUDA 9.2 and Nvidia driver version 440.64. Other versions might work but are not guaranteed, especially with a different driver version. 

Following command will install some necessary dependencies.

    sudo apt-get install build-essential libgl1-mesa-dev freeglut3-dev

### Create conda environment
Create a conda environment and activate it: `conda env create -f environment.yml`

### Compile PyFleX
1. Go to the root folder of softgym and run `. ./prepare_1.0.sh`
2. Compile PyFleX with CMake & Pybind11 by running `. ./compile_1.0.sh`

### Run test examples    
```
python examples/random_env.py
```

### PyFleX APIs
Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.

## References

- NVIDIA FleX - 1.2.0 [README](doc/README_FleX.md)
- PyFleX: https://github.com/YunzhuLi/PyFleX