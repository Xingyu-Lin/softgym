# TODO
- [ ] Make it into a python package that can be easily installed

# SoftGym

### Prerequisite
This codebase is tested with Ubuntu 16.04 LTS and CUDA 9.1. Other versions might work but are not guaranteed. Following command will install some necessary dependencies.

    sudo apt-get install build-essential libgl1-mesa-dev freeglut3-dev

### Compile PyFleX
1. Go to the root folder of softgym and run `. ./prepare.sh`
2. Compile PyFleX with CMake & Pybind11

    
    cd PyFlex/bindings/
    mkdir build; cd build; cmake ..; make -j
    
<!-- 1. Create a conda environment and activate it: `conda env create -f environment.yml && . activate softgym`-->


### PyFlex APIs

Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.


## References

- NVIDIA FleX - 1.2.0 [README](doc/README_FleX.md)
- PyFleX: https://github.com/YunzhuLi/PyFleX
