# SoftGym

## Instructions 

1. Join GameWorks on Github
SoftGym builds on top of the FleX physics engine, which is open to the public upon joining the github team. Join on github by following this link: https://developer.nvidia.com/gameworks-source-github

2. This codebase is tested with Ubuntu 16.04 LTS, CUDA 9.2 and Nvidia driver version 440.64. Other versions might work but are not guaranteed, especially with a different driver version. 
The following command will install some necessary dependencies.

    sudo apt-get install build-essential libgl1-mesa-dev freeglut3-dev

3. Create conda environment
Create a conda environment and activate it: `conda env create -f environment.yml`

4. Merge FleX with SoftGym scenes and python interface by running `./merge_flex.sh`.

5. Compile PyFleX: Go to the root folder of softgym and run `. ./prepare_1.0.sh`. After that, compile PyFleX with CMake & Pybind11 by running `. ./compile_1.0.sh` Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.

### Run test examples    
```
python examples/random_env.py
```

## References
- NVIDIA FleX - 1.2.0 [README](doc/README_FleX.md)
- Our python interface build on top of PyFleX: https://github.com/YunzhuLi/PyFleX