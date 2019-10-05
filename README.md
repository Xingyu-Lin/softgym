# SoftGym
1. Create a conda environment and activate it: `conda env create -f environment.yml && . activate softgym`

# TODO
- [ ] Make it into a python package that can be easily installed

# PyFleX

[NVIDIA FleX](https://developer.nvidia.com/flex) is an amazing particle-based simulator for simulating interactions between rigid bodies, fluids, clothing, etc. In this repo, we have developed Python bindings for setting up and interacting with a few [enviroments](#demo) that we can used to evaluate different simulation and control algorithms.

## Prerequisite

This codebase is tested with Ubuntu 16.04 LTS and CUDA 9.1. Other versions might work but are not guaranteed. Following command will install some necessary dependencies.

    sudo apt-get install build-essential libgl1-mesa-dev freeglut3-dev

## Compile and play with the original demo provided by NVIDIA FleX

First compile the demo

    cd demo/compiler/makelinux64/
    make -j

Then go to the target folder and start the demo!

    cd ../../../bin/linux64
    ./NvFlexDemoReleaseCUDA_x64


## Compile PyFleX with CMake & Pybind11

Go to the root folder of `PyFleX`, and set up paths

    export PYFLEXROOT=${PWD}
    export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
    export export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH

Compile PyFleX

    cd bindings/
    mkdir build; cd build; cmake ..; make -j

Try with `FluidFall` example. A window will pop up showing the simulation results.

    cd ${PYFLEXROOT}/bindings/examples
    python test_FluidFall.py


## Demo

Following we provided 6 environments for you to play with. Directly run the python scripts to see the simulation results. Screenshots will be stored in `${PYFLEXROOT}/bindings/examples/test_[env]/`.


**FluidFall** - Two drops of high-viscosity fluids are falling down and merging with each other.

    cd ${PYFLEXROOT}/bindings/examples
    python test_FluidFall.py

![](imgs/FluidFall.gif)


**BoxBath** - A block of water is flushing a rigid cube.

    cd ${PYFLEXROOT}/bindings/examples
    python test_BoxBath.py

![](imgs/BoxBath.gif)


**FluidShake** - Shake a box of fluids. The following script will first simulate the scene, and then playback the motion of the particles with the frontal wall removed for visualization.

    cd ${PYFLEXROOT}/bindings/examples
    python test_FluidShake.py

![](imgs/FluidShake.gif)


**RiceGrip** - Grip an object that can deform both elastically and plastically (e.g., sticky rice).

    cd ${PYFLEXROOT}/bindings/examples
    python test_RiceGrip.py

![](imgs/RiceGrip.gif)


**RigidFall** - A stack of rigid cubes falling down.

    cd ${PYFLEXROOT}/bindings/examples
    python test_RigidFall.py

![](imgs/RigidFall.gif)


**FluidIceShake** - Shake a box of fluids and a rigid cube. The following script will first simulate the scene, and then playback the motion of the particles with the frontal wall removed for visualization.

    cd ${PYFLEXROOT}/bindings/examples
    python test_FluidIceShake.py

![](imgs/FluidIceShake.gif)


## APIs

Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.


## References

- NVIDIA FleX - 1.2.0 [README](doc/README_FleX.md)
