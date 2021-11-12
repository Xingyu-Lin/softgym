# Running SoftGym inside a docker

We provide both Dockerfile and pre-built Docker container for compiling SoftGym. Part of the docker solutions are borrowed from [PyFlex](https://github.com/YunzhuLi/PyFleX/blob/master/bindings/docs/docker.md) 

## Prerequisite

- Install [docker-ce](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)
- Install [Anaconda](https://www.anaconda.com/distribution/)
- Install Pybind11 using `conda install pybind11`

## Running pre-built Dockerfile

- Pull the pre-built docker file

```
sudo docker pull xingyu/softgym
```

- Assuming you are using conda, using the following command to run docker, 
which will mount the python environment and SoftGym into the docker container. 
Make sure you have replaced `PATH_TO_SoftGym` and `PATH_TO_CONDA` with the corresponding paths (make sure to use absolute path!).

```
nvidia-docker run \
  -v PATH_TO_SoftGym:/workspace/softgym \
  -v PATH_TO_CONDA:PATH_TO_CONDA \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it xingyu/softgym:latest bash
```
As an example:
```
nvidia-docker run \
  -v ~/softgym:~/softgym \
  -v ~/software/miniconda3/:~/software/miniconda3/ \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -it xingyu/softgym:latest bash
```
This solution follows [this tutorial]( https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1) for running GL and CUDA application inside the docker. It is important to mount the conda path inside the docker container to be exactly the same as in your home machine.

- Now you are in the Docker environment. Go to the softgym directory and compile PyFlex

```
export PATH="PATH_TO_CONDA/bin:$PATH"
. ./prepare_1.0.sh && ./compile_1.0.sh
```


## Running with Dockerfile

We also posted the [Dockerfile](Dockerfile). To generate the pre-built file, download the Dockerfile in this directory and run
```
docker build -t softgym .
```
in the directory that contains the Dockerfile.
