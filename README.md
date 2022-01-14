# SoftGym
<a href="https://sites.google.com/view/softgym/home">SoftGym</a> is a set of benchmark environments for deformable object manipulation including tasks involving fluid, cloth and rope. It is built on top of the Nvidia FleX simulator and has standard Gym API for interaction with RL agents. A number of RL algorithms benchmarked on SoftGym can be found in <a href="https://github.com/Xingyu-Lin/softagent">SoftAgent</a>

## Latest updates

- [12/06/2021] Support depth rendering. Example:`python examples/random_env.py --test_depth 1` to visualize the depth image.

## Using Docker
If you are using Ubuntu 16.04 LTS and CUDA 9.2, you can follow the steps in the next section on this page for compilation. For other versions of Ubuntu or CUDA, we provide the pre-built Docker image and Dockerfile for running SoftGym. Please refer to our [Docker](docker/docker.md) page.

## Instructions for Installation
1. This codebase is tested with Ubuntu 16.04 LTS, CUDA 9.2 and Nvidia driver version 440.64. Other versions might work but are not guaranteed, especially with a different driver version. Please use our docker for other versions.

The following command will install some necessary dependencies.
```
sudo apt-get install build-essential libgl1-mesa-dev freeglut3-dev libglfw3 libgles2-mesa-dev
```

2. Create conda environment
   Create a conda environment and activate it: `conda env create -f environment.yml`

3. Compile PyFleX: Go to the root folder of softgym and run `. ./prepare_1.0.sh`. After that, compile PyFleX with CMake & Pybind11 by running `. ./compile_1.0.sh` Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.

## SoftGym Environments
|Image|Name|Description|
|----------|:-------------|:-------------|
|![Gif](./examples/ClothDrop.gif)|[DropCloth](softgym/envs/cloth_drop.py) | Lay a piece of cloth in the air flat on the floor|
|![Gif](./examples/ClothFold.gif)|[FoldCloth](softgym/envs/cloth_fold.py) | Fold a piece of flattened cloth in half|
|![Gif](./examples/ClothFlatten.gif)|[SpreadCloth](softgym/envs/cloth_flatten.py)| Spread a crumpled cloth on the floor|
|![Gif](./examples/PourWater.gif)|[PourWater](softgym/envs/pour_water.py)| Pour a cup of water into a target cup |
|![Gif](./examples/PassWater.gif)|[TransportWater](softgym/envs/pass_water.py)| Move a cup of water to a target position as fast as possible without spilling out the water|
|![Gif](./examples/RopeFlatten.gif)|[StraightenRope](softgym/envs/rope_flatten.py)| Straighten a rope starting from a random configuration|
|![Gif](./examples/PourWaterAmount.gif)|[PourWaterAmount](softgym/envs/pour_water_amount.py)| This task is similar to PourWater but requires a specific amount of water poured into the target cup. The required water level is indicated by a red line.|
|![Gif](./examples/ClothFoldCrumpled.gif)|[FoldCrumpledCloth](softgym/envs/cloth_fold_crumpled.py)| This task is similar to FoldCloth but the cloth is initially crumpled| 
|![Gif](./examples/ClothFoldDrop.gif)|[DropFoldCloth](softgym/envs/cloth_fold_drop.py)| This task has the same initial state as DropCloth but requires the agent to fold the cloth instead of just laying it on the ground|
|![Gif](./examples/RopeConfiguration.gif)|[RopeConfiguration](softgym/envs/rope_configuration.py)| This task is similar to StraightenCloth but the agent needs to manipulate the rope into a specific configuration from different starting locations.|

To have a quick view of different tasks listed in the paper (with random actions), run the following commands:
For SoftGym-Medium:
- TransportWater: `python examples/random_env.py --env_name PassWater`
- PourWater: `python examples/random_env.py --env_name PourWater`
- StraightenRope: `python examples/random_env.py --env_name RopeFlatten`
- SpreadCloth: `python examples/random_env.py --env_name ClothFlatten`
- FoldCloth: `python examples/random_env.py --env_name ClothFold`
- DropCloth: `python examples/random_env.py --env_name ClothDrop`

For SoftGym-Hard:
- PourWaterAmount: `python examples/random_env.py --env_name PourWaterAmount`
- FoldCrumpledCloth: `python examples/random_env.py --env_name ClothFoldCrumpled`
- DropFoldCloth: `python examples/random_env.py --env_name ClothFoldDrop`
- RopeConfiguration:
  First download the rope configuration file using [this link](https://drive.google.com/file/d/1f3FK_7gwnJLVm3VaSacvYS7o-19-XPrr/view?usp=sharing) then run `python examples/random_env.py --env_name RopeConfiguration`

Turn on the `--headless` option if you are running on a cluster machine that does not have a display environment. Otherwise you will get segmentation issues. Please refer to `softgym/registered_env.py` for the default parameters and source code files for each of these environments.

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{corl2020softgym,
 title={SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation},
 author={Lin, Xingyu and Wang, Yufei and Olkin, Jake and Held, David},
 booktitle={Conference on Robot Learning},
 year={2020}
}
```

## References
- NVIDIA FleX - 1.2.0: https://github.com/NVIDIAGameWorks/FleX
- Our python interface builds on top of PyFleX: https://github.com/YunzhuLi/PyFleX
- If you run into problems setting up SoftGym, Daniel Seita wrote a nice blog that may help you get started on SoftGym: https://danieltakeshi.github.io/2021/02/20/softgym/
