import gym
import numpy as np
import pyflex
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.envs.rope_manipulate import RopeManipulateEnv
import os, argparse, sys
import softgym
from matplotlib import pyplot as plt
import torch, torchvision, cv2
from softgym.registered_env import  env_arg_dict
import argparse, time

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='debug')
args.add_argument("--headless", type=int, default=1)
args.add_argument("--obs_mode", type=str, default='cam_rgb')
args = args.parse_args()

env_name = 'PourWater'
dic = env_arg_dict[env_name]
dic['headless'] = 0
dic['observation_mode'] = 'cam_rgb'
dic['action_repeat'] = 8
env = PourWaterPosControlEnv(**dic)


env.reset()
beg = time.time()
print(beg)
for t in range(10000):
    a = env.action_space.sample()
    s, _, _, _ = env.step(a)

end = time.time()
print(end)
print("speed {}".format(10000 / (end - beg)))