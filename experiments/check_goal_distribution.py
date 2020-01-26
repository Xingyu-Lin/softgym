from softgym.envs.rope_manipulate import RopeManipulate
from softgym.envs.cloth_manipulate import ClothManipulate
from softgym.envs.pour_water_multitask import PourWaterPosControlGoalConditionedEnv
from softgym.envs.pass_water_multitask import PassWater1DGoalConditionedEnv
import click
import os.path as osp
import numpy as np
import torchvision
import torch
import os
from matplotlib import pyplot as plt


@click.command()
@click.argument('headless', type=bool, default=True)
@click.argument('episode', type=int, default=5)
@click.argument('goal_num', type=int, default=1)
@click.argument('save_dir', type=str, default='./data/img/env_goals')
@click.argument('img_size', type=int, default=256)
@click.argument('use_cached_states', type=bool, default=True)
@click.option('--deterministic/--no-deterministic', default=False)
def main(headless, episode, goal_num, save_dir, img_size, use_cached_states, deterministic):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    """ Generate demos for all environments with different variations"""
    envs = {
        # 'ClothManipulate': ClothManipulate(
        #     observation_mode='cam_rgb',
        #     action_mode='picker',
        #     num_picker=2,
        #     render=True,
        #     headless=headless,
        #     horizon=75,
        #     action_repeat=8,
        #     render_mode='cloth',
        #     num_variations=10,
        #     goal_num=20,
        #     use_cached_states=use_cached_states,
        #     deterministic=deterministic),
        # 'RopeManipulate': RopeManipulate(
        #     observation_mode='cam_rgb',
        #     action_mode='picker',
        #     num_picker=2,
        #     render=True,
        #     headless=headless,
        #     horizon=100,
        #     action_repeat=8,
        #     render_mode='rope',
        #     num_variations=10,
        #     goal_num=20,
        #     use_cached_states=use_cached_states,
        #     deterministic=deterministic),
        'PassWaterGoalConditioned': PassWater1DGoalConditionedEnv(
            observation_mode = 'point_cloud', 
            horizon = 75, 
            action_mode = 'direct', 
            deterministic=True, 
            render_mode = 'fluid', 
            render = True, 
            headless= False,
            num_variations=10
        )
    }

    for (env_name, env) in envs.items():
        print("build goal distributions of env {}".format(env_name))
        fig = plt.figure(figsize=(episode * 3, (goal_num+1) * 3))
        for i in range(episode):
            env.reset()
            ax = fig.add_subplot(episode, goal_num+1, i*(goal_num+1) + 1)
            ax.imshow(env.get_image(img_size, img_size))
            for j in range(goal_num):
                ax = fig.add_subplot(episode, goal_num + 1, i*(goal_num+1) + 1 + j + 1)
                env.resample_goals()
                env.set_to_goal(env.get_goal())
                ax.imshow(env.get_image(img_size, img_size))

        save_name = (env_name if not deterministic else env_name + '_deterministic') + '.png'
        plt.tight_layout()
        plt.savefig(osp.join(save_dir, save_name))
        plt.close()


if __name__ == '__main__':
    main()
