from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.envs.dough_flatten import DoughFlattenEnv
from softgym.utils.visualization import save_numpy_as_gif
import click
import os.path as osp
import numpy as np
import torchvision
import torch
import os


@click.command()
@click.argument('headless', type=bool, default=True)
@click.argument('episode', type=int, default=16)
@click.argument('save_dir', type=str, default='./data/video/env_demos')
@click.argument('img_size', type=int, default=128)
@click.argument('use_cached_states', type=bool, default=True)
@click.option('--deterministic/--no-deterministic', default=False)
def main(headless, episode, save_dir, img_size, use_cached_states, deterministic):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    """ Generate demos for all environments with different variations"""
    envs = {
        # 'RopeFlatten': RopeFlattenEnv(
        #     observation_mode='cam_rgb',
        #     action_mode='picker',
        #     num_picker=2,
        #     render=True,
        #     headless=headless,
        #     horizon=75,
        #     action_repeat=8,
        #     render_mode='cloth',
        #     num_variations=200,
        #     use_cached_states=use_cached_states,
        #     deterministic=deterministic),
        # 'ClothFlatten': ClothFlattenEnv(
        #     observation_mode='key_point',
        #     action_mode='picker',
        #     num_picker=2,
        #     render=True,
        #     headless=headless,
        #     horizon=100,
        #     action_repeat=8,
        #     render_mode='cloth',
        #     num_variations=200,
        #     use_cached_states=use_cached_states,
        #     deterministic=deterministic),
        # 'ClothFold': ClothFoldEnv(
        #     observation_mode='key_point',
        #     action_mode='picker',
        #     num_picker=2,
        #     render=True,
        #     headless=headless,
        #     horizon=100,
        #     action_repeat=8,
        #     render_mode='cloth',
        #     num_variations=200,
        #     use_cached_states=use_cached_states,
        #     deterministic=deterministic),
        # 'PourWater': PourWaterPosControlEnv(
        #     observation_mode='cam_rgb',
        #     horizon=75,
        #     render=True,
        #     headless=headless,
        #     action_mode='direct',
        #     deterministic=False,
        #     render_mode='fluid'),
        'DoughFlatten': DoughFlattenEnv(
            observation_mode='cam_rgb',
            action_mode='direct',
            render=True,
            headless=False,
            horizon=75,
            action_repeat=8,
            render_mode='dough',
            num_variations=2,
            use_cached_states=True,
            deterministic=False)
    }

    for (env_name, env) in envs.items():
        all_frames = []
        for i in range(episode):
            frames = []
            env.reset()

            frames.append(env.get_image(img_size, img_size))
            for _ in range(30):
                action = env.action_space.sample()
                env.step(action)
                frames.append(env.get_image(img_size, img_size))
            all_frames.append(frames)
        # Convert to T x index x C x H x W for pytorch
        all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
        grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

        save_name = (env_name if not deterministic else env_name + '_deterministic') + '.gif'
        save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))


if __name__ == '__main__':
    main()
