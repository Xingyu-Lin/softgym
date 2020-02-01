from softgym.utils.visualization import save_numpy_as_gif
import click
import os.path as osp
import numpy as np
import torchvision
import torch
import os
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS


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

    """ Generate demos for all environments with different variations, as well as making generate cached states"""

    envs = []
    for env_name, env_class in SOFTGYM_ENVS.items():
        env_arg_dict[env_name]['render'] = True
        env_arg_dict[env_name]['headless'] = headless
        env_arg_dict[env_name]['observation_mode'] = 'point_cloud'
        env_arg_dict[env_name]['use_cached_states'] = use_cached_states
        env = env_class(**env_arg_dict[env_name])
        envs.append(env)
    for env_name, env in zip(SOFTGYM_ENVS.keys(), envs):
        all_frames = []
        for i in range(episode):
            frames = []
            env.reset()
            frames.append(env.get_image(img_size, img_size))
            for _ in range(env.horizon):
                action = env.action_space.sample()
                _, _, _, info = env.step(action, True, img_size)
                frames.extend(info['flex_env_recorded_frames'])
            all_frames.append(frames)
        # Convert to T x index x C x H x W for pytorch
        all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
        grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

        save_name = (env_name if not deterministic else env_name + '_deterministic') + '.gif'
        save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name))


if __name__ == '__main__':
    main()
