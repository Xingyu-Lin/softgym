import gym
import numpy as np
import pyflex
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.normalized_env import normalize
import torch, torchvision, cv2, time

def test_picker(num_picker=3, save_dir='./videos'):
    env = ClothFoldEnv(
        observation_mode='key_point',
        action_mode='picker',
        num_picker=2,
        render=True,
        headless=False,
        horizon=75,
        action_repeat=1,
        render_mode='cloth',
        num_variations=1000,
        deterministic=True,
        cached_init_state_path=None)

    imgs = []
    for _ in range(1):
        env.reset()
        print("picker low: ", env.action_tool.picker_low)
        print("picker high: ", env.action_tool.picker_high)
        print("clothsize: ", env.get_current_config()['ClothSize'])
        # env.action_tool.picker_high[0] = 1.0
        total_reward = 0

        pos = pyflex.get_positions().reshape((-1, 4))
        minx = np.min(pos[:, 0])
        maxx = np.max(pos[:, 0])
        minz = np.min(pos[:, 2])
        maxz = np.max(pos[:, 2])

        corner1 = np.array([minx, 0.05, minz])
        corner2 = np.array([minx, 0.05, maxz])
        print("corner1: ", corner1)
        print("corner2: ", corner2)

        picker_pos, _ = env.action_tool._get_pos()

        differ1 = corner1 - picker_pos[0]
        differ2 = corner2 - picker_pos[1]

        steps = 40
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            action[0, :3] = differ1 / steps
            action[1, :3] = differ2 / steps

            env.step(action)
            imgs.append(env.render('rgb_array'))

        print("=" * 50, "move to corner done!", "=" * 50)
        picker_pos, _ = env.action_tool._get_pos()
        print(picker_pos[0])
        print(picker_pos[1])
        time.sleep(5)
        
        steps = 20
        for i in range(steps):
            action = np.zeros((num_picker, 4))
            action[:,-1] = 1
            action[:, 1] = 0.02
            env.step(action)
            imgs.append(env.render('rgb_array'))

        print("=" * 50, "lift corner up done!", "=" * 50)

        target_corner_1 = np.array([maxx, 0.10, minz])
        target_corner_2 = np.array([maxx, 0.10, maxz])
        print("target_corner_1: ", target_corner_1)
        print("target_corner_2: ", target_corner_2)
        
        picker_pos, _ = env.action_tool._get_pos()
        print(picker_pos[0])
        print(picker_pos[1])

        differ1 = target_corner_1 - picker_pos[0]
        differ2 = target_corner_2 - picker_pos[1]
        
        steps = 500
        for i in range(steps):
            action = np.ones((num_picker, 4))
            action[0, :3] = differ1 / steps
            action[1, :3] = differ2 / steps
            env.step(action)
            imgs.append(env.render('rgb_array'))

        print("=" * 50, "move to target corner done!", "=" * 50)
        picker_pos, _ = env.action_tool._get_pos()
        print(picker_pos[0])
        print(picker_pos[1])

        # for i in range(60):
        #     print('step: ', i)
        #     action = np.zeros((num_picker, 4))
        #     if i < 12:
        #         action[0, :3] = (key_pos1 - picker_pos[0, :]) * 0.01
        #         action[1, :3] = (key_pos2 - picker_pos[1, :]) * 0.01
        #         action[:, 3] = 0
        #     elif i < 42:
        #         action[:, 1] = 0.005
        #         action[:, 0] = 0.01
        #         action[:, 3] = 1
        #     _, reward, _, _ = env.step(action)
        #     total_reward += reward
        #     print('total reward"', total_reward)
        #     img = env.render(mode='rgb_array')
        #     imgs.append(img)
    # fp_out = './videos/fold_picker_random_{}.gif'.format(num_picker)
    # save_numpy_as_gif(np.array(imgs), fp_out)

    num = 8
    show_imgs = []
    factor = len(imgs) // num
    for i in range(num):
        img = imgs[i * factor].transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))

    grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs=grid_imgs[:, :, ::-1]
    cv2.imwrite('cloth_fold.jpg', grid_imgs)


def test_random(env, N=5):
    N = 5
    for i in range(N):
        print('episode {}'.format(i))
        env.reset()
        for _ in range(env.horizon):
            action = env.action_space.sample()
            env.step(action)


if __name__ == '__main__':
    test_picker(num_picker=2)