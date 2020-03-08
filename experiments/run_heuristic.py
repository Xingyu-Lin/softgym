from heuristics.pour_water_heuristic import run_heuristic as pourwater_heuristic
from heuristics.cloth_drop_heuristic import run_heuristic as clothdrop_heuristic
from heuristics.cloth_fold_heuristic import run_heuristic as clothfold_heuristic
from heuristics.rope_flatten_heuristic import run_heuristic as ropeflatten_heuristic
from heuristics.cloth_flatten_heuristic import run_heuristic as clothflatten_heuristic
from heuristics.pass_water_heuristic import run_heuristic as pass_water_heuristic
from softgym.utils.visualization import save_numpy_as_gif

import numpy as np
import argparse, sys
import cv2, torch, torchvision
import os.path as osp
import os

args = argparse.ArgumentParser(sys.argv[0])
args.add_argument("--mode", type=str, default='test')
args.add_argument("--headless", type=int, default=1)
args.add_argument("--obs_mode", type=str, default='point_cloud')
args.add_argument("--env_name", type=str, default='all')
args.add_argument("--use_cached_states", type=int, default=1)
args.add_argument("--imsize", type=int, default=256)
args = args.parse_args()

heuristic_funcs = {
    "pour_water": pourwater_heuristic,
    "cloth_drop": clothdrop_heuristic,
    "cloth_flatten": clothflatten_heuristic,
    "cloth_fold": clothfold_heuristic,
    "rope_flatten": ropeflatten_heuristic,
    "pass_water": pass_water_heuristic
}

def animation(all_frames, goal_image, save_dir='data/video', save_name='pourwater'):
    all_frames = np.asarray(all_frames)
    _, imwidth, imheight, imchannel = all_frames.shape
    all_frames = np.asarray(all_frames).reshape((8, -1, imwidth, imheight, imchannel))
    all_frames = np.array(all_frames).transpose([1, 0, 4, 2, 3])
    grid_imgs = [torchvision.utils.make_grid(torch.from_numpy(frame), nrow=4).permute(1, 2, 0).data.cpu().numpy() for frame in all_frames]

    save_numpy_as_gif(np.array(grid_imgs), osp.join(save_dir, save_name + '.gif'))
    goal_img = goal_image[:, :, ::-1]
    save_name =  osp.join(save_dir, save_name + '_goal.png')
    cv2.imwrite(save_name, goal_img)

def statistics(returns, final_performances):
    print("returns mean {}".format(np.mean(returns)))
    print("returns std {}".format(np.std(returns)))
    print("final performances mean {}".format(np.mean(final_performances)))

def plot_snapshots(imgs, goal_img, savepath='data/video/pour_water_goal.jpg'):
    num = 7
    show_imgs = []
    factor = len(imgs) // num
    for i in range(num):
        img = imgs[i * factor].transpose(2, 0, 1)
        print(img.shape)
        show_imgs.append(torch.from_numpy(img.copy()))

    # goal_img = goal_img.transpose(2, 0, 1)
    # show_imgs.append(torch.from_numpy(goal_img.copy()))
    goal_img = goal_img[:, :, ::-1]
    save_name =  savepath
    cv2.imwrite(save_name, goal_img)

    grid_imgs = torchvision.utils.make_grid(show_imgs, padding=20, pad_value=120).data.cpu().numpy().transpose(1, 2, 0)
    grid_imgs=grid_imgs[:, :, ::-1]
    print(savepath)
    cv2.imwrite(savepath, grid_imgs)

def run_single_env(args):
    returns, final_performances, imgs, goal_img = heuristic_funcs[args.env_name](args)
    if args.mode == 'test': # return the statistics of the heuristic policy, mainly mean and std return
        statistics(returns, final_performances)
    elif args.mode == 'visual': # plot the snapshots of the heuristic trajectory and the goal image. 
        plot_snapshots(imgs, goal_img, savepath='data/icml/' + args.env_name + '.jpg')
    elif args.mode == 'animation': # make a 1x4 animation of the heuristic policy, and goal image on the right.
        animation(imgs, goal_img, save_dir='data/video', save_name=args.env_name + args.imsize)

if args.env_name == 'all':
    for env_name in heuristic_funcs.keys():
        args.env_name = env_name
        args.headless = True
        run_single_env(args)

else:
    run_single_env(args)
