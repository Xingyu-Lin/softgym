import os
import cv2
import numpy as np
import imageio
import glob

def make_gif(video_path, video_name, video_idx_st, video_idx_en, video_height, video_width, clean_dir=True):
    """
    Compile a set of .tga files into a .gif and save to the same directory
    :param video_path: The path to the directory that contains all the frames, in the format of .tga
    :param video_name:
    :param video_idx_st: Start index of the frame
    :param video_idx_en: End index of the frame
    :param video_height:
    :param video_width:
    :param clean_dir: Delete all the render_*.tga in the video_path folder
    :return:
    """
    images = []
    for i in range(video_idx_st, video_idx_en):
        filename = os.path.join(video_path, 'render_%d.tga' % i)
        img = imageio.imread(filename)
        img = cv2.resize(img, (video_width, video_height), interpolation=cv2.INTER_AREA)
        images.append(img[:])

    imageio.mimsave(os.path.join(video_path, video_name + '.gif'), images, duration=1. / 60.)
    if clean_dir:
        print('Removing all render_*.tga files in {}'.format(video_path))
        for f in glob.glob(os.path.join(video_path, "render_*.tga")):
            os.remove(f)

