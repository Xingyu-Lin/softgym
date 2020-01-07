import os
import cv2
import numpy as np
import imageio
import glob
from PIL import Image
from moviepy.editor import ImageSequenceClip


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


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

