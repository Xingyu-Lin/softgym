import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

import os, sys, argparse, logging
from collections import deque
from time import time

import numpy as np
from autolab_core import YamlConfig

from flex_gym.flex_horovod import set_flex_bin_path, make_flex_vec_env

class flex_gather_env_data(object):

    def __init__(self, sess):

        logging.getLogger().setLevel(logging.INFO)
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', '-c', type=str, default='cfg/ant.yaml')
        args = parser.parse_args()
        cfg = YamlConfig(args.cfg)

        set_flex_bin_path(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../bin'))
        self.env = make_flex_vec_env(cfg)
        logging.info('Total envs: {}'.format(self.env.num_envs))

        self.rank = hvd.rank()

        self.arr_obs = np.ones(shape=(1, self.env.num_envs * self.env.num_obs), dtype=np.float32)
        self.arr_rew = np.ones(shape=(1, self.env.num_envs), dtype=np.float32)
        self.arr_new = np.ones(shape=(1, self.env.num_envs), dtype=np.float32)


        self.arr_obs_tf_placeholder = tf.placeholder(shape=(1, self.env.num_envs * self.env.num_obs), dtype=tf.float32)
        self.arr_rew_tf_placeholder = tf.placeholder(shape=(1, self.env.num_envs), dtype=tf.float32)
        self.arr_new_tf_placeholder = tf.placeholder(shape=(1, self.env.num_envs), dtype=tf.float32)

        self.sess = sess

        self.obs_tensor  = self.arr_obs_tf_placeholder
        self.rew_tensor  = self.arr_rew_tf_placeholder
        self.new_tensor  = self.arr_new_tf_placeholder

        self.gathered_obs = hvd.allgather(self.obs_tensor)
        self.gathered_rew = hvd.allgather(self.rew_tensor)
        self.gathered_new = hvd.allgather(self.new_tensor)


    def get_num_obs(self):
        return self.env.num_obs

    def get_num_acts(self):
        return self.env.num_acts

    def get_num_envs(self):
        return self.env.num_envs

    def reset(self):

        obs = self.env.reset()
        self.arr_obs[0] = obs

        return self.sess.run(self.gathered_obs, feed_dict={self.arr_obs_tf_placeholder: self.arr_obs})

    def step(self, actions):

        obs, rew, new, info = self.env.step(actions)

        self.arr_obs[0] = obs
        self.arr_rew[0] = rew
        self.arr_new[0] = new

        return self.gather_tensors(),[{}]*hvd.size() * self.env.num_envs

    def gather_tensors(self):
        return self.sess.run([self.gathered_obs, self.gathered_rew, self.gathered_new],
                             feed_dict={self.arr_obs_tf_placeholder: self.arr_obs,
                                        self.arr_rew_tf_placeholder: self.arr_rew,
                                        self.arr_new_tf_placeholder: self.arr_new})


def test_horovod_allgather():
    """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    with tf.Session(config=config) as session:

        my_gather = flex_gather_env_data(session)

        num_acts, num_envs = my_gather.get_num_acts(), my_gather.get_num_envs()
        actions = np.random.rand(num_envs, num_acts) * 2 - 1

        gathered_tensors, _ = flex_gather_env_data.step(actions=actions)
        gathered_obs, gathered_rew, gathered_new = gathered_tensors[0], gathered_tensors[1], gathered_tensors[2]

        while True:
            print('gathered_obs_tensor = ', gathered_obs.shape, ' gathered_rew_tensor = ', gathered_rew.shape, ' gathered_new_tensor = ', gathered_new.shape)


test_horovod_allgather()
