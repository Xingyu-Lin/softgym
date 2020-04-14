import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np

class hvd_gather(object):

    def __init__(self, sess):

        rank = hvd.rank()
        self.np_array = np.ones(shape=(1,2), dtype=np.float32)*rank

        self.sess = sess
        self.tensor = tf.convert_to_tensor(self.np_array, dtype=tf.float32)

        self.gathered = hvd.allgather(self.tensor)

    def gather_and_print(self):
        return self.sess.run(self.gathered)



def test_horovod_allgather():
    """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    
    with tf.Session(config=config) as session:

        my_gather = hvd_gather(session)
        gathered_tensor = my_gather.gather_and_print()

        while True:
            print('gathered_tensor = ', gathered_tensor)


test_horovod_allgather()
