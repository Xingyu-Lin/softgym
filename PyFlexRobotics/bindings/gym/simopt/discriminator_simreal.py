import tensorflow as tf
import numpy as np
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U
from baselines.common import dataset, fmt_row
from baselines import logger
from baselines.common.mpi_adam import MpiAdam

class DiscriminatorSimReal:
    def __init__(self, input_shape,
                 num_hidden_layer=2, hidden_size=64, activation_fn=tf.nn.relu,
                 entcoeff=0.0, lr_rate=1e-3, gan_type='lsgan',
                 scope="discriminator", idx=None):
        self.scope = scope
        self.input_shape = input_shape
        self.num_hidden_layer = num_hidden_layer
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.generator_data_ph = tf.placeholder(tf.float32, [None, self.input_shape], name="generator_data_ph")
        self.expert_data_ph = tf.placeholder(tf.float32, [None, self.input_shape], name="expert_data_ph")

        # Report
        print("=== Discriminator {} ===\n"
              "GAN type: {}\n"
              "Learning rate: {}\n"
              "Entcoeff: {}\n"
              "Num of hidden layers: {}\n"
              "Hidden size: {}\n"
              "Hidden activation: {}".format(
            idx, gan_type, lr_rate, entcoeff,
            num_hidden_layer, hidden_size, activation_fn.__name__))

        # Build grpah
        generator_logits = self.build_graph(self.generator_data_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_data_ph, reuse=True)

        # https: // github.com / ankurhanda / tf - unet / blob / master / UNet.py
        # classes = tf.cast(tf.argmax(prediction, 3), tf.uint8)
        # pt = tf.nn.softmax(prediction)
        # pt = tf.reshape(pt, [-1, num_classes])
        # one_hot_labels = tf.one_hot(label_batch, num_classes)
        # ce_pt = -tf.multiply(tf.log(pt + 1e-8), one_hot_labels)
        # modulation = tf.pow(tf.multiply(1. - pt, one_hot_labels), 2.0)
        # loss_map = tf.reduce_max(tf.multiply(ce_pt, modulation), axis=1)

        if gan_type == 'vanilla':
            # Build accuracy
            generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
            fake_prob = tf.reduce_mean(tf.nn.sigmoid(generator_logits))  # should go to 0
            expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
            real_prob = tf.reduce_mean(tf.nn.sigmoid(expert_logits))  # should go to 1
            # Build regression loss
            # let x = logits, z = targets.
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generator_logits, labels=tf.zeros_like(generator_logits))
            generator_loss = tf.reduce_mean(generator_loss)
            expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                                  labels=tf.ones_like(expert_logits))
            expert_loss = tf.reduce_mean(expert_loss)
        elif gan_type == 'lsgan' or gan_type == 'wgan':
            # Build accuracy
            generator_acc = tf.reduce_mean(tf.to_float(generator_logits < 0.5))
            fake_prob = tf.reduce_mean(generator_logits)  # should go to 0
            expert_acc = tf.reduce_mean(tf.to_float(expert_logits > 0.5))
            real_prob = tf.reduce_mean(expert_logits)  # should go to 1
            # Build loss
            if gan_type == 'lsgan':
                generator_loss = tf.reduce_mean((generator_logits - tf.zeros_like(generator_logits)) ** 2)
                expert_loss = tf.reduce_mean((expert_logits - tf.ones_like(generator_logits)) ** 2)
            elif gan_type == 'wgan':
                generator_loss = tf.reduce_mean(tf.abs(generator_logits - tf.zeros_like(generator_logits)))
                expert_loss = tf.reduce_mean(tf.abs(expert_logits - tf.ones_like(generator_logits)))
        else:
            raise ValueError("The specified GAN type is not implemented.")

        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)

        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(self.logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy

        # Loss + Accuracy terms
        self.total_loss = generator_loss + expert_loss + entropy_loss
        self.losses = [generator_loss,
                       expert_loss, entropy, entropy_loss, self.total_loss,
                       generator_acc, expert_acc, fake_prob, real_prob]
        self.loss_name = [
            "generator_loss",
            "expert_loss", "entropy", "entropy_loss", "total_loss",
            "generator_acc", "expert_acc", "fake_prob", "real_prob"]
        if idx:
            self.loss_name = ["discriminator{}/{}".format(idx, v) for v in self.loss_name]

        # self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function(
            [self.generator_data_ph, self.expert_data_ph],
            self.losses + [U.flatgrad(self.total_loss, var_list)])

        # create weight histograms
        with tf.variable_scope(self.scope):
            self.weight_hists = []
            for v in var_list:
                self.weight_hists.append(tf.summary.histogram(v.name, v))

        self.adam = MpiAdam(self.get_trainable_variables(), beta1=0.5)

    def get_weight_summaries(self):
        return tf.summary.merge(self.weight_hists)

    def build_graph(self, input_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("ganfilter"):
                self.input_rms = RunningMeanStd(shape=self.input_shape)
            _input = (input_ph - self.input_rms.mean / self.input_rms.std)

            h = _input
            for i in range(self.num_hidden_layer):
                h = tf.contrib.layers.fully_connected(
                    h, self.hidden_size, activation_fn=self.activation_fn)
            logits = tf.contrib.layers.fully_connected(h, 1, activation_fn=None)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def logsigmoid(self, a):
        '''Equivalent to tf.log(tf.sigmoid(a))'''
        return -tf.nn.softplus(-a)

    def logit_bernoulli_entropy(self, logits):
        ent = (1.-tf.nn.sigmoid(logits))*logits - self.logsigmoid(logits)
        return ent

    def train(self, generator_data, expert_data, optim_batchsize, stepsize):
        optim_batchsize = optim_batchsize or generator_data.shape[0]

        self.input_rms.update(np.concatenate((generator_data, expert_data), 0))

        d_losses = []
        for gen_batch, ex_batch in dataset.iterbatches(
                (generator_data, expert_data),
                include_final_partial_batch=False, batch_size=optim_batchsize):

            *newlosses, g = self.lossandgrad(gen_batch, ex_batch)
            self.adam.update(g, stepsize)
            d_losses.append(newlosses)
        d_losses_mean = np.mean(d_losses, axis=0)
        logger.log(fmt_row(13, d_losses_mean))
        for d in range(0, len(self.loss_name)):
            logger.record_tabular(self.loss_name[d], np.mean(d_losses_mean[d]))

    def get_reward(self, sample):
        sess = U.get_session()
        if len(sample.shape) == 1:
            sample = np.expand_dims(sample, 0)
        feed_dict = {self.generator_data_ph:sample}
        reward = sess.run(self.reward_op, feed_dict)
        return reward
