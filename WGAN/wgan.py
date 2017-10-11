import os
import time
import importlib
import matplotlib
matplotlib.use('Agg')
import argparse
import tensorflow as tf
from scipy.misc import imsave
import matplotlib.pyplot as plt

import sys
sys.path.append(os.getcwd())

import visualize
class WassersteinGAN(object):

    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = -tf.reduce_mean(self.d_)
        self.d_loss = -tf.reduce_mean(self.d) + tf.reduce_mean(self.d_)

        self.d_adam, self.g_adam = None, None

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=self.g_net.vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, batch_size=64, num_batches=1000000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 5
            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_adam, feed_dict={self.z: bz})

            if t % 5000 == 0:
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(self.d_loss, feed_dict={self.x: bx, self.z:bz})
                g_loss = self.sess.run(self.g_loss, feed_dict={self.z: bz})

                print('Iter [%8d] Time[%5.4f] d_loss [%.4f] g_loss [%.4f]' % (t, time.time()-start_time, d_loss, g_loss))

            if t % 5000 == 0:
                bz = self.z_sampler(batch_size, self.z_dim)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                bx = visualize.grid_transform(bx, xs.shape)
                imsave('logs/{}/{}.png'.format(self.data, t/5000), bx)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='1')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gups

    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data+'.'+args.model)

    xs = data.DataSampler()
    zs = data.NoiseSampler()

    d_net = model.Discriminator()
    g_net = model.Generator()

    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model)
    wgan.train()
