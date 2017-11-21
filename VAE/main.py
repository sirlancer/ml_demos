"""
author:lancer
"""
import input_data
import tensorflow as tf
from layers import *
from utils import *
import os
from scipy.misc import imsave


class VAE(object):
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.n_examples = self.mnist.train._num_examples

        self.n_z = 20
        self.batch_size = 100

        self.images = tf.placeholder(tf.float32, [None, 784])
        image_matrix = tf.reshape(self.images, [-1, 28, 28, 1])
        z_mean, z_stddev = self.recognition(image_matrix)

        sampels = tf.random_normal([self.batch_size, self.n_z], 0, 1, dtype=tf.float32)

        guessed_z = z_mean + sampels*z_stddev

        self.generated_images = self.generation(guessed_z)

        generated_flat = tf.reshape(self.generated_images, [self.batch_size, 28*28])

        self.generation_loss = -tf.reduce_sum(self.images*tf.log(10e-8+generated_flat)+(1-self.images)*tf.log(10e-8 + 1 - self.images),1)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean)+tf.square(z_stddev) - 1 + tf.log(tf.square(z_stddev)),1)

        self.cost = tf.reduce_mean(self.generation_loss+self.latent_loss)

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)


    # encoder
    def recognition(self, input_images, name="recognition"):
        with tf.variable_scope(name):
            h1 = lrelu(conv2d(input_images, 1, 16, "d_h1"))
            h2 = lrelu(conv2d(h1, 16, 32, "d_h2"))
            h2_flat = tf.reshape(h2, [self.batch_size, 7*7*32])

            w_mean = dense(h2_flat, 7*7*32, self.n_z, "w_mean")
            w_stddev = dense(h2_flat, 7*7*32, self.n_z, "w_stddev")

            return w_mean, w_stddev

    # decoder
    def generation(self, z, name="generation"):
        with tf.variable_scope(name):
            z_develop = dense(z, self.n_z, 7*7*32, "z_matrix")
            z_matrix = tf.reshape(z_develop, [self.batch_size, 7, 7, 32])

            h1 = tf.nn.relu(conv2d_transpose(z_matrix, [self.batch_size, 14, 14, 16], "g_h1"))
            h2 = conv2d_transpose(h1, [self.batch_size, 28, 28, 1], "g_h2")
            h2 = tf.nn.sigmoid(h2)

            return h2

    def train(self):
        visulization = self.mnist.train.next_batch(self.batch_size)[0]
        reshaped_vis = tf.reshape(visulization, [self.batch_size,28, 28])
        result_dir = "./results"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        imsave("./result/base.jpg", merge(reshaped_vis[:64], [8,8]))

        checkpoint_dir = "./model"
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_examples / self.batch_size)):
                    batch = self.mnist.train.next_batch(self.batch_size)[0]
                    _, gen_loss, lat_loss = sess.run([self.optimizer, self.generation_loss, self.latent_loss], feed_dict={self.images:batch})
                    if idx % 500 == 0:
                        print("epoch %d, gen_loss:%f, lat_loss:%f" %(epoch, gen_loss, lat_loss))

                        saver.save(sess, checkpoint_dir, global_step=epoch)
                        generated_test = sess.run(self.generated_images, feed_dict={self.image:visulization})
                        generated_test = generated_test.reshape(self.batch_size,28,28)
                        imsave(result_dir+'/'+str(epoch)+'-'+str(idx)+'.jpg', merge(generated_test[:64],[8,8]))



vae = VAE()
vae.train()