"""
author:lancer
"""
import tensorflow as tf
import numpy as np

# convolution layer
def conv2d(x, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        # w, h, in_channel, out_channel
        w = tf.get_variable("w", [5, 5, inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputFeatures], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(x, w, strides=[1,2,2,1], padding="SAME") + b
        return conv

def conv2d_transpose(x, outputShape, name):
    with tf.variable_scope(name):
        # w, h, out_channel, in_channel
        w = tf.get_variable("w",[5, 5, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputShape[-1]], initializer=tf.constant_initializer(0.0))

        conv_transpose = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,2,2,1]) + b

        return conv_transpose

def deconv2d(input_, outputShape, k_h, k_w, s_h, s_w, stddev=0.02, name="deconv2d"):

    with tf.variable_scope(name):
        # w, h, out_channel, in_channel
        w = tf.get_variable("w",[k_h, k_w, outputShape[-1], input_.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable("b", [outputShape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=outputShape, strides=[1, s_h, s_w, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        return deconv

# Full connection
def dense(x, inputFeatures, outputFeatures, name, with_w=False):

    with tf.variable_scope(name or "Linear"):
        w = tf.get_variable("w", [inputFeatures, outputFeatures], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputFeatures], initializer=tf.constant_initializer(0.0))

        if with_w:
            return tf.matmul(x,w) + b, w, b
        return tf.matmul(x, w) + b

# Leaky relu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5*(1+leak)
        f2 = 0.5*(1-leak)
        return f1*x + f2*abs(x)





