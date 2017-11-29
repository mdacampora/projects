# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:02:19 2017

@author: Mike James
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
#sess = tf.InteractiveSession()
#
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.placeholder(tf.float32, shape=[None, 10])

y = tf.matmul(x,W) + b


# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1,], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

#First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# then convolve x_image with the weight tensor, add the bias, apply the ReLU
# function, and finally max pool. The max_pool_2x2 method will reduce the image
#size to 14x14

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer
#the second layer will have 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
#now that the image size has been reduced to 7x7, we add a fully-connected
#layer with 1024 neurons to allow processing on the entire image. We reshape 
#the tensor from the pooling layer into a batch of vectors, multiply by a 
#weight matrix, add a bias, and apply a ReLU.
W_fc1 = weight_variable([7* 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout, to reduce overfitting
# create a placeholder for the probability taht a neurons output is kept during
# dropout, which allows for turning dropout on during training, and off during
# testing. tff.nn.dropout auto-handles scaling neuron outputs
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
# add a layer just like for the one layer softmax regresssion above
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and Evaluate the Model
# diffrences - 1. we replace the steepest Gradient Descent Optimizer with ADAM
# optimizer. 2. Include additional paramter keep_prob in feed_dict to control
# dropout rate. 3. add loggin to every 100th iteration in the training process

# Use tf.Session instead of tf.InteractiveSession to better 
# separate the process of creating the graph(model specification) and the 
# process of evaluating the graph (model fitting). It generally makes for 
# cleaner code. The tf.Session is created withina  with block so that it is
# automatically destroyed once the block is exited.

cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))









