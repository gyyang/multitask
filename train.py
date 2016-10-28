"""
2016/06/03 Restart, with Blocks

Main training loop and network structure
"""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from task import *
from network import LeakyRNNCell

tf.reset_default_graph()
HDIM = 50
N_RING = 16


config = {'h_type'      : 'leaky_rec_ei',
          'alpha'       : 0.2, # \Delta t/tau
          'dt'          : 0.2*TAU,
          'HDIM'        : HDIM,
          'N_RING'      : N_RING,
          'shape'       : (1+2*N_RING+N_RULE, HDIM, N_RING+1),
          'save_addon'  : 'tf_'+str(HDIM)}


# Parameters
learning_rate = 0.001
training_iters = 50000
batch_size = 10
display_step = 100

# Network Parameters
n_input, n_hidden, n_output = config['shape']

# tf Graph input
x = tf.placeholder("float", [None, None, n_input]) # time * batch * n_input
y = tf.placeholder("float", [None, n_output])
c_mask = tf.placeholder("float", [None, n_output])

# Define weights
w_out = tf.Variable(tf.random_normal([n_hidden, n_output]))
b_out = tf.Variable(tf.random_normal([n_output]))

# Recurrent activity
rnn_cell = LeakyRNNCell(n_hidden)
h, states = rnn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

# Output
y_hat = tf.sigmoid(tf.matmul(tf.reshape(h, (-1, n_hidden)), w_out) + b_out)

# Loss
cost = tf.reduce_mean((y-y_hat)**2*c_mask)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        task = generate_onebatch(GO, config, 'random', batch_size=batch_size)
        sess.run(optimizer, feed_dict={x: task.x,
                                       y: task.y.reshape((-1,n_output)),
                                       c_mask: task.c_mask.reshape((-1,n_output))})
        if step % display_step == 0:
            # Calculate batch accuracy
            c_sample = sess.run(cost, feed_dict={x: task.x,
                                       y: task.y.reshape((-1,n_output)),
                                       c_mask: task.c_mask.reshape((-1,n_output))})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(c_sample))
        step += 1
    print("Optimization Finished!")
