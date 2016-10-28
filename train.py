"""
2016/06/03 Restart, with Blocks

Main training loop and network structure
"""

from __future__ import division

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from task import *
from network import LeakyRNNCell, get_perf

tf.reset_default_graph()

HDIM = 300
N_RING = 16

config = {'h_type'      : 'leaky_rec',
          'alpha'       : 0.2, # \Delta t/tau
          'dt'          : 0.2*TAU,
          'HDIM'        : HDIM,
          'N_RING'      : N_RING,
          'shape'       : (1+2*N_RING+N_RULE, HDIM, N_RING+1),
          'save_addon'  : 'tf_debug_'+str(HDIM)}

# Rules
rules = [FIXATION, GO, INHGO, DELAYGO,\
CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,\
REMAP, INHREMAP, DELAYREMAP,\
DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]
rule_weights = np.ones(len(rules))
rule_weights[[rules.index(r) for r in [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]]] = 10
rule_weights[[rules.index(r) for r in [DELAYMATCHNOGO, DELAYMATCHGO, DMCNOGO, DMCGO]]] = 5


# Parameters
learning_rate = 0.0001
training_iters = 1500000
batch_size_train = 10
batch_size_test = 200
display_step = 1000

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
h, states = rnn.dynamic_rnn(LeakyRNNCell(n_hidden), x, dtype=tf.float32)

# Output
y_hat = tf.sigmoid(tf.matmul(tf.reshape(h, (-1, n_hidden)), w_out) + b_out)

# Loss
cost = tf.reduce_mean((y-y_hat)**2*c_mask)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Store results
trials     = []
times      = []
cost_tests = {rule:[] for rule in rules}
perf_tests = {rule:[] for rule in rules}

# Launch the graph
t_start = time.time()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size_train < training_iters:
        try:
            # Training
            rule = np.random.choice(rules, p=rule_weights/rule_weights.sum())
            task = generate_onebatch(rule, config, 'random', batch_size=batch_size_train)
            sess.run(optimizer, feed_dict={x: task.x,
                                           y: task.y.reshape((-1,n_output)),
                                           c_mask: task.c_mask.reshape((-1,n_output))})

            # Validation
            if step % display_step == 0:
                trials.append(step*batch_size_train)
                times.append(time.time()-t_start)
                print('Trial {:7d}'.format(trials[-1]) +
                      '  | Time {:0.2f} s'.format(times[-1]))
                # Calculate batch accuracy
                c_tests = list()
                for rule in rules:
                    task = generate_onebatch(rule, config, 'random', batch_size=batch_size_test)
                    y_hat_test = sess.run(y_hat, feed_dict={x: task.x})
                    y_hat_test = y_hat_test.reshape((-1,batch_size_test,n_output))
                    c_test = np.mean(task.c_mask*(y_hat_test-task.y)**2)
                    perf_test = get_perf(y_hat_test, task.y_loc)
                    cost_tests[rule].append(c_test)
                    perf_tests[rule].append(perf_test)
                    print('{:15s}'.format(rule_name[rule]) +
                          '| cost {:0.2f}'.format(c_test)  +
                          '| perf {:0.2f}'.format(perf_test))
            step += 1
        except KeyboardInterrupt:
            break

    # Saving the model
    save_path = saver.save(sess, os.path.join('data', config['save_addon']+'.ckpt'))
    print("Model saved in file: %s" % save_path)

    config['trials']     = trials
    config['times']      = times
    config['cost_tests'] = cost_tests
    config['perf_tests'] = perf_tests
    with open('data/config'+config['save_addon']+'.pkl', 'wb') as f:
        pickle.dump(config, f)

    print("Optimization Finished!")
