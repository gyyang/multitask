"""
2016/06/03 @ Guangyu Robert Yang

Main training loop and network structure
"""

from __future__ import division

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import errno

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from task import *
from network import LeakyRNNCell, get_perf

def mkdir_p(path):
    """
    Portable mkdir -p
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def train(HDIM=300, s=1, learning_rate=0.001, training_iters=5000000, save_addon=None):
    '''
    Training the network
    :param HDIM: Number of recurrent units
    :param s: Type of training tasks
    :param learning_rate: default 0.001. This is a good default value.
    :param training_iters:
    :param save_addon:
    :return:
    '''

    # Number of input rings
    num_ring = 2

    if s == 0:
        save_addon_type = 'allrule_nonoise'
    elif s == 1:
        save_addon_type = 'allrule_weaknoise'
    elif s == 2:
        save_addon_type = 'allrule_strongnoise'
    elif s == 3:
        save_addon_type = 'attendonly_nonoise'
    elif s == 4:
        save_addon_type = 'attendonly_weaknoise'
    elif s == 5:
        save_addon_type = 'attendonly_strongnoise'
    elif s == 6:
        save_addon_type = 'choiceonly_nonoise'
    elif s == 7:
        save_addon_type = 'choiceonly_weaknoise'
    elif s == 8:
        save_addon_type = 'choiceonly_strongnoise'
    elif s == 9:
        save_addon_type = 'delaychoiceonly_nonoise'
    elif s == 10:
        save_addon_type = 'delaychoiceonly_weaknoise'
    elif s == 11:
        save_addon_type = 'delaychoiceonly_strongnoise'
    elif s == 12:
        save_addon_type = 'oiconly_weaknoise'

    tf.reset_default_graph()

    N_RING = 16

    if 'nonoise' in save_addon_type:
        sigma_rec = 0
    elif 'weaknoise' in save_addon_type:
        sigma_rec = 0.05
    elif 'strongnoise' in save_addon_type:
        sigma_rec = 0.15

    if 'allrule' in save_addon_type:
        # Rules
        rules = [GO, INHGO, DELAYGO,\
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
        REMAP, INHREMAP, DELAYREMAP,\
        DMSGO, DMSNOGO, DMCGO, DMCNOGO]
    elif 'attendonly' in save_addon_type:
        rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    elif 'delaychoiceonly' in save_addon_type: # This has to be before choiceonly
        rules = [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2]
    elif 'choiceonly' in save_addon_type:
        rules = [CHOICE_MOD1, CHOICE_MOD2]
    elif 'oiconly' in save_addon_type:
        rules = [OIC]

    if OIC in rules:
        num_ring = 3


    rule_weights = np.ones(len(rules))

    if 'allrule' in save_addon_type:
        # Make them 5 times more common
        rule_weights[rules.index(CHOICEATTEND_MOD1)] = 5
        rule_weights[rules.index(CHOICEATTEND_MOD2)] = 5

    if save_addon is None:
        save_addon = save_addon_type
    else:
        save_addon = save_addon_type + '_' + save_addon

    # Parameters
    batch_size_train = 50
    batch_size_test = 2000
    display_step = 1000

    # learning_rate = 0.001
    # training_iters = 500000
    # batch_size_train = 100
    # batch_size_test = 200
    # display_step = 200

    config = {'h_type'      : 'leaky_rec',
              'alpha'       : 0.2, # \Delta t/tau
              'dt'          : 0.2*TAU,
              'sigma_rec'   : sigma_rec,
              'HDIM'        : HDIM,
              'N_RING'      : N_RING,
              'num_ring'    : num_ring,
              'shape'       : (1+num_ring*N_RING+N_RULE, HDIM, N_RING+1),
              'save_addon'  : save_addon,
              'rules'       : rules,
              'rule_weights': rule_weights,
              'learning_rate': learning_rate,
              'training_iters' : training_iters,
              'batch_size_train' : batch_size_train,
              'batch_size_test' : batch_size_test}

    for key, val in config.iteritems():
        print('{:20s} = '.format(key) + str(val))

    # Network Parameters
    nx, nh, ny = config['shape']

    # tf Graph input
    x = tf.placeholder("float", [None, None, nx]) # (time, batch, nx)
    y = tf.placeholder("float", [None, ny])
    c_mask = tf.placeholder("float", [None, ny])

    # Define weights
    wy = tf.Variable(tf.random_normal([nh, ny], stddev=0.4/np.sqrt(nh)))
    by = tf.Variable(tf.zeros([ny]))

    # Initial state (requires tensorflow later than 0.10)
    h_init = tf.Variable(0.3*tf.ones([1, nh]))
    h_init_bc = tf.tile(h_init, [tf.shape(x)[1], 1]) # broadcast to size (batch, n_h)

    # Recurrent activity
    cell = LeakyRNNCell(nh, config['alpha'], sigma_rec=config['sigma_rec'])
    h, states = rnn.dynamic_rnn(cell, x, initial_state=tf.abs(h_init_bc),
                                dtype=tf.float32, time_major=True) # time_major is important

    # Output
    y_hat = tf.sigmoid(tf.matmul(tf.reshape(h, (-1, nh)), wy) + by)

    # Loss
    cost = tf.reduce_mean(tf.square((y-y_hat)*c_mask))
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
                                               y: task.y.reshape((-1,ny)),
                                               c_mask: task.c_mask.reshape((-1,ny))})

                # Validation
                if step % display_step == 0:
                    trials.append(step*batch_size_train)
                    times.append(time.time()-t_start)
                    print('Trial {:7d}'.format(trials[-1]) +
                          '  | Time {:0.2f} s'.format(times[-1]))
                    for rule in rules:
                        n_rep = 20
                        batch_size_test_rep = int(batch_size_test/n_rep)
                        c_rep = list()
                        perf_rep = list()
                        for i_rep in range(n_rep):
                            task = generate_onebatch(rule, config, 'random', batch_size=batch_size_test_rep)
                            y_hat_test = sess.run(y_hat, feed_dict={x: task.x})
                            y_hat_test = y_hat_test.reshape((-1,batch_size_test_rep,ny))

                            # Cost is first summed over time, and averaged across batch and units
                            # We did the averaging over time through c_mask
                            c_test = np.mean(np.sum(((y_hat_test-task.y)*task.c_mask)**2, axis=0))
                            perf_test = np.mean(get_perf(y_hat_test, task.y_loc))
                            c_rep.append(c_test)
                            perf_rep.append(perf_test)

                        cost_tests[rule].append(np.mean(c_rep))
                        perf_tests[rule].append(np.mean(perf_rep))
                        print('{:15s}'.format(rule_name[rule]) +
                              '| cost {:0.5f}'.format(cost_tests[rule][-1])  +
                              '  | perf {:0.2f}'.format(perf_tests[rule][-1]))
                step += 1
            except KeyboardInterrupt:
                break

        mkdir_p('data')

        # Saving the model
        save_path = saver.save(sess, os.path.join('data', config['save_addon']+'.ckpt'))
        print("Model saved in file: %s" % save_path)

        config['trials']     = trials
        config['times']      = times
        config['cost_tests'] = cost_tests
        config['perf_tests'] = perf_tests
        with open(os.path.join('data', 'config'+config['save_addon']+'.pkl'), 'wb') as f:
            pickle.dump(config, f)

        print("Optimization Finished!")

    if 'allrule' in save_addon_type:
        from variance import compute_variance
        compute_variance(config['save_addon'], 'rule', rules)
        print('Computed variance')

        from performance import compute_choicefamily_varytime
        for rule in [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]:
            compute_choicefamily_varytime(save_addon, rule)

if __name__ == '__main__':
    pass
    train(HDIM=20, s=12, save_addon='test', training_iters=200000)