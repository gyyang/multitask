"""Tests for network."""

from __future__ import division

import unittest
import numpy as np
import tensorflow as tf
import network


class NetworkTest(unittest.TestCase):

    def testTFPopVec(self):
        n_units = 36
        batch_size = n_units
        ys = [
            np.random.rand(batch_size, n_units),
            np.eye(batch_size, n_units)
        ]

        for y in ys:
            theta = network.popvec(y)

            y2 = tf.placeholder('float', [batch_size, n_units])
            theta2 = network.tf_popvec(y2)
            with tf.Session() as sess:
                theta2_val = sess.run(theta2, feed_dict={y2: y})
            self.assertTrue(np.allclose(theta, theta2_val))

    def testLesion(self):
        num_ring = 2
        n_rule = 20
        n_eachring = 32
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
        hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 256,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # Type of loss functions
            'loss_type': 'lsq',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'diag',
            # a default weak regularization prevents instability
            'l1_h': 1.0 * 0.0001,
            # l2 regularization on activity
            'l2_h': 1.0 * 0,
            # l2 regularization on weight
            'l1_weight': 0.0001 * 0,
            # l2 regularization on weight
            'l2_weight': 0.0001 * 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0.0001 * 0,
            # Stopping performance
            'target_perf': 1.,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1 + num_ring * n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': 'mante',
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'param_intsyn': None,
            # random seed
            'seed': 0
        }
        hp['rnn_type'] = 'LeakyRNN'
        # Build the model
        model = network.Model(None, hp=hp)

        # Use customized session that launches the graph as well
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            model.lesion_units(sess, [hp['n_rnn']-1])

            for v in model.var_list:
                if 'kernel' in v.name or 'weight' in v.name:
                    v_val = sess.run(v)

                self.assertEqual(sum(abs(v_val[-1, :])), 0)


if __name__ == '__main__':
    unittest.main()
