"""Main training loop"""

from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import task
from task import generate_trials
from network import Model, get_perf
import variance
import clustering
import tools


def get_default_hp(ruleset):
    '''Get a default hpuration.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
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
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 1.,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'param_intsyn': None
            }

    return hp


def update_intsyn():
    # Only if using intelligent synapses
    v_current = sess.run(model.var_list)

    if i_rule_train == 0:
        v_anc0 = v_current
        Omega0 = [
            np.zeros(v.shape, dtype='float32') for v in v_anc0]
    else:
        v_anc0_prev = v_anc0
        v_anc0 = v_current
        v_delta = [
            v - v_prev for v, v_prev in zip(v_anc0, v_anc0_prev)]

        # Make sure all elements in omega0 are non-negative
        # Penalty
        Omega0 = [(O + o * (o > 0.) / (v_d**2 + ksi_intsyn))
                  for O, o, v_d in zip(Omega0, omega0, v_delta)]

        # Update cost
        extra_cost = tf.constant(0.)
        for v, w, v_val in zip(model.var_list, Omega0, v_current):
            extra_cost += c_intsyn*tf.reduce_sum(
                    tf.multiply(w, tf.square(v - v_val)))
        model.set_optimizer(extra_cost=extra_cost)

    # Reset
    omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
    return 


def update_intsyn2():
    # Continual learning
    v_prev = v_current

    _, grads_and_vars_ = sess.run(
            [model.optimizer, model.grads_and_vars],
            feed_dict=feed_dict)
    v_grad, v_current = zip(*grads_and_vars_)

    # Update synaptic importance
    omega0 = [
        o - (v_c - v_p) * v_g for o, v_c, v_p, v_g in
        zip(omega0, v_current, v_prev, v_grad)
        ]


def do_eval(sess, model, log, rule_train):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training '+rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(
                rule_test, hp, 'random', batch_size=batch_size_test_rep)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            c_lsq, c_reg, y_hat_test = sess.run(
                [model.cost_lsq, model.cost_reg, model.y_hat],
                feed_dict=feed_dict)

            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_'+rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_'+rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    # Saving the model
    model.save()
    tools.save_log(log)

    return log


def train_old(train_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='mante',
          reuse=False,
          seed=0,
          ):
    '''Train the network

    Args:
        train_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        reuse: boolean. If True, reload previous checkpoints
        seed: int, random seed to be used

    Returns:
        model is stored at train_dir/model.ckpt
        training configuration is stored at train_dir/hp.json
    '''

    tools.mkdir_p(train_dir)

    # Network parameters
    # Number of units each ring has
    if reuse:
        raise NotImplementedError()  # temporarily disable
        # Build the model from save_name
        model = Model(train_dir)
        hp = model.hp

    else:
        # Random number generator used
        default_hp = get_default_hp(ruleset)
        if hp is not None:
            default_hp.update(hp)
        hp = default_hp
        hp['seed'] = seed
        tools.save_hp(hp, train_dir)
        # rng can not be serialized
        hp['rng'] = np.random.RandomState(seed)

        # Build the model
        model = Model(train_dir, hp=hp)

    # Display hpuration
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Number of training iterations for each rule
    rule_train_iters = []
    for rule_train in hp['rule_trains']:
        if not hasattr(rule_train, '__iter__'):
            tmp = 1
        else:
            tmp = len(rule_train)
        rule_train_iters.append(tmp*max_steps)

    # Using continual learning or not
    if hp['param_intsyn']:
        c_intsyn, ksi_intsyn = hp['param_intsyn']
        print('Using continual learning')

    # Store results
    log = defaultdict(list)

    # Record time
    t_start = time.time()

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        if reuse:
            model.restore()
        else:
            sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            # TODO: Need checking
            anchor_ws = sess.run(model.weight_list)

            # TODO: only change weights
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w-w_val))

            model.set_optimizer()

        # Looping
        step_total = 0
        for i_rule_train, rule_train in enumerate(hp['rule_trains']):
            step = 0

            # At the beginning of new tasks
            if hp['param_intsyn']:
                update_intsyn()

            # Keep training until reach max iterations
            while (step * hp['batch_size_train'] <=
                   rule_train_iters[i_rule_train]):
                try:
                    # Validation
                    if step % display_step == 0:
                        trial = step_total * hp['batch_size_train']
                        log['trials'].append(trial)
                        log['times'].append(time.time()-t_start)
                        log['rule_now'].append(rule_train)
                        log = do_eval(sess, model, log, rule_train)
                        if log['perf_avg'][-1] > model.hp['target_perf']:
                            print('Perf reached the target: {:0.2f}'.format(
                                hp['target_perf']))
                            break

                    # Training
                    if not hasattr(rule_train, '__iter__'):
                        rule_train_now = rule_train
                    else:
                        p = hp['rule_probs'][i_rule_train]
                        rule_train_now = hp['rng'].choice(rule_train, p=p)
                    # Generate a random batch of trials.
                    # Each batch has the same trial length
                    trial = generate_trials(
                            rule_train_now, hp, 'random',
                            batch_size=hp['batch_size_train'])

                    # Generating feed_dict.
                    feed_dict = tools.gen_feed_dict(model, trial, hp)

                    if hp['param_intsyn']:
                        update_intsyn2()
                    else:
                        sess.run(model.train_step, feed_dict=feed_dict)

                    step += 1
                    step_total += 1

                except KeyboardInterrupt:
                    print("Optimization interrupted by user")
                    break

        print("Optimization Finished!")


def display_rich_output(model, sess, step, log, train_dir):
    """Display step by step outputs during training."""
    variance._compute_variance_bymodel(model, sess)
    rule_pair = ['contextdm1', 'contextdm2']
    save_name = '_atstep' + str(step)
    title = ('Step ' + str(step) +
             ' Perf. {:0.2f}'.format(log['perf_avg'][-1]))
    variance.plot_hist_varprop(train_dir, rule_pair,
                               figname_extra=save_name,
                               title=title)
    plt.close('all')


def train(train_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          rich_output=False,
          ):
    """Train the network

    Args:
        train_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at train_dir/model.ckpt
        training configuration is stored at train_dir/hp.json
    """

    tools.mkdir_p(train_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))
    tools.save_hp(hp, train_dir)

    # Build the model
    model = Model(train_dir, hp=hp)

    # Display hpuration
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['train_dir'] = train_dir 
    
    # Record time
    t_start = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w - w_val))

            model.set_optimizer()

        # partial weight training
        if ('p_weight_train' in hp and
            (hp['p_weight_train'] is not None) and
            hp['p_weight_train'] < 1.0):
            for w in model.weight_list:
                w_val = sess.run(w)
                w_size = sess.run(tf.size(w))
                w_mask_tmp = np.linspace(0, 1, w_size)
                hp['rng'].shuffle(w_mask_tmp)
                ind_fix = w_mask_tmp > hp['p_weight_train']
                w_mask = np.zeros(w_size, dtype=np.float32)
                w_mask[ind_fix] = 1e-1  # will be squared in l2_loss
                w_mask = tf.constant(w_mask)
                w_mask = tf.reshape(w_mask, w.shape)
                model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
            model.set_optimizer()

        step = 0
        while step * hp['batch_size_train'] <= max_steps:
            try:
                # Validation
                if step % display_step == 0:
                    log['trials'].append(step * hp['batch_size_train'])
                    log['times'].append(time.time()-t_start)
                    log = do_eval(sess, model, log, hp['rule_trains'])
                    #if log['perf_avg'][-1] > model.hp['target_perf']:
                    #check if minimum performance is above target    
                    if log['perf_min'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        break

                    if rich_output:
                        display_rich_output(model, sess, step, log, train_dir)

                # Training
                rule_train_now = hp['rng'].choice(hp['rule_trains'],
                                                       p=hp['rule_probs'])
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                        rule_train_now, hp, 'random',
                        batch_size=hp['batch_size_train'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)
                sess.run(model.train_step, feed_dict=feed_dict)

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")


if __name__ == '__main__':
    pass
    run_analysis = []
    hp = {'rnn_type': 'LeakyRNN',
               'n_rnn': 128,
               'activation': 'softplus',
               'l1_h': 0,
               'l2_h': 0,
               'l1_weight': 0,
               'l2_weight': 0,
               'l2_weight_init': 0,
               'p_weight_train': 0.5,
               'target_perf': 0.9,
               'w_rec_init': 'randortho'}
    train('data/mantetemp', seed=1, hp=hp, ruleset='mante',
          display_step=500, rich_output=False)
    
    # rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    # hp = {'rnn_type': 'LeakyGRU', 'n_rnn': 128}
    # train('debug',hp=hp, ruleset='all',rule_prob_map=rule_prob_map,seed=1)
