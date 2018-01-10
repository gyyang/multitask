"""Main training loop"""

from __future__ import division

import os
import sys
import time
from collections import defaultdict
import numpy as np
import tensorflow as tf

import task
from task import generate_trials
from network import Model, get_perf
import tools


def get_default_hparams(ruleset):
    '''Get a default hparamsuration.

    Useful for debugging.

    Returns:
        hparams : a dictionary containing training hparamsuration
    '''
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hparams = {
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
            'l1_h': 1.0*0.0001,
            # l2 regularization on activity
            'l2_h': 1.0*0,
            # l2 regularization on weight
            'l1_weight': 0.0001*0,
            # l2 regularization on weight
            'l2_weight': 0.0001*0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0.0001*0,
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
            'n_rnn': 64,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.01,
            # intelligent synapses parameters, tuple (c, ksi)
            'param_intsyn': None
            }

    return hparams


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


def gen_feed_dict(model, trial, hparams):
    if hparams['in_type'] == 'normal':
        feed_dict = {model.x: trial.x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}
    elif hparams['in_type'] == 'multi':
        n_time, batch_size = trial.x.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hparams['rule_start']*hparams['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hparams['rule_start']:])
            i_start = ind_rule*hparams['rule_start']
            x[:, i, i_start:i_start+hparams['rule_start']] = \
                trial.x[:, i, :hparams['rule_start']]

        feed_dict = {model.x: x,
                     model.y: trial.y,
                     model.c_mask: trial.c_mask}

    return feed_dict



def do_eval(sess, model, log, rule_train):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hparams = model.hparams
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training '+rule_name_print)

    for rule_test in hparams['rules']:
        n_rep = 16
        batch_size_test_rep = int(hparams['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(
                rule_test, hparams, 'random', batch_size=batch_size_test_rep)
            feed_dict = gen_feed_dict(model, trial, hparams)
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

        log['cost_'+rule_test].append(np.mean(clsq_tmp))
        log['creg_'+rule_test].append(np.mean(creg_tmp))
        log['perf_'+rule_test].append(np.mean(perf_tmp))
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

    # Saving the model
    model.save()
    tools.save_log(log, hparams['save_name'])

    return log


def train_old(train_dir,
          hparams=None,
          max_steps=1000000,
          display_step=500,
          ruleset='mante',
          reuse=False,
          seed=0,
          ):
    '''Train the network

    Args:
        train_dir: str, training directory
        hparams: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        reuse: boolean. If True, reload previous checkpoints
        seed: int, random seed to be used

    Returns:
        model is stored at train_dir/model.ckpt
        training configuration is stored at train_dir/hparams.json
    '''

    tools.mkdir_p(train_dir)

    # Network parameters
    # Number of units each ring has
    if reuse:
        raise NotImplementedError()  # temporarily disable
        # Build the model from save_name
        model = Model(train_dir)
        hparams = model.hparams

    else:
        # Random number generator used
        default_hparams = get_default_hparams(ruleset)
        if hparams is not None:
            default_hparams.update(hparams)
        hparams = default_hparams
        hparams['seed'] = seed
        tools.save_hparams(hparams, train_dir)
        # rng can not be serialized
        hparams['rng'] = np.random.RandomState(seed)

        # Build the model
        model = Model(train_dir, hparams=hparams)

    # Display hparamsuration
    for key, val in hparams.iteritems():
        print('{:20s} = '.format(key) + str(val))

    # Number of training iterations for each rule
    rule_train_iters = []
    for rule_train in hparams['rule_trains']:
        if not hasattr(rule_train, '__iter__'):
            tmp = 1
        else:
            tmp = len(rule_train)
        rule_train_iters.append(tmp*max_steps)

    # Using continual learning or not
    if hparams['param_intsyn']:
        c_intsyn, ksi_intsyn = hparams['param_intsyn']
        print('Using continual learning')

    # Store results
    log = defaultdict(list)

    # Record time
    t_start = time.time()

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        if reuse:
            model.restore(sess)
        else:
            sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hparams['l2_weight_init'] > 0:
            # TODO: Need checking
            anchor_vars = sess.run(model.var_list)

            for v, v_val in zip(model.var_list, anchor_vars):
                model.cost_reg += (hparams['l2_weight_init'] *
                                   tf.reduce_sum(tf.square(v-v_val)))

            model.set_optimizer()

        # Looping
        step_total = 0
        for i_rule_train, rule_train in enumerate(hparams['rule_trains']):
            step = 0

            # At the beginning of new tasks
            if hparams['param_intsyn']:
                update_intsyn()

            # Keep training until reach max iterations
            while (step * hparams['batch_size_train'] <=
                   rule_train_iters[i_rule_train]):
                try:
                    # Validation
                    if step % display_step == 0:
                        trial = step_total * hparams['batch_size_train']
                        log['trials'].append(trial)
                        log['times'].append(time.time()-t_start)
                        log['rule_now'].append(rule_train)
                        log = do_eval(sess, model, log, rule_train)
                        if log['perf_avg'][-1] > model.hparams['target_perf']:
                            print('Perf reached the target: {:0.2f}'.format(
                                hparams['target_perf']))
                            break

                    # Training
                    if not hasattr(rule_train, '__iter__'):
                        rule_train_now = rule_train
                    else:
                        p = hparams['rule_probs'][i_rule_train]
                        rule_train_now = hparams['rng'].choice(rule_train, p=p)
                    # Generate a random batch of trials.
                    # Each batch has the same trial length
                    trial = generate_trials(
                            rule_train_now, hparams, 'random',
                            batch_size=hparams['batch_size_train'])

                    # Generating feed_dict.
                    feed_dict = gen_feed_dict(model, trial, hparams)

                    if hparams['param_intsyn']:
                        update_intsyn2()
                    else:
                        sess.run(model.train_step, feed_dict=feed_dict)

                    step += 1
                    step_total += 1

                except KeyboardInterrupt:
                    print("Optimization interrupted by user")
                    break

        print("Optimization Finished!")


def train(train_dir,
          hparams=None,
          max_steps=1000000,
          display_step=500,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          reuse=False,
          seed=0,
          ):
    '''Train the network

    Args:
        train_dir: str, training directory
        hparams: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        reuse: boolean. If True, reload previous checkpoints
        seed: int, random seed to be used

    Returns:
        model is stored at train_dir/model.ckpt
        training configuration is stored at train_dir/hparams.json
    '''

    tools.mkdir_p(train_dir)

    # Network parameters
    # Number of units each ring has
    if reuse:
        raise NotImplementedError()  # temporarily disable
        # Build the model from save_name
        model = Model(train_dir)
        hparams = model.hparams

    else:
        # Random number generator used
        default_hparams = get_default_hparams(ruleset)
        if hparams is not None:
            default_hparams.update(hparams)
        hparams = default_hparams
        hparams['seed'] = seed
        tools.save_hparams(hparams, train_dir)
        # rng can not be serialized
        hparams['rng'] = np.random.RandomState(seed)

        # Build the model
        model = Model(train_dir, hparams=hparams)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hparams['rule_trains'] = task.rules_dict[ruleset]
    else:
        hparams['rule_trains'] = rule_trains
    hparams['rules'] = hparams['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hparams['rule_probs'] = None
    if hasattr(hparams['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hparams['rule_trains']])
        hparams['rule_probs'] = list(rule_prob/np.sum(rule_prob))

    # Display hparamsuration
    for key, val in hparams.iteritems():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)

    # Record time
    t_start = time.time()

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        if reuse:
            model.restore(sess)
        else:
            sess.run(tf.global_variables_initializer())

        step = 0
        while step * hparams['batch_size_train'] <= max_steps:
            try:
                # Validation
                if step % display_step == 0:
                    log['trials'].append(step * hparams['batch_size_train'])
                    log['times'].append(time.time()-t_start)
                    log = do_eval(sess, model, log, hparams['rule_trains'])
                    if log['perf_avg'][-1] > model.hparams['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hparams['target_perf']))
                        break

                # Training
                rule_train_now = hparams['rng'].choice(hparams['rule_trains'],
                                                       p=hparams['rule_probs'])
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                        rule_train_now, hparams, 'random',
                        batch_size=hparams['batch_size_train'])

                # Generating feed_dict.
                feed_dict = gen_feed_dict(model, trial, hparams)
                sess.run(model.train_step, feed_dict=feed_dict)

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization Finished!")

# function to create save_name TODO(gryang): move to tools?
def to_savename(
        n_hidden=64,
        seed=None,
        activation='tanh',
        rnn_type='LeakyGRU',
        w_rec_init='diag',
        l1_h=1.0*0.0001,
        l2_h=1.0*0,
        l1_weight=1.0*0.0001,
        l2_weight=0.0001*0):
    # add either l1 or l2 reg independently to both activation and weights.
    # if l1_h + l2_h + l1_weight + l2_weight> 0: #if adding regularization.
    # TODO(gryang): clean this up
    if (max(l1_weight, l2_weight) > 0 and max(l1_h, l2_h) > 0):
        save_name = (
            'hidden_' + str(n_hidden) + '_seed_' + str(seed) +
            '_' + activation + '_' +
            rnn_type + '_' + w_rec_init + '_' +
            '_regwt_L' + str(int(1 + np.argmax([l1_weight, l2_weight]))) +
            '_1e_min_' + str(int(-np.log10(max(l1_weight, l2_weight)))) +
            '_regact_L' + str(int(1+np.argmax([l1_h, l2_h]))) +
            '_1e_min_' + str(int(-np.log10(max(l1_h, l2_h)))))

    elif (max(l1_weight, l2_weight) == 0 and max(l1_h, l2_h) > 0):
        save_name = (
            'hidden_' + str(n_hidden) + '_seed_' + str(seed) +
            '_' + activation + '_' + rnn_type + '_' + w_rec_init + '_' +
            '_regwt_None' + '_regact_L' +
            str(int(1 + np.argmax([l1_h, l2_h]))) +
            '_1e_min_' + str(int(-np.log10(max(l1_h, l2_h)))))

    elif (max(l1_weight, l2_weight) > 0 and max(l1_h, l2_h) == 0):
        save_name = (
                'hidden_' + str(n_hidden) + '_seed_' + str(seed) + '_' +
                activation + '_' + rnn_type + '_' + w_rec_init + '_' +
                '_regwt_L' + str(int(1+np.argmax([l1_weight, l2_weight]))) +
                '_1e_min_' + str(int(-np.log10(max(l1_weight, l2_weight)))) +
                '_regact_None')

    else:
        save_name = (
                'hidden' + str(n_hidden) + '_seed_' + str(seed) + '_' +
                activation + '_' + rnn_type + '_' + w_rec_init + '_' +
                '_regu_None')
    return save_name


if __name__ == '__main__':
    pass
    run_analysis = []
    train('debug', seed=1, hparams={'learning_rate': 0.001}, ruleset='mante',
          display_step=500)

#==============================================================================
#     #maddy added - start
#     n_hidden=64
#     seed = 2
#     activation='tanh'
#     rnn_type='LeakyGRU'
#     training_iters   = 50000#150000
#     w_rec_init = 'diag' #maddy added below 5 lines.  
#     l1_h        = 1.0*0.0001
#     l2_h        = 1.0*0
#     l1_weight   = 1.0*0.0001
#     l2_weight   = 0.0001*0
#     # continual learning not factored in. neither are batch_train/test size.    
#     
#     save_name = to_savename(n_hidden = n_hidden,  seed = seed,
#                                activation = activation,
#           rnn_type = rnn_type, w_rec_init    = w_rec_init,  l1_h = l1_h,
#           l2_h = l2_h, l1_weight   = l1_weight, l2_weight   = l2_weight)
#     
#     train(save_name = save_name, n_hidden=n_hidden, seed=seed,
#           activation=activation, rnn_type=rnn_type, 
#           w_rec_init = w_rec_init, 
#           l1_h = l1_h, l2_h = l2_h, l1_weight = l1_weight, l2_weight = l2_weight)
#     #maddy added - end
#==============================================================================
