"""Main training loop"""

from __future__ import division

import sys
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import task
from task import generate_trials
from network import Model, get_perf
from analysis import variance
import tools


def get_default_hp(ruleset):
    '''Get a default hp.

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
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
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
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            }

    return hp


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

    # TODO: This needs to be fixed since now rules are strings
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


def display_rich_output(model, sess, step, log, model_dir):
    """Display step by step outputs during training."""
    variance._compute_variance_bymodel(model, sess)
    rule_pair = ['contextdm1', 'contextdm2']
    save_name = '_atstep' + str(step)
    title = ('Step ' + str(step) +
             ' Perf. {:0.2f}'.format(log['perf_avg'][-1]))
    variance.plot_hist_varprop(model_dir, rule_pair,
                               figname_extra=save_name,
                               title=title)
    plt.close('all')


def train(model_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          rich_output=False,
          load_dir=None,
          trainables=None,
          ):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

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
    tools.save_hp(hp, model_dir)

    # Build the model
    model = Model(model_dir, hp=hp)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir
    
    # Record time
    t_start = time.time()

    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        elif trainables == 'input':
            # train all nputs
            var_list = [v for v in model.var_list
                        if ('input' in v.name) and ('rnn' not in v.name)]
        elif trainables == 'rule':
            # train rule inputs only
            var_list = [v for v in model.var_list if 'rule_input' in v.name]
        else:
            raise ValueError('Unknown trainables')
        model.set_optimizer(var_list=var_list)

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] *
                                   tf.nn.l2_loss(w - w_val))

            model.set_optimizer(var_list=var_list)

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
            model.set_optimizer(var_list=var_list)

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
                        display_rich_output(model, sess, step, log, model_dir)

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


def train_sequential(
        model_dir,
        rule_trains,
        hp=None,
        max_steps=1e7,
        display_step=500,
        ruleset='mante',
        seed=0,
        ):
    '''Train the network sequentially.

    Args:
        model_dir: str, training directory
        rule_trains: a list of list of tasks to train sequentially
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps for each list of tasks
        display_step: int, display steps
        ruleset: the set of rules to train
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    '''

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_trains'] = rule_trains
    # Get all rules by flattening the list of lists
    hp['rules'] = [r for rs in rule_trains for r in rs]

    # Number of training iterations for each rule
    rule_train_iters = [len(r)*max_steps for r in rule_trains]

    tools.save_hp(hp, model_dir)
    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Using continual learning or not
    c, ksi = hp['c_intsyn'], hp['ksi_intsyn']

    # Build the model
    model = Model(model_dir, hp=hp)
    
    grad_unreg = tf.gradients(model.cost_lsq, model.var_list)

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # tensorboard summaries
    placeholders = list()
    for v_name in ['Omega0', 'omega0', 'vdelta']:
        for v in model.var_list:
            placeholder = tf.placeholder(tf.float32, shape=v.shape)
            tf.summary.histogram(v_name + '/' + v.name, placeholder)
            placeholders.append(placeholder)
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(model_dir + '/tb')

    def relu(x):
        return x * (x > 0.)

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            raise NotImplementedError()

        # Looping
        step_total = 0
        for i_rule_train, rule_train in enumerate(hp['rule_trains']):
            step = 0

            # At the beginning of new tasks
            # Only if using intelligent synapses
            v_current = sess.run(model.var_list)

            if i_rule_train == 0:
                v_anc0 = v_current
                Omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
                omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
                v_delta = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
            elif c > 0:
                v_anc0_prev = v_anc0
                v_anc0 = v_current
                v_delta = [v-v_prev for v, v_prev in zip(v_anc0, v_anc0_prev)]

                # Make sure all elements in omega0 are non-negative
                # Penalty
                Omega0 = [relu(O + o / (v_d ** 2 + ksi))
                          for O, o, v_d in zip(Omega0, omega0, v_delta)]
                
                # Update cost
                model.cost_reg = tf.constant(0.)
                for v, w, v_val in zip(model.var_list, Omega0, v_current):
                    model.cost_reg += c * tf.reduce_sum(
                        tf.multiply(tf.constant(w),
                                    tf.square(v - tf.constant(v_val))))
                model.set_optimizer()

            # Store Omega0 to tf summary
            feed_dict = dict(zip(placeholders, Omega0 + omega0 + v_delta))
            summary = sess.run(merged, feed_dict=feed_dict)
            test_writer.add_summary(summary, i_rule_train)

            # Reset
            omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]

            # Keep training until reach max iterations
            while (step * hp['batch_size_train'] <=
                   rule_train_iters[i_rule_train]):
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
                rule_train_now = hp['rng'].choice(rule_train)
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                        rule_train_now, hp, 'random',
                        batch_size=hp['batch_size_train'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)

                # Continual learning with intelligent synapses
                v_prev = v_current

                # This will compute the gradient BEFORE train step
                _, v_grad = sess.run([model.train_step, grad_unreg],
                                     feed_dict=feed_dict)
                # Get the weight after train step
                v_current = sess.run(model.var_list)

                # Update synaptic importance
                omega0 = [
                    o - (v_c - v_p) * v_g for o, v_c, v_p, v_g in
                    zip(omega0, v_current, v_prev, v_grad)
                ]

                step += 1
                step_total += 1

        print("Optimization Finished!")


def train_rule_only(
        model_dir,
        rule_trains,
        max_steps,
        hp=None,
        ruleset='all',
        seed=0,
):
    '''Customized training function.

    The network sequentially but only train rule for the second set.
    First train the network to perform tasks in group 1, then train on group 2.
    When training group 2, only rule connections are being trained.

    Args:
        model_dir: str, training directory
        rule_trains: a list of list of tasks to train sequentially
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps for each list of tasks
        display_step: int, display steps
        ruleset: the set of rules to train
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    '''

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['rule_trains'] = rule_trains
    # Get all rules by flattening the list of lists
    hp['rules'] = [r for rs in rule_trains for r in rs]

    # Number of training iterations for each rule
    if hasattr(max_steps, '__iter__'):
        rule_train_iters = max_steps
    else:
        rule_train_iters = [len(r) * max_steps for r in rule_trains]

    tools.save_hp(hp, model_dir)
    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Build the model
    model = Model(model_dir, hp=hp)

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            raise NotImplementedError()

        # Looping
        step_total = 0
        for i_rule_train, rule_train in enumerate(hp['rule_trains']):
            step = 0

            if i_rule_train == 0:
                display_step = 200
            else:
                display_step = 50

            if i_rule_train > 0:
                # var_list = [v for v in model.var_list
                #             if ('input' in v.name) and ('rnn' not in v.name)]
                var_list = [v for v in model.var_list if 'rule_input' in v.name]
                model.set_optimizer(var_list=var_list)

            # Keep training until reach max iterations
            while (step * hp['batch_size_train'] <=
                   rule_train_iters[i_rule_train]):
                # Validation
                if step % display_step == 0:
                    trial = step_total * hp['batch_size_train']
                    log['trials'].append(trial)
                    log['times'].append(time.time() - t_start)
                    log['rule_now'].append(rule_train)
                    log = do_eval(sess, model, log, rule_train)
                    if log['perf_avg'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        break

                # Training
                rule_train_now = hp['rng'].choice(rule_train)
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                    rule_train_now, hp, 'random',
                    batch_size=hp['batch_size_train'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)

                # This will compute the gradient BEFORE train step
                _ = sess.run(model.train_step, feed_dict=feed_dict)

                step += 1
                step_total += 1

        print("Optimization Finished!")


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='data/debug')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {'activation': 'softplus',
          'n_rnn': 64,
          'mix_rule': True,
          'l1_h': 0.,
          'use_separate_input': True}
    train(args.modeldir,
          seed=1,
          hp=hp,
          ruleset='all',
          rule_trains=['contextdelaydm1', 'contextdelaydm2',
                                 'contextdm1', 'contextdm2'],
          display_step=500)