"""Main training loop"""

from __future__ import division

import os
import sys
import time
import pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf

from task import generate_trials
from network import Model, get_perf
import tools


def get_defaultconfig():
    '''Get a default configuration.

    Useful for debugging.

    Returns:
        config : a dictionary containing training configuration
    '''
    ruleset = 'mante'

    from task import get_num_ring, get_num_rule, rules_dict
    num_ring = get_num_ring(ruleset)
    n_rule = get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    config = {
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
            # 'w_rec_init'  : 'diag',
            # a default weak regularization prevents instability
            # 'l1_h'        : 1.0*0.0001,
            # l2 regularization on activity
            # 'l2_h'        : 1.0*0,
            # l2 regularization on weight
            # 'l2_weight'   : 0.0001*0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0.0001*0,
            # Stopping performance
            'target_perf': 1.,
            # random seed
            'seed': 0,
            # random number generator
            'rng': None,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # input, recurrent, output shape
            'shape': (n_input, 64, n_output),
            # name to save
            'save_name': 'test',
            # rules to train
            'rule_trains': rules_dict[ruleset],
            # rules to test
            'rules': rules_dict[ruleset],
            # learning rate
            'learning_rate': 0.01,
            # intelligent synapses parameters, tuple (c, ksi)
            'param_intsyn': None
            }
    return config


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
    config = model.config
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training '+rule_name_print)

    for rule_test in config['rules']:
        n_rep = 16
        batch_size_test_rep = int(config['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            trial = generate_trials(rule_test, config, 'random', batch_size=batch_size_test_rep)
            y_hat_test = model.get_y(trial.x)
            feed_dict = {model.x: trial.x,
                         model.y: trial.y.reshape(
                             (-1, config['shape'][2])),
                         model.c_mask: trial.c_mask}
            c_lsq, c_reg = sess.run([model.cost_lsq, model.cost_reg],
                                    feed_dict=feed_dict)

            # Cost is first summed over time, and averaged across batch and units
            # We did the averaging over time through c_mask

            # IMPORTANT CHANGES: take overall mean
            perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_'+rule_test].append(np.mean(clsq_tmp))
        log['creg_'+rule_test].append(np.mean(creg_tmp))
        log['perf_'+rule_test].append(np.mean(perf_tmp))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp))  +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp))  +
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

    tools.save_log(log, config['save_name'])

    return log


def train(save_name,
          ruleset='mante',
          n_hidden=64,
          learning_rate=0.01,
          reuse=False,
          seed=None,
          target_perf=0.9,
          activation='tanh',
          rnn_type='LeakyGRU',
          param_intsyn=None,
          batch_size_train=64,
          batch_size_test=128,
          display_step=500,
          training_iters=150000,
          rule_trains=None,
          rule_tests=None,
          rule_prob_map=None,
          run_analysis=None,
          w_rec_init='diag',
          l1_h=0,
          l2_h=0,
          l1_weight=0,
          l2_weight=0,
          sigma_rec=0.05,
          sigma_x=0.01,
          **kwargs):
    '''Train the network

    Args:
        save_name : model will be saved as save_name.ckpt
        n_hidden : number of hidden units
        learning_rate : learning rate
        reuse : if True, attempt to load a saved model from file
        seed : random seed
        target_perf : target performance
        activation : activation function
        rnn_type : type of recurrent network
        param_intsyn : parameteres for continual learning, currently disabled

        batch_size_train : batch size for training
        batch_size_test : batch size for testing
        display_step : number of batches between each display
        training_iters : maximum training iterations

        rule_trains : list of lists of trained rules.
         rule_trains = [list1, list2, list3, ...].
         Lists are trained sequentially.
         Within each list, rules are trained simultaneously

        rule_tests : list of rules to compute performance
        rule_prob_map : dictionary of relative probabilities of rules
            appearing during training

        run_analysis : list of strings deciding the analysis
            to be ran at the end

        w_rec_init   : diag, randortho, randgauss, only used for leaky_rec
        l1_h         : a default weak regularization on activity
            prevents instability
        l2_h         : l2 regularization on activity
        l1_weight    : l1 regularization on weight
        l2_weight    : l2 regularization on weight

    Returns:
        model is stored at save_name.ckpt
        training configuration is stored at 'config_'+'save_name.ckpt'
        training progress is stored at 'log_'+'save_name.ckpt'

    '''

    tools.mkdir_p('data')

    # Network parameters
    # Number of units each ring has
    from task import get_num_ring, get_num_rule, rules_dict
    num_ring = get_num_ring(ruleset)
    n_rule = get_num_rule(ruleset)
    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # automatically train all rules
        rule_trains = [rules_dict[ruleset]]

    if rule_tests is None:
        rule_tests = rules_dict[ruleset]

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    rule_probs = list()
    for rule_train in rule_trains:
        if not hasattr(rule_train, '__iter__'):
            rule_probs.append(None)
        else:
            # Set default as 1.
            rule_prob = np.array(
                    [rule_prob_map.get(r, 1.) for r in rule_train])
            rule_probs.append(rule_prob/np.sum(rule_prob))

    if reuse:
        # Build the model from save_name
        model = Model(config=save_name)
        config = model.config

    else:
        # Random number generator used
        rng = np.random.RandomState(seed)

        config = get_defaultconfig()
        config['ruleset'] = ruleset
        config['rnn_type'] = rnn_type
        config['activation'] = activation
        config['target_perf'] = target_perf
        config['seed'] = seed
        config['rng'] = rng
        config['n_eachring'] = n_eachring
        config['num_ring'] = num_ring
        config['rule_start'] = 1+num_ring*n_eachring
        config['shape'] = (n_input, n_hidden, n_output)
        config['save_name'] = save_name
        config['rule_trains'] = rule_trains
        config['rule_probs'] = rule_probs
        config['rules'] = rule_tests
        config['learning_rate'] = learning_rate
        config['param_intsyn'] = param_intsyn
        config['w_rec_init'] = w_rec_init
        config['l1_h'] = l1_h
        config['l2_h'] = l2_h
        config['l1_weight'] = l1_weight
        config['l2_weight'] = l2_weight
        config['sigma_rec'] = sigma_rec
        config['sigma_x'] = sigma_x

        # Allow for additional configuration options
        for key, val in kwargs.iteritems():
            config[key] = val

        tools.save_config(config, save_name)

        # Build the model
        model = Model(config=config)

    # Display configuration
    for key, val in config.iteritems():
        print('{:20s} = '.format(key) + str(val))

    # Number of training iterations for each rule
    rule_train_iters = []
    for rule_train in rule_trains:
        if not hasattr(rule_train, '__iter__'):
            tmp = 1
        else:
            tmp = len(rule_train)
        rule_train_iters.append(tmp*training_iters)
    print(rule_train_iters)

    # Using continual learning or not
    if (param_intsyn is not None) and (param_intsyn[0] > 0):
        intsyn = True
        c_intsyn, ksi_intsyn = param_intsyn
        print('Using continual learning')
    else:
        intsyn = False

    # Store results
    trials = []
    rule_now = []
    times = []
    cost_tests = {rule: [] for rule in rule_tests}
    creg_tests = {rule: [] for rule in rule_tests}
    perf_tests = {rule: [] for rule in rule_tests}
    cost_conts = []
    log = defaultdict(list())

    # Record time
    t_start = time.time()

    # Use customized session that launches the graph as well
    with tf.Session() as sess:
        if reuse:
            model.restore(sess)
        else:
            model.initialize(sess)

        # penalty on deviation from initial weight
        if config['l2_weight_init'] > 0:
            # TODO: Need checking
            pass
            anchor_vars = sess.run(model.var_list)

            for v, v_val in zip(model.var_list, anchor_vars):
                model.cost_reg += (config['l2_weight_init'] *
                                   tf.reduce_sum(tf.square(v-v_val)))

            model.set_optimizer()

        # Looping
        step_total = 1
        for i_rule_train, rule_train in enumerate(rule_trains):
            step = 1

            # At the beginning of new tasks
            if intsyn:
                update_intsyn()

            # Keep training until reach max iterations
            while step * batch_size_train <= rule_train_iters[i_rule_train]:
                try:
                    # Training
                    if not hasattr(rule_train, '__iter__'):
                        rule_train_now = rule_train
                    else:
                        rule_train_now = config['rng'].choice(
                                rule_train, p=rule_probs[i_rule_train])

                    # Generate a random batch of trials. 
                    # Each batch has the same trial length
                    trial = generate_trials(
                            rule_train_now, config, 'random',
                            batch_size=batch_size_train)

                    # Generating feed_dict.
                    feed_dict = {model.x: trial.x,
                                 model.y: trial.y.reshape((-1, n_output)),
                                 model.c_mask: trial.c_mask}

                    if not intsyn:
                        sess.run(model.optimizer, feed_dict=feed_dict)
                    else:
                        update_intsyn2()

                    # Validation
                    if step % display_step == 0:
                        log['trials'].append(step_total*batch_size_train)
                        log['times'].append(time.time()-t_start)
                        log['rule_now'].append(rule_train)
                        log = do_eval(sess, model, log, rule_train, rule_tests)
                        if log['perf_avg'] > model.config['target_perf']:
                            print('Perf reached the target: {:0.2f}'.format(
                                config['target_perf']))
                            break

                    step += 1
                    step_total += 1

                except KeyboardInterrupt:
                    print("Optimization interrupted by user")
                    break

        print("Optimization Finished!")

    # Run a set of standard analysis
    if run_analysis is None:
        run_analysis = list()

    if 'var' in run_analysis:
        # Compute variance
        from variance import compute_variance
        compute_variance(save_name)
        compute_variance(save_name, random_rotation=True)
        print('Computed variance')

    if 'psy' in run_analysis:
        # Psychometric analysis
        import performance
        for rule in ['dm1', 'contextdm1', 'multidm']:
            if rule in rule_tests:
                performance.compute_choicefamily_varytime(save_name, rule)

    if 'compare' in run_analysis:
        # Compute similarity with data and store results
        from contextdm_analysis import run_score
        log['score_train'], log['score_test'] = run_score(save_name)
        print('Data matching score : {:0.3f}'.format(log['score_test'].mean()))

        with open(os.path.join('data', 'log_'+save_name+'.pkl'), 'wb') as f:
            pickle.dump(log, f)

    if 'taskset' in run_analysis:
        # Run analysis for task representation
        from taskset import compute_taskspace, compute_replacerule_performance
        for setup in [1, 2]:
            compute_taskspace(save_name, setup, restore=False)
            compute_replacerule_performance(save_name, setup, restore=False)


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
    run_analysis = ['compare']
    train('debug', n_hidden=64, seed=2, activation='tanh',
          rnn_type='LeakyGRU', run_analysis=run_analysis)

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
    
