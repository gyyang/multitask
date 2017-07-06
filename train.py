"""
2016/06/03 @ Guangyu Robert Yang

Main training loop and network structure

This code runs with tensorflow 0.11
"""

from __future__ import division

import os
import sys
import time
import pickle
import errno

import matplotlib.pyplot as plt
import tensorflow as tf

from run import Run
from task import *
from network import get_perf

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

def train(HDIM=256, s=2, learning_rate=0.001, training_iters=3000000,
          batch_size_train=64, batch_size_test=256, display_step=1000,
          save_addon=None, seed=None, **kwargs):
    '''
    Training the network
    :param HDIM: Number of recurrent units
    :param s: Type of training tasks
    :param learning_rate: default 0.001. This is a good default value.
    :param training_iters:
    :param save_addon:
    :return:
    '''

    mkdir_p('data')

    # Number of input rings
    num_ring = 2
    # Number of units each ring has
    N_RING = 32
    # N_RING = 16

    sigma_rec = 0.15
    # sigma_rec = 0.05
    sigma_x   = 0.01
    if 'beta_anchor' in kwargs:
        beta_anchor = kwargs['beta_anchor']
    else:
        beta_anchor = 0.0

    w_rec_init = 'diag'
    early_stop = None # If not None, stop at this performance level

    rng  = np.random.RandomState(seed)

    if s == -1:
        save_addon_type = 'debug_relu'
    elif s == 0:
        save_addon_type = 'allrule_softplus'
    elif s == 1:
        save_addon_type = 'allrule_tanh'
    elif s == 2:
        save_addon_type = 'allrule_relu'
    elif s == 3:
        save_addon_type = 'attendonly_softplus'
    elif s == 4:
        save_addon_type = 'attendonly_tanh'
    elif s == 5:
        save_addon_type = 'attendonly_relu'
    elif s == 6:
        save_addon_type = 'choiceonly_softplus'
    elif s == 7:
        save_addon_type = 'choiceonly_tanh'
    elif s == 8:
        save_addon_type = 'choiceonly_relu'
    elif s == 9:
        save_addon_type = 'delaychoiceonly_softplus'
    elif s == 10:
        save_addon_type = 'delaychoiceonly_tanh'
    elif s == 11:
        save_addon_type = 'delaychoiceonly_relu'
    elif s == 12:
        save_addon_type = 'oicdmconly_softplus'
    elif s == 13:
        save_addon_type = 'matchfamily_softplus'
    elif s == 14:
        save_addon_type = 'choicefamily_softplus'
    elif s == 15:
        save_addon_type = 'goantifamily_softplus'
    elif s == 16:
        save_addon_type = 'allrule_relu_randortho'
    elif s == 17:
        save_addon_type = 'attendonly_relu_randortho'
    elif s == 18:
        save_addon_type = 'allrule_relu_randgauss'
    elif s == 19:
        save_addon_type = 'attendonly_relu_randgauss'
    elif s == 20:
        save_addon_type = 'allrule_softplus_randortho'
    elif s == 21:
        save_addon_type = 'attendonly_softplus_randortho'
    elif s == 22:
        save_addon_type = 'attendonly_suplin'


    tf.reset_default_graph()


    if 'allrule' in save_addon_type:
        # Rules
        rules = range(N_RULE)

    elif 'attendonly' in save_addon_type:
        rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
        early_stop = 0.95
    elif 'delaychoiceonly' in save_addon_type: # This has to be before choiceonly
        rules = [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2]
    elif 'choiceonly' in save_addon_type:
        rules = [CHOICE_MOD1, CHOICE_MOD2]
    elif 'matchfamily' in save_addon_type:
        rules = [DMSGO, DMSNOGO, DMCGO, DMCNOGO]
    elif 'choicefamily' in save_addon_type:
        rules = [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
    elif 'goantifamily' in save_addon_type:
        rules = [FDGO, REACTGO, DELAYGO, FDANTI, REACTANTI, DELAYANTI]
    elif 'oicdmconly' in save_addon_type:
        rules = [OIC, DMC]
    elif 'debug' in save_addon_type:
        rules = [REACTGO, DELAYGO, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

    if OIC in rules:
        num_ring = 3

    if 'softplus' in save_addon_type:
        activation = 'softplus'
    elif 'tanh' in save_addon_type:
        activation = 'tanh'
    elif 'relu' in save_addon_type:
        activation = 'relu'
    elif 'suplin' in save_addon_type:
        activation = 'suplin'

    rule_weights = np.ones(len(rules))

    if 'allrule' in save_addon_type:
        # Make them 5 times more common
        rule_weights[rules.index(CHOICEATTEND_MOD1)] = 5
        rule_weights[rules.index(CHOICEATTEND_MOD2)] = 5

    if 'randortho' in save_addon_type:
        w_rec_init = 'randortho'
    elif 'randgauss' in save_addon_type:
        w_rec_init = 'randgauss'

    if save_addon is None:
        save_addon = save_addon_type
    else:
        save_addon = save_addon_type + '_' + save_addon

    config = {'h_type'      : 'leaky_rec',
              'loss_type'   : 'lsq',
              'activation'  : activation,
              'alpha'       : 0.2, # \Delta t/tau
              'dt'          : 0.2*TAU,
              'sigma_rec'   : sigma_rec,
              'sigma_x'     : sigma_x,
              'w_rec_init'  : w_rec_init,
              'beta_anchor' : beta_anchor,
              'l_ewc'       : 0.,
              'early_stop'  : early_stop,
              'seed'        : seed,
              'rng'         : rng,
              'HDIM'        : HDIM,
              'N_RING'      : N_RING,
              'num_ring'    : num_ring,
              'rule_start'  : 1+num_ring*N_RING,
              'shape'       : (1+num_ring*N_RING+N_RULE, HDIM, N_RING+1),
              'save_addon'  : save_addon,
              'rules'       : rules,
              'rule_weights': rule_weights,
              'learning_rate': learning_rate,
              'training_iters' : training_iters,
              'batch_size_train' : batch_size_train,
              'batch_size_test' : batch_size_test
              }

    for key, val in kwargs.iteritems():
        config[key] = val

    for key, val in config.iteritems():
        print('{:20s} = '.format(key) + str(val))

    # Network Parameters
    n_input, n_hidden, n_output = config['shape']

    # Store results
    trials     = []
    times      = []
    cost_tests = {rule:[] for rule in rules}
    creg_tests = {rule:[] for rule in rules}
    perf_tests = {rule:[] for rule in rules}
    delta_ws   = []

    # Launch the graph
    t_start = time.time()

    # Use customized session that launches the graph as well
    with Run(config=config) as R:
        step = 1
        # Keep training until reach max iterations
        while step * batch_size_train <= training_iters:
            try:
                # Training
                rule = rng.choice(rules, p=rule_weights/rule_weights.sum())
                task = generate_onebatch(rule, config, 'random', batch_size=batch_size_train)

                feed_dict = {R.x: task.x,
                             R.y: task.y.reshape((-1,n_output)),
                             R.c_mask: task.c_mask}
                R.run(R.optimizer, feed_dict=feed_dict)

                # Validation
                if step % display_step == 0:
                    trials.append(step*batch_size_train)
                    times.append(time.time()-t_start)
                    print('Trial {:7d}'.format(trials[-1]) +
                          '  | Time {:0.2f} s'.format(times[-1]))

                    # delta_ws.append(R.f_delta_w())

                    for rule_test in rules:
                        n_rep = 20
                        batch_size_test_rep = int(batch_size_test/n_rep)
                        clsq_tmp = list()
                        creg_tmp = list()
                        perf_tmp = list()
                        for i_rep in range(n_rep):
                            task = generate_onebatch(rule_test, config, 'random', batch_size=batch_size_test_rep)
                            y_hat_test = R.f_y_from_x(task.x)
                            feed_dict = {R.x: task.x,
                                         R.y: task.y.reshape((-1,n_output)),
                                         R.c_mask: task.c_mask}
                            c_lsq, c_reg = R.run([R.cost_lsq, R.cost_reg], feed_dict=feed_dict)

                            # Cost is first summed over time, and averaged across batch and units
                            # We did the averaging over time through c_mask

                            # IMPORTANT CHANGES: take overall mean
                            perf_test = np.mean(get_perf(y_hat_test, task.y_loc))
                            clsq_tmp.append(c_lsq)
                            creg_tmp.append(c_reg)
                            perf_tmp.append(perf_test)

                        cost_tests[rule_test].append(np.mean(clsq_tmp))
                        creg_tests[rule_test].append(np.mean(creg_tmp))
                        perf_tests[rule_test].append(np.mean(perf_tmp))
                        print('{:15s}'.format(rule_name[rule_test]) +
                              '| cost {:0.3f}'.format(cost_tests[rule_test][-1])  +
                              '| c_reg {:0.3f}'.format(creg_tests[rule_test][-1])  +
                              '  | perf {:0.2f}'.format(perf_tests[rule_test][-1]))
                        sys.stdout.flush()

                    # Saving the model
                    R.save()

                    config['trials']     = trials
                    config['times']      = times
                    config['cost_tests'] = cost_tests
                    config['perf_tests'] = perf_tests
                    config['delta_ws']   = delta_ws
                    with open(os.path.join('data', 'config'+config['save_addon']+'.pkl'), 'wb') as f:
                        pickle.dump(config, f)

                    if early_stop is not None:
                        perf_tests_mean = np.mean([perf_tests[rule][-1] for rule in rules])
                        if perf_tests_mean > early_stop:
                            print('Performance reached early stopping point')
                            break

                step += 1
                
            except KeyboardInterrupt:
                break

        # Saving the model
        R.save()

        config['trials']     = trials
        config['times']      = times
        config['cost_tests'] = cost_tests
        config['creg_tests'] = creg_tests
        config['perf_tests'] = perf_tests
        config['delta_ws']   = delta_ws
        with open(os.path.join('data', 'config'+config['save_addon']+'.pkl'), 'wb') as f:
            pickle.dump(config, f)

        print("Optimization Finished!")

    if 'debug' not in save_addon_type:
        from variance import compute_variance
        compute_variance(config['save_addon'], 'rule', rules, fast_eval=True)
        compute_variance(config['save_addon'], 'epoch', rules, fast_eval=True)
        print('Computed variance')

        if 'allrule' in save_addon_type:
            rule_list_performance = [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
        elif 'attendonly' in save_addon_type:
            rule_list_performance = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
        else:
            rule_list_performance = []

        from performance import compute_choicefamily_varytime
        for rule in rule_list_performance:
            compute_choicefamily_varytime(save_addon, rule)

        from taskset import compute_taskspace, compute_replacerule_performance
        for setup in [1, 2]:
            compute_taskspace(save_addon, setup, restore=False)
            compute_replacerule_performance(save_addon, setup, restore=False)

def train_cont(HDIM=256, s=3, learning_rate=0.01, training_iters=100000,
          batch_size_train=64, batch_size_test=256, display_step=500,
          save_addon=None, seed=None, early_stop=None, **kwargs):
    '''
    Continually training the network
    :param HDIM: Number of recurrent units
    :param s: Type of training tasks
    :param learning_rate: default 0.001. This is a good default value.
    :param training_iters:
    :param save_addon:
    :return:
    '''

    mkdir_p('data')
    tf.reset_default_graph()

    # Number of input rings
    num_ring = 2
    # Number of units each ring has
    # N_RING = 16
    N_RING = 32

    # early_stop = None
    # early_stop = 0.90

    # sigma_rec = 0.15
    sigma_rec = 0.05
    sigma_x   = 0.01
    if 'beta_anchor' in kwargs:
        beta_anchor = kwargs['beta_anchor']
    else:
        beta_anchor = 0.0

    w_rec_init = 'diag'

    rng  = np.random.RandomState(seed)

    # # In the order of training appearance
    # rule_trains = [CHOICE_INT, (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2)]
    # rule_tests = [CHOICE_INT, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    # # Proportion of training trials for each rule (or rule pairs)
    # rule_train_weights = np.array([1, 10])

    if s == 1:
        # rule_trains = [REACTGO, REACTANTI, FDGO, FDANTI, DELAYGO, DELAYANTI,
        #                (CHOICE_MOD1, CHOICE_MOD2), (CHOICEDELAY_MOD1, CHOICEDELAY_MOD2),
        #                (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2), (CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2),
        #                 CHOICE_INT, CHOICEDELAY_INT]

        rule_trains = [FDGO, DELAYGO,
        CHOICE_MOD1, CHOICE_MOD2, (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2), CHOICE_INT,
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, (CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2), CHOICEDELAY_INT]

        rule_tests = [FDGO, DELAYGO,
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT]

        save_addon_type = 'cont_allrule'

    elif s == 2:
        rule_trains = [(REACTGO, REACTANTI), (FDGO, FDANTI), (DELAYGO, DELAYANTI),
        CHOICE_MOD1, CHOICE_MOD2, (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2), CHOICE_INT,
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, (CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2), CHOICEDELAY_INT]

        rule_tests = [REACTGO, REACTANTI, FDGO, FDANTI, DELAYGO, DELAYANTI,
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT]

        save_addon_type = 'cont_allrule'
        
    elif s == 3:
        rule_trains = [(REACTGO, DELAYGO), (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2)]

        rule_tests = [REACTGO, DELAYGO, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

        save_addon_type = 'cont_type3'

    elif s == 4:
        rule_trains = [REACTGO, DELAYGO, (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2), CHOICE_INT]

        rule_tests = [REACTGO, DELAYGO, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]

        save_addon_type = 'cont_type4'

    elif s == 5:

        rule_trains = [CHOICE_INT, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICEATTEND_MOD1]

        rule_tests = [CHOICE_INT, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

        save_addon_type = 'debuging'

    elif s == 6:
        rule_trains = [REACTGO, DELAYGO, (CHOICE_MOD1, CHOICE_MOD2)]

        rule_tests = [REACTGO, DELAYGO, CHOICE_MOD1, CHOICE_MOD2]

        save_addon_type = 'cont_type6'

    elif s == 7:
        rule_trains = [REACTGO, DELAYGO, (CHOICE_MOD1, CHOICE_MOD2), (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2)]

        rule_tests = [REACTGO, DELAYGO, CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

        save_addon_type = 'cont_type7'

    elif s == 8:
        # rule_trains = [REACTGO, REACTANTI, DELAYGO, DELAYANTI, (CHOICE_MOD1, CHOICE_MOD2),
        #                (CHOICEDELAY_MOD1, CHOICEDELAY_MOD2), (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2)]

        rule_tests = [REACTGO, REACTANTI, DELAYGO, DELAYANTI, CHOICE_MOD1, CHOICE_MOD2,
                      CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2,
                      CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, DMSGO, DMSNOGO, DMCGO, DMCNOGO]

        rule_trains = rule_tests

        save_addon_type = 'cont_type8'

    # rule_train_iters = (training_iters*rule_train_weights/np.sum(rule_train_weights)).astype(int)
    # rule_train_iters = [training_iters] * len(rule_trains)

    # Number of training iterations for each rule
    rule_train_iters = []
    for rule_train in rule_trains:
        if not hasattr(rule_train, '__iter__'):
            tmp = 1
        else:
            tmp = len(rule_train)
        rule_train_iters.append(tmp*training_iters)

    activation = 'relu'
    # activation = 'softplus' # has substantial difficulty learning Ctx DM 1 & 2

    if save_addon is None:
        save_addon = save_addon_type
    else:
        save_addon = save_addon_type + '_' + save_addon

    config = {'easy_task'   : True,
              'h_type'      : 'leaky_rec',
              'loss_type'   : 'lsq',
              'activation'  : activation,
              'alpha'       : 0.2, # \Delta t/tau
              'dt'          : 0.2*TAU,
              'sigma_rec'   : sigma_rec,
              'sigma_x'     : sigma_x,
              'w_rec_init'  : w_rec_init,
              'beta_anchor' : beta_anchor,
              'l_ewc'       : 0., # elastic weight (Kirkpatrick et al.) (Doesn't work for these tasks)
              'c_intsyn'    : 0.1, # intelligent synapse (Zenke et al.) Value 0.1 is found by grid search
              'ksi_intsyn'  : 0.01, # Value 0.01 found by grid search as well
              'early_stop'  : early_stop,
              'seed'        : seed,
              'rng'         : rng,
              'HDIM'        : HDIM,
              'N_RING'      : N_RING,
              'num_ring'    : num_ring,
              'rule_start'  : 1+num_ring*N_RING,
              'shape'       : (1+num_ring*N_RING+N_RULE, HDIM, N_RING+1),
              'save_addon'  : save_addon,
              'rule_trains' : rule_trains,
              'rules'       : rule_tests,
              'learning_rate': learning_rate,
              'training_iters' : training_iters,
              'batch_size_train' : batch_size_train,
              'batch_size_test' : batch_size_test
              }

    for key, val in kwargs.iteritems():
        config[key] = val

    for key, val in config.iteritems():
        print('{:20s} = '.format(key) + str(val))

    if config['l_ewc'] > 0:
        config['l_ewc_sqrt'] = np.sqrt(config['l_ewc']) # For actual computation
        ewc = True
    else:
        ewc = False

    if config['c_intsyn'] > 0:
        intsyn = True
    else:
        intsyn = False

    assert not (ewc and intsyn) # can't both be true

    # Network Parameters
    n_input, n_hidden, n_output = config['shape']

    # Store results
    trials     = []
    rule_now   = []
    times      = []
    cost_tests = {rule:[] for rule in rule_tests}
    creg_tests = {rule:[] for rule in rule_tests}
    perf_tests = {rule:[] for rule in rule_tests}
    delta_ws   = []
    cost_conts  = []

    # Launch the graph
    t_start = time.time()

    # Use customized session that launches the graph as well
    with Run(config=config) as R:
        step_total = 1 # initialize only
        for i_rule_train, rule_train in enumerate(rule_trains):
            step = 1
            perf_test_early_stop = 0

            v_current = R.f_vars() # intialize

            # At the beginning of new tasks
            if ewc:
                start = time.time()
                print('Computing Fisher Diagonal...'),

                # Compute the Fisher diagonal and the anchor weight
                if i_rule_train > 0:

                    rule_fisher = rule_trains[i_rule_train-1]
                    if not hasattr(rule_fisher, '__iter__'):
                        rule_fisher = [rule_fisher]

                    n_samples = 500
                    F = [np.zeros_like(v) for v in v_current]
                    for j in range(n_samples):
                        r = rng.choice(rule_fisher)
                        task = generate_onebatch(r, config, 'random', batch_size=1)

                        feed_dict = {R.x: task.x,
                                     R.y: task.y.reshape((-1,n_output)),
                                     R.c_mask: task.c_mask}
                        grad_eval = R.run(R.grad_lsq, feed_dict=feed_dict)

                        for f, g in zip(F, grad_eval):
                            f += g**2

                    for f in F:
                        f /= n_samples

                    R.update_cost(F, v_current, config['l_ewc']/2.)

                print(' Finished. Time taken {:0.2f}s'.format(time.time()-start))

            elif intsyn:
                if i_rule_train == 0:
                    v_anc0 = R.f_vars()
                    Omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]
                else:
                    v_anc0_prev = v_anc0
                    v_anc0 = R.f_vars()
                    v_delta = [v - v_prev for v, v_prev in zip(v_anc0, v_anc0_prev)]

                    # Make sure all elements in omega0 are non-negative
                    # Penalty
                    Omega0 = [(O + o*(o>0.)/(v_d**2+config['ksi_intsyn']))
                              for O,o,v_d in zip(Omega0,omega0,v_delta)]

                    # for v_d in v_delta:
                    #     print(np.percentile((v_d.flatten())**2, [1, 50, 99]))

                    R.update_cost(Omega0, v_current, config['c_intsyn'], reset=True)

                # Reset
                omega0 = [np.zeros(v.shape, dtype='float32') for v in v_anc0]




            # Keep training until reach max iterations
            while step * batch_size_train <= rule_train_iters[i_rule_train]:
                try:
                    # Training
                    if not hasattr(rule_train, '__iter__'):
                        rule_train_now = rule_train
                    else:
                        rule_train_now = rng.choice(rule_train)
                    task = generate_onebatch(rule_train_now, config, 'random', batch_size=batch_size_train)

                    feed_dict = {R.x: task.x,
                                 R.y: task.y.reshape((-1,n_output)),
                                 R.c_mask: task.c_mask}
                    if not intsyn:
                        R.run(R.optimizer, feed_dict=feed_dict)

                    else:
                        v_prev = v_current

                        # R.run(R.optimizer, feed_dict=feed_dict)

                        _, grads_and_vars_ = R.run([R.optimizer, R.grads_and_vars],
                                                   feed_dict=feed_dict)
                        v_grad, v_current = zip(*grads_and_vars_) # Unzip

                        # _, v_grad = R.run([R.optimizer, R.grad_lsq],
                        #                            feed_dict=feed_dict)
                        # v_current = R.run(tf.trainable_variables())

                        # Update synaptic importance
                        omega0 = [o-(v_c-v_p)*v_g
                                  for o,v_c,v_p,v_g in zip(omega0, v_current, v_prev, v_grad)]


                    # Validation
                    if step % display_step == 0:
                        trials.append(step_total*batch_size_train)
                        times.append(time.time()-t_start)
                        rule_now.append(rule_train)

                        if not hasattr(rule_train, '__iter__'):
                            rule_name_print = rule_name[rule_train]
                        else:
                            rule_name_print = ' & '.join([rule_name[r] for r in rule_train])

                        print('Trial {:7d}'.format(trials[-1]) +
                              '  | Time {:0.2f} s'.format(times[-1]) +
                              '  | Now training '+rule_name_print)

                        for rule_test in rule_tests:
                            # n_rep = 20
                            n_rep = batch_size_test
                            batch_size_test_rep = int(batch_size_test/n_rep)
                            clsq_tmp = list()
                            creg_tmp = list()
                            perf_tmp = list()
                            for i_rep in range(n_rep):
                                task = generate_onebatch(rule_test, config, 'random', batch_size=batch_size_test_rep)
                                y_hat_test = R.f_y_from_x(task.x)
                                feed_dict = {R.x: task.x,
                                             R.y: task.y.reshape((-1,n_output)),
                                             R.c_mask: task.c_mask}
                                c_lsq, c_reg = R.run([R.cost_lsq, R.cost_reg], feed_dict=feed_dict)

                                # Cost is first summed over time, and averaged across batch and units
                                # We did the averaging over time through c_mask

                                # IMPORTANT CHANGES: take overall mean
                                perf_test = np.mean(get_perf(y_hat_test, task.y_loc))
                                clsq_tmp.append(c_lsq)
                                creg_tmp.append(c_reg)
                                perf_tmp.append(perf_test)

                            cost_tests[rule_test].append(np.mean(clsq_tmp))
                            creg_tests[rule_test].append(np.mean(creg_tmp))
                            perf_tests[rule_test].append(np.mean(perf_tmp))
                            print('{:15s}'.format(rule_name[rule_test]) +
                                  '| cost {:0.3f}'.format(cost_tests[rule_test][-1])  +
                                  '| c_reg {:0.3f}'.format(creg_tests[rule_test][-1])  +
                                  '  | perf {:0.2f}'.format(perf_tests[rule_test][-1]))
                            sys.stdout.flush()

                        # Saving the model
                        R.save()
                        config['trials']     = trials
                        config['times']      = times
                        config['rule_now']   = rule_now
                        config['cost_tests'] = cost_tests
                        config['perf_tests'] = perf_tests
                        config['delta_ws']   = delta_ws
                        config['cost_conts']  = cost_conts
                        with open(os.path.join('data', 'config'+config['save_addon']+'.pkl'), 'wb') as f:
                            pickle.dump(config, f)

                        if early_stop is not None:
                            if hasattr(rule_train, '__iter__'):
                                rule_tmp = rule_train
                            else:
                                rule_tmp = [rule_train]
                            perf_tests_mean = np.mean([perf_tests[r][-1] for r in rule_tmp])
                            if perf_tests_mean > early_stop:
                                perf_test_early_stop += 1
                            else:
                                perf_test_early_stop = 0

                            if perf_test_early_stop >= 3:
                                # Has to pass the threshold for two consecutive tests
                                print('Performance reached early stopping point')
                                step = 1e15 # This will break out of the loop

                    step += 1
                    step_total += 1

                except KeyboardInterrupt:
                    break

        # Saving the model
        R.save()

        config['trials']     = trials
        config['times']      = times
        config['rule_now']   = rule_now
        config['cost_tests'] = cost_tests
        config['creg_tests'] = creg_tests
        config['perf_tests'] = perf_tests
        config['delta_ws']   = delta_ws
        config['cost_conts'] = cost_conts
        with open(os.path.join('data', 'config'+config['save_addon']+'.pkl'), 'wb') as f:
            pickle.dump(config, f)

        print("Optimization Finished!")


    from variance import compute_variance
    compute_variance(config['save_addon'], 'rule', rule_tests, fast_eval=True)
    print('Computed variance')

    # from performance import compute_choicefamily_varytime
    # for rule in rules:
    #     compute_choicefamily_varytime(save_addon, rule)

if __name__ == '__main__':
    pass
    train(HDIM=200, s=15, save_addon='debug', training_iters=300000, learning_rate=0.01,
          batch_size_train=50, batch_size_test=100, display_step=200, seed=1)

    # train_cont(HDIM=200, s=2, save_addon='test', training_iters=150000,
    #       batch_size_train=50, batch_size_test=200, display_step=100, seed=2, c_intsyn=10000)

#==============================================================================
#     train_cont(HDIM=200, s=2, save_addon='test', training_iters=1000000, learning_rate=0.01,
#           batch_size_train=50, batch_size_test=100, display_step=200, seed=10,
#                c_intsyn=0.1, ksi_intsyn=0.01)
# 
#==============================================================================

#==============================================================================
#     train_cont(HDIM=300, s=4, save_addon='test', training_iters=1000000, learning_rate=0.01,
#           batch_size_train=50, batch_size_test=100, display_step=200, seed=1,
#                c_intsyn=1.0, ksi_intsyn=0.1)
#==============================================================================
