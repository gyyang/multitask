"""
2016/06/03 @ Guangyu Robert Yang

Main training loop and network structure

This code runs with tensorflow 0.11
"""

from __future__ import division

import os
import time
import pickle
import errno

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

def train(HDIM=300, s=1, learning_rate=0.001, training_iters=2000000,
          batch_size_train=50, batch_size_test=2000, display_step=1000, save_addon=None):
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


    tf.reset_default_graph()

    N_RING = 16

    sigma_rec = 0.05
    sigma_x   = 0.01

    if 'allrule' in save_addon_type:
        # Rules
        rules = range(N_RULE)

    elif 'attendonly' in save_addon_type:
        rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    elif 'delaychoiceonly' in save_addon_type: # This has to be before choiceonly
        rules = [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2]
    elif 'choiceonly' in save_addon_type:
        rules = [CHOICE_MOD1, CHOICE_MOD2]
    elif 'matchfamily' in save_addon_type:
        rules = [DMSGO, DMSNOGO, DMCGO, DMCNOGO]
    elif 'choicefamily' in save_addon_type:
        rules = [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
    elif 'goantifamily' in save_addon_type:
        rules = [GO, INHGO, DELAYGO, REMAP, INHREMAP, DELAYREMAP]
    elif 'oicdmconly' in save_addon_type:
        rules = [OIC, DMC]

    if OIC in rules:
        num_ring = 3

    if 'softplus' in save_addon_type:
        activation = 'softplus'
    elif 'tanh' in save_addon_type:
        activation = 'tanh'
    elif 'relu' in save_addon_type:
        activation = 'relu'

    rule_weights = np.ones(len(rules))

    if 'allrule' in save_addon_type:
        # Make them 5 times more common
        rule_weights[rules.index(CHOICEATTEND_MOD1)] = 5
        rule_weights[rules.index(CHOICEATTEND_MOD2)] = 5

    if save_addon is None:
        save_addon = save_addon_type
    else:
        save_addon = save_addon_type + '_' + save_addon

    config = {'h_type'      : 'leaky_rec',
              'activation'  : activation,
              'alpha'       : 0.2, # \Delta t/tau
              'dt'          : 0.2*TAU,
              'sigma_rec'   : sigma_rec,
              'sigma_x'     : sigma_x,
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
              'batch_size_test' : batch_size_test}

    for key, val in config.iteritems():
        print('{:20s} = '.format(key) + str(val))

    # Network Parameters
    n_input, n_hidden, n_output = config['shape']

    # Store results
    trials     = []
    times      = []
    cost_tests = {rule:[] for rule in rules}
    perf_tests = {rule:[] for rule in rules}

    # Launch the graph
    t_start = time.time()

    # Use customized session that launches the graph as well
    with Run(config=config) as R:

        step = 1
        # Keep training until reach max iterations
        while step * batch_size_train < training_iters:
            try:
                # Training
                rule = np.random.choice(rules, p=rule_weights/rule_weights.sum())
                task = generate_onebatch(rule, config, 'random', batch_size=batch_size_train)

                R.train_one_step(task.x,
                                 task.y.reshape((-1,n_output)),
                                 task.c_mask.reshape((-1,n_output)))

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
                            y_hat_test = R.f_y_from_x(task.x)

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
        R.save()

        config['trials']     = trials
        config['times']      = times
        config['cost_tests'] = cost_tests
        config['perf_tests'] = perf_tests
        with open(os.path.join('data', 'config'+config['save_addon']+'.pkl'), 'wb') as f:
            pickle.dump(config, f)

        print("Optimization Finished!")


    from variance import compute_variance
    compute_variance(config['save_addon'], 'rule', rules, fast_eval=True)
    print('Computed variance')

    if 'allrule' in save_addon_type:
        from performance import compute_choicefamily_varytime
        for rule in [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]:
            compute_choicefamily_varytime(save_addon, rule)

if __name__ == '__main__':
    pass
    train(HDIM=37, s=0, save_addon='test', training_iters=300000, batch_size_train=50, batch_size_test=200, display_step=100)