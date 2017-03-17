"""
Analyze dimennsionality by counting implementable linear classifiers
Rigotti et al 2013 Nature

@ Robert Yang 2017
"""

from __future__ import division

import os
import time
import numpy as np
import pickle
import itertools
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def expand_task_var(task_var):
    # little helper function that calculate a few more things
    task_var['stim_dir_sign'] = (task_var['stim_dir']>0).astype(int)*2-1
    task_var['stim_col2dir_sign'] = (task_var['stim_col2dir']>0).astype(int)*2-1
    return task_var


def get_trial_avg_traintest(analyze_data=True, **kwargs):
    # get trial-averaged rate for each condition
    # For now: 6 * 6 * 2 = 72 conditions in total
    # Split training and testing as well

    # var_keys = ['stim_dir', 'stim_col2dir', 'context']
    # var_keys = ['targ_dir', 'stim_dir', 'stim_col2dir', 'context']
    # var_keys = ['stim_dir_sign', 'stim_col2dir_sign', 'context']
    var_keys = ['targ_dir', 'stim_dir_sign', 'stim_col2dir_sign', 'context']

    # number of variables
    n_var = len(var_keys)

    # initialize
    if analyze_data:
        data = kwargs['data']
        task_var = data[0].task_variable.__dict__ # turn into dictionary
        n_unit = len(data)
    else:
        task_var = kwargs['task_var']
        n_unit = kwargs['data'].shape[1]

    task_var = expand_task_var(task_var)
    n_cond = np.prod([len(np.unique(task_var[k])) for k in var_keys])

    p_train = 0.7 # proportion of training data
    data_train = np.zeros((n_cond, n_unit))
    data_test  = np.zeros((n_cond, n_unit))


    for i_unit in range(n_unit):
        if analyze_data:
            task_var = data[i_unit].task_variable.__dict__ # turn into dictionary
        task_var = expand_task_var(task_var)

        # number of trials
        n_trial = len(task_var[var_keys[0]])

        # dictionary of unique task variable values
        var_unique = [np.unique(task_var[k]) for k in var_keys]

        # list of number of unique task variable values
        n_var_unique = [len(v) for v in var_unique]

        # number of condition
        n_cond = np.prod(n_var_unique)

        # List of indices for each task variable
        ind_var_conds = np.unravel_index(range(n_cond),n_var_unique)

        if analyze_data:
            # Responses of this unit
            response = data[i_unit].response # (trial, time)
            # Average over time, temporary
            response = response.mean(axis=1)
        else:
            response = kwargs['data'][:, i_unit] # data has to be (n_trial, n_unit)

        for i_cond in range(n_cond):
            ind_cond = np.ones(n_trial, dtype=bool)
            for i_var in range(n_var):
                ind_var_cond = ind_var_conds[i_var][i_cond]
                ind_cond_tmp = task_var[var_keys[i_var]]==var_unique[i_var][ind_var_cond]
                ind_cond *= ind_cond_tmp
            # Turn into actual indices
            ind_cond = np.where(ind_cond)[0]
            # randomly shuffle
            np.random.shuffle(ind_cond)
            # split into training and testing ones
            n_trial_cond = len(ind_cond)
            n_trial_cond_train = int(n_trial_cond*p_train)

            ind_cond_train = ind_cond[:n_trial_cond_train]
            ind_cond_test  = ind_cond[n_trial_cond_train:]

            # if n_trial_cond_train == 0:
            #     print(i_unit, i_cond, n_trial_cond)

            data_train[i_cond, i_unit] = np.mean(response[ind_cond_train])
            data_test[i_cond, i_unit] = np.mean(response[ind_cond_test])


    return data_train, data_test


def compute_implementable_classifier(data_train, data_test):
    # data_train & data_test should be (n_condition, n_unit)

    # number of conditions
    n_condition = data_train.shape[0]

    # classification
    classifier = SVC(kernel='linear', C=1.0)
    # classifier = LinearDiscriminantAnalysis() # much slower

    n_coloration = 2**(n_condition-1)-1
    if n_coloration > 10**6:
        raise ValueError('too many colorations')

    performance_train = list()
    performance_test  = list()
    colors_list = list()
    for i, colors_ in enumerate(itertools.product([0,1], repeat=n_condition-1)):
        if i == 0:
            # Don't use [0, 0, ..., 0]
            continue

        colors = np.array([0]+list(colors_)) # the first one is always zero to break symmetry

        # Fit
        classifier.fit(data_train, colors)

        color_train_predict = classifier.predict(data_train)
        color_test_predict  = classifier.predict(data_test)

        performance_train.append(np.mean(colors==color_train_predict))
        performance_test.append(np.mean(colors==color_test_predict))
        colors_list.append(colors)

    performance_train = np.array(performance_train)
    performance_test = np.array(performance_test)

    # finish = time.time()
    # print('Time taken {:0.5f}s'.format(finish-start))

    threshold = 0.9
    n_implementable_train = np.sum(performance_train>threshold)
    n_implementable_test  = np.sum(performance_test >threshold)

    # Estimate total number of implementable classifications
    n_total_classification = 2**(n_condition-1)-1

    N_implementable_train = n_total_classification * (n_implementable_train/n_coloration)
    N_implementable_test  = n_total_classification * (n_implementable_test /n_coloration)


    return N_implementable_train, N_implementable_test

    # print(np.log2(N_implementable_train), np.log2(N_implementable_test))


def generate_test_data():
    # generate some data (n_batch can be the same or much larger than n_condition)
    n_time, n_batch, n_unit = 10, 100, 100

    data = np.random.rand(n_time, n_batch, n_unit)

    # trial condition
    n_condition = 72
    conditions = np.array(range(n_condition)*int(n_batch/n_condition))

    # TODO: Splitting training and testing data
    # For now assume the same, and assume batch=condition
    data_train = np.random.rand(n_time, n_condition, n_unit)
    data_test  = np.random.randn(n_time, n_condition, n_unit)*0.1 + data_train

    # pick one data point
    i_t = 0
    data_train_t = data_train[i_t] # (n_batch, n_unit)
    data_test_t  = data_test[i_t] # (n_batch, n_unit)

    return data_train_t, data_test_t


def get_dimension(data_train, data_test, n_unit_used=None):

    # Temporarily excluding neurons because there are not enough trials
    n_unit = data_train.shape[1]
    ind_units = range(n_unit)
    excluding = np.where(np.isnan(np.sum(data_train, axis=0)+np.sum(data_test, axis=0)))[0]
    for exc in excluding:
        ind_units.pop(ind_units.index(exc))
    data_train, data_test = data_train[:, ind_units], data_test[:, ind_units]
    n_unit = data_train.shape[1]

    if n_unit_used is not None:
        ind_used = np.arange(n_unit)
        np.random.shuffle(ind_used)
        ind_used = ind_used[:n_unit_used]
        data_train, data_test = data_train[:, ind_used], data_test[:, ind_used]

    N_implementable_train, N_implementable_test = \
        compute_implementable_classifier(data_train, data_test)

    # print(np.log2(N_implementable_train), np.log2(N_implementable_test))

    return N_implementable_train, N_implementable_test

analyze_data = False
if analyze_data:
    from mante_data_analysis import load_mante_data
    data = load_mante_data()
    data_train, data_test = get_trial_avg_traintest(data=data)
else:
    from choiceattend_analysis import StateSpaceAnalysis
    save_addon = 'allrule_softplus_400largeinput'
    sigma_rec = 0.5 # chosen to reproduce the behavioral level performance
    ssa = StateSpaceAnalysis(save_addon, lesion_units=None,
                             z_score=False, n_rep=30, sigma_rec=sigma_rec)

    data_train, data_test = get_trial_avg_traintest(
        analyze_data=False, task_var=ssa.task_var, data=ssa.H_original.mean(axis=0))

n_unit_used_list = range(1, 400, 40)
N_implementable_test_list = list()
for n_unit_used in n_unit_used_list:
    start = time.time()
    N_implementable_train, N_implementable_test = \
        get_dimension(data_train, data_test, n_unit_used)

    N_implementable_test_list.append(N_implementable_test)
    print('Time taken {:0.3f}s'.format(time.time()-start))
    print(n_unit_used, np.log2(N_implementable_test))

plt.plot(n_unit_used_list, np.log2(N_implementable_test_list))