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
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import seaborn.apionly as sns
from contextdm_data_analysis import condition_averaging_split_trte
from contextdm_data_analysis import get_cond_16_dim
from contextdm_data_analysis import get_condavg_simu_16_dim
from contextdm_data_analysis import run_simulation


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

    threshold = 0.8
    
    threshold = 0.9 #maddy changed. 
    
    n_implementable_train = np.sum(performance_train>threshold)
    n_implementable_test  = np.sum(performance_test >threshold)

    # Estimate total number of implementable classifications
    n_total_classification = 2**(n_condition-1)-1

    N_implementable_train = n_total_classification * (n_implementable_train/n_coloration)
    N_implementable_test  = n_total_classification * (n_implementable_test /n_coloration)
    print N_implementable_train, N_implementable_test
    
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


def _get_dimension(data_train, data_test, n_unit_used=None):
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


def get_dimension(data_train, data_test, n_unit_used=None, n_rep=1):
    N_implementable_trains = np.zeros(n_rep)
    N_implementable_tests = np.zeros(n_rep)
    for i_rep in range(n_rep):
        N_implementable_train, N_implementable_test = \
            _get_dimension(data_train, data_test, n_unit_used=n_unit_used)
        N_implementable_tests[i_rep] = N_implementable_test
        N_implementable_trains[i_rep]= N_implementable_train

    return np.log2(np.mean(N_implementable_trains)), np.log2(np.mean(N_implementable_tests))


def _get_dimension_16_dim(data_train, data_test, n_unit_used=None):
    # Temporarily excluding neurons because there are not enough trials
    n_unit = data_train.shape[1]
    ind_units = range(n_unit)
    exc_train_notactive = np.where(0==np.sum(data_train, axis=0))[0]#maddy added below. 
    exc_test_notactive = np.where(0==np.sum(data_test, axis=0))[0]
    excluding = np.concatenate([exc_train_notactive, exc_test_notactive])
    for exc in excluding:
        ind_units.pop(ind_units.index(exc))
        
    """
    excluding = np.where(np.isnan(np.sum(data_train, axis=0)+np.sum(data_test, axis=0)))[0]
    for exc in excluding:
        ind_units.pop(ind_units.index(exc))
    data_train, data_test = data_train[:, ind_units], data_test[:, ind_units]    
    """
    
    if n_unit_used == None:
        raise ValueError("specify no. of units used.")
    else:
        ind_used = np.arange(n_unit)
        np.random.shuffle(ind_used)
        ind_used = ind_used[:n_unit_used]
        data_train, data_test = data_train[:, ind_used], data_test[:, ind_used]

    N_implementable_train, N_implementable_test = \
        compute_implementable_classifier(data_train, data_test)

    # print(np.log2(N_implementable_train), np.log2(N_implementable_test))

    return N_implementable_train, N_implementable_test


def get_dimension_16_dim(data_train, data_test, n_unit_used=None, n_rep=1):
    N_implementable_trains = np.zeros(n_rep)
    N_implementable_tests = np.zeros(n_rep)
    for i_rep in range(n_rep):
        N_implementable_train, N_implementable_test = \
            _get_dimension_16_dim(data_train, data_test, n_unit_used=n_unit_used)
        N_implementable_tests[i_rep] = N_implementable_test
        N_implementable_trains[i_rep]= N_implementable_train

    print n_unit_used, np.log2(np.mean(N_implementable_trains)), np.log2(np.mean(N_implementable_tests))
    return np.log2(np.mean(N_implementable_trains)), np.log2(np.mean(N_implementable_tests))


def _get_dimension_varyusedunit(analyze_data=False, n_unit_used_list=None, **kwargs):
    if analyze_data:
        from mante_data_analysis import load_mante_data
        data = load_mante_data()
        data_train, data_test = get_trial_avg(data=data)
    else:
        from choiceattend_analysis import StateSpaceAnalysis
        # save_addon = 'allrule_softplus_400largeinput'
        save_addon = kwargs['save_addon']
        sigma_rec = 5 # chosen to reproduce the behavioral level performance
        ssa = StateSpaceAnalysis(save_addon, lesion_units=None,
                                 z_score=False, n_rep=30, sigma_rec=sigma_rec)

        data_train, data_test = get_trial_avg(
            analyze_data=False, task_var=ssa.task_var, data=ssa.H_original.mean(axis=0))
        
    if n_unit_used_list is None:
        n_unit_used_list = range(1, 400, 40)

    N_implementable_test_list = list()
    N_implementable_train_list = list()
    for n_unit_used in n_unit_used_list:
        start = time.time()
        N_implementable_train, N_implementable_test = \
            get_dimension(data_train, data_test, n_unit_used)

        N_implementable_test_list.append(N_implementable_test)
        N_implementable_train_list.append(N_implementable_train)

        print('Time taken {:0.3f}s'.format(time.time()-start))
        print(n_unit_used, np.log2(N_implementable_test))

    return n_unit_used_list, N_implementable_test_list

    # plt.plot(n_unit_used_list, np.log2(N_implementable_test_list))


def get_dimension_varyusedunit(analyze_data=False, n_rep=3, **kwargs):
    results = dict()
    if analyze_data:
        n_unit_used_list = range(1, 701, 50)
        save_name = 'dimension_data'
    else:
        n_unit_used_list = range(1, 300, 50)
        save_name = 'dimension_'+kwargs['save_addon']

    results['n_unit_used_list'] = n_unit_used_list
    N_implementable_test_matrix = list()
    for i_rep in range(n_rep):
        n_unit_used_list, N_implementable_test_list = \
            _get_dimension_varyusedunit(analyze_data, n_unit_used_list, **kwargs)
        N_implementable_test_matrix.append(N_implementable_test_list)

    results['N_implementable_test_matrix'] = N_implementable_test_matrix

    with open(os.path.join('data', save_name+'.pkl'), 'wb') as f:
        pickle.dump(results, f)


def _get_dimension_varyusedunit_16_dim(data_train, data_test, n_unit_used_list, n_rep):

    N_implementable_test_list = list()
    N_implementable_train_list = list()
    
    for n_unit_used in n_unit_used_list:
        
        start = time.time()
        N_implementable_train, N_implementable_test = \
            get_dimension_16_dim(data_train, data_test, n_unit_used,n_rep=n_rep)

        N_implementable_test_list.append(N_implementable_test)
        N_implementable_train_list.append(N_implementable_train)

        print('Time taken {:0.3f}s'.format(time.time()-start))

    return n_unit_used_list, N_implementable_test_list

#def get_dimension_varyusedunit_16_dim(data_train, data_test, n_rep=3):#maddy
def get_dimension_varyusedunit_16_dim(save_name, data_train, data_test, n_rep):#maddy
    results_new = dict() #maddy
    n_unit_used_list = range(1, 352, 50)#range(1, 702, 50)
    save_name = 'dimension_'+save_name

    results_new['n_unit_used_list'] = n_unit_used_list
    N_implementable_test_matrix = list()
    
    #for i_rep in range(n_rep):
    
    n_unit_used_list, N_implementable_test_list = \
        _get_dimension_varyusedunit_16_dim(data_train, data_test, n_unit_used_list, n_rep=n_rep)
    
    N_implementable_test_matrix.append(N_implementable_test_list)

    results_new['N_implementable_test_matrix'] = N_implementable_test_matrix

    filename = os.path.join('data', save_name+'.pkl')#maddy added. 
    if os.path.isfile(filename):
        with open(filename, 'rb') as resold:
            results = pickle.load(resold)
            results['N_implementable_test_matrix'].append(N_implementable_test_matrix)
        with open(os.path.join('data', save_name+'.pkl'), 'wb') as f:
            pickle.dump(results, f)
            print "results", results
    else:
        with open(os.path.join('data', save_name+'.pkl'), 'wb') as f:
            pickle.dump(results_new, f)
            print "results_new", results_new
    #with open(os.path.join('data', save_name+'.pkl'), 'wb') as f:
    #    pickle.dump(results, f)

def call_get_dimension_varyusedunit_16_dim(save_name = 'debug', n_rep=10):#10. #maddy
    
    #fname = os.path.join('data', 'config_' + save_name + '.pkl') 

    #if save_name == 'Data':
    #    fname = os.path.join('data', 'ManteData.pkl') #'ManteDataCond.pkl'
    
    #if os.path.isfile(fname):
    #    with open(fname, 'rb') as f:
    #        Data = pickle.load(f) 

    train_set, test_set = get_condavg_simu_16_dim(save_name)

    #ind_time = 14##maddy changed. 
    for ind_time in np.arange(0,15):#15
        print "ind_time", ind_time
        train_timept, test_timept = train_set[ind_time], test_set[ind_time]
        get_dimension_varyusedunit_16_dim(save_name, train_timept, test_timept, n_rep=n_rep)


def plot_dimension_estimation(save_addon_list):
    import scipy as sp
    import scipy.stats

    def mean_confidence_interval(data, confidence=0.95, **kwargs):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a, **kwargs), scipy.stats.sem(a, **kwargs)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m, m-h, m+h

    # colors = dict(zip(['data','model'], sns.xkcd_palette(['black', 'red'])))
    plt.figure()
    for i_plt, save_addon in enumerate(save_addon_list):
    # save_name = 'dimension_data'
        with open(os.path.join('data', 'dimension'+save_addon+'.pkl'), 'rb') as f:
            results = pickle.load(f)
        N_implementable_test_matrix = np.array(results['N_implementable_test_matrix'])
        n_unit_used_list = results['n_unit_used_list']

        mean_val, lower_bnds, higher_bnds = mean_confidence_interval(N_implementable_test_matrix, axis=0)
        print(mean_val[1])
        plt.plot(n_unit_used_list, np.log2(mean_val), color=plt.cm.cool(1.0*i_plt/len(save_addon_list)))
        # plt.fill_between(n_unit_used_list, np.log2(lower_bnds), np.log2(higher_bnds),
        #                  alpha=0.3, color=color)

    plt.plot(n_unit_used_list, [15]*len(n_unit_used_list), '--', color='black')

    # plt.legend(loc=4)
    plt.xlabel('number of neurons or units')
    plt.ylabel('number of dimensions')
    # plt.savefig(os.path.join('figure','dimension_estimation.pdf'), transparent=True)


if __name__ == '__main__':
    pass

    save_name = 'hidden_64_seed_2_softplus_LeakyRNN_diag__regwt_L1_1e_min_4_regact_L1_1e_min_4'
    #save_name = 'Data'
    call_get_dimension_varyusedunit_16_dim(save_name=save_name)

    print "here"

