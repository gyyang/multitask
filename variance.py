"""
Compute Variance
"""

from __future__ import division

import os
import time
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from task import *
from run import Run

def compute_variance(save_addon, data_type, rules):
    ########################## Running the network ################################
    n_rules = len(rules)

    h_all_byrule  = OrderedDict()
    h_all_byepoch = OrderedDict()
    with Run(save_addon, sigma_rec=0) as R:
        config = R.config
        for rule in rules:
            task = generate_onebatch(rule=rule, config=config, mode='test')
            h_all_byrule[rule] = R.f_h(task.x)
            for e_name, e_time in task.epochs.iteritems():
                if 'fix' not in e_name:
                    h_all_byepoch[(rule, e_name)] = h_all_byrule[rule][e_time[0]:e_time[1],:,:]

    w_in  = R.w_in # for later sorting
    w_out = R.w_out

    # Reorder h_all_byepoch by epoch-first
    keys = h_all_byepoch.keys()
    ind_key_sort = np.lexsort(zip(*keys))
    h_all_byepoch = OrderedDict([(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

    nx, nh, ny = config['shape']
    n_ring = config['N_RING']

    if data_type == 'rule':
        h_all = h_all_byrule
        t_start = 500 # Important: Ignore the initial transition
    elif data_type == 'epoch':
        h_all = h_all_byepoch
        t_start = None
    else:
        raise ValueError

    h_var_all = np.zeros((nh, len(h_all.keys())))
    for i, val in enumerate(h_all.values()):
        # val is Time, Batch, Units
        # Variance across time and stimulus
        # h_var_all[:, i] = val[t_start:].reshape((-1, nh)).var(axis=0)
        # Variance acros stimulus, then averaged across time
        h_var_all[:, i] = val[t_start:].var(axis=1).mean(axis=0)

    result = {'h_var_all' : h_var_all, 'keys' : h_all.keys()}
    with open('data/variance'+data_type+save_addon+'.pkl','wb') as f:
        pickle.dump(result, f)


def plot_variance(save_addon, data_type):
    with open('data/variance'+data_type+save_addon+'.pkl','rb') as f:
        res = pickle.load(f)
    # Plot total variance distribution
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.3,0.3,0.6,0.5])
    hist, bins_edge = np.histogram(np.log10(res['h_var_all'].sum(axis=1)), bins=30)
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
           color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
    plt.xlabel(r'$log_{10}$ total variance', fontsize=7, labelpad=1)
    plt.ylabel('counts', fontsize=7)
    plt.locator_params(nbins=3)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('figure/hist_totalvar.pdf', transparent=True)
    plt.show()


if __name__ == '__main__':
    # save_addon = 'tf_withrecnoise_500'
    save_addon = 'tf_attendonly_150'
    data_type = 'rule'

    # rules = [GO, INHGO, DELAYGO,\
    #     CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    #     CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,\
    #     REMAP, INHREMAP, DELAYREMAP,\
    #     DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]

    rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

    start = time.time()
    compute_variance(save_addon, data_type, rules)
    print('Time taken {:0.2f} s'.format(time.time()-start))