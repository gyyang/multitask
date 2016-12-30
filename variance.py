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

def compute_variance(save_addon, data_type, rules,
                     random_rotation=False, fast_eval=False):
    ########################## Running the network ################################
    n_rules = len(rules)

    h_all_byrule  = OrderedDict()
    h_all_byepoch = OrderedDict()
    with Run(save_addon, sigma_rec=0, fast_eval=fast_eval) as R:
        config = R.config
        nx, nh, ny = config['shape']

        if random_rotation:
            # Generate random orthogonal matrix
            from scipy.stats import ortho_group
            random_ortho_matrix = ortho_group.rvs(dim=nh)

        for rule in rules:
            task = generate_onebatch(rule=rule, config=config, mode='test')
            h = R.f_h(task.x)
            if random_rotation:
                h = np.dot(h, random_ortho_matrix) # randomly rotate
            h_all_byrule[rule] = h
            for e_name, e_time in task.epochs.iteritems():
                if 'fix' not in e_name:
                    h_all_byepoch[(rule, e_name)] = h_all_byrule[rule][e_time[0]:e_time[1],:,:]

    w_in  = R.w_in # for later sorting
    w_out = R.w_out

    # Reorder h_all_byepoch by epoch-first
    keys = h_all_byepoch.keys()
    ind_key_sort = np.lexsort(zip(*keys))
    h_all_byepoch = OrderedDict([(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

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
    if random_rotation:
        save_addon += '_rr'

    with open('data/variance'+data_type+save_addon+'.pkl','wb') as f:
        pickle.dump(result, f)


def plot_hist_totalvar(save_addon, data_type):
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
    # plt.savefig('figure/hist_totalvar.pdf', transparent=True)
    plt.show()



def get_random_rotation_variance(save_addon, data_type): ##TODO: Need more work
    # save_addon = 'allrule_weaknoise_300'
    # data_type = 'rule'

    # If not computed, use variance.py
    # fname = 'data/variance'+data_type+save_addon+'_rr'
    fname = 'data/variance'+data_type+save_addon
    with open(fname+'.pkl','rb') as f:
        res = pickle.load(f)
    h_var_all = res['h_var_all']
    keys      = res['keys']

    # First only get active units. Total variance across tasks larger than 1e-3
    ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
    h_var_all  = h_var_all[ind_active, :]

    # Normalize by the total variance across tasks
    h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T


    rule_hist = CHOICEATTEND_MOD1
    data_plot = h_normvar_all[:, keys.index(rule_hist)]

    p_low, p_high = 2.5, 97.5
    normvar_low, normvar_high = np.percentile(data_plot, [p_low, p_high])

    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.3,0.3,0.6,0.5])
    hist, bins_edge = np.histogram(data_plot, bins=30)
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
           color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
    # ax.set_xlim([0,0.3])
    ax.plot([normvar_low]*2,  [0, hist.max()], 'black')
    ax.plot([normvar_high]*2, [0, hist.max()], 'black')
    plt.locator_params(nbins=3)
    ax.set_ylim(bottom=-1)

    print('{:0.1f} percentile: {:0.2f}'.format(p_low,  normvar_low))
    print('{:0.1f} percentile: {:0.2f}'.format(p_high, normvar_high))


if __name__ == '__main__':
    save_addon = 'allrule_weaknoise_300'
    data_type = 'rule'

    rules = [GO, INHGO, DELAYGO,\
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
        REMAP, INHREMAP, DELAYREMAP,\
        DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]

    # rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    # rules = [DELAYMATCHGO, DMCGO]
    # rules = [CHOICE_MOD1, CHOICE_MOD2]
    
    # start = time.time()
    # compute_variance(save_addon, data_type, rules, random_rotation=True)
    # plot_hist_totalvar(save_addon, data_type)
    # print('Time taken {:0.2f} s'.format(time.time()-start))
    get_random_rotation_variance(save_addon, data_type)


    # rule_hist = CHOICEATTEND_MOD1
    # fig = plt.figure(figsize=(1.5,1.2))
    # ax = fig.add_axes([0.3,0.3,0.6,0.5])
    # hist, bins_edge = np.histogram(h_normvar_all[:, keys.index(rule_hist)], bins=30, range=(0,1))
    # ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
    #        color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([hist.max()*-0.05, hist.max()*1.1])
    # plt.xlabel(r'Variance ratio', fontsize=7, labelpad=1)
    # plt.ylabel('counts', fontsize=7)
    # plt.title(rule_name[rule_hist], fontsize=7)
    # plt.locator_params(nbins=3)
    # ax.tick_params(axis='both', which='major', labelsize=7)
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    # # plt.savefig('figure/hist_totalvar.pdf', transparent=True)
    # plt.show()

