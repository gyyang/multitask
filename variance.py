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

save = True

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
        t_start = int(500/config['dt']) # Important: Ignore the initial transition
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

def compute_hist_varprop(save_type, rules):
    data_type = 'rule'
    assert len(rules) == 2
    assert data_type == 'rule'

    HDIMs = range(200, 1000)
    hists = list()
    hdims = list()
    for HDIM in HDIMs:
        save_addon = save_type+'_'+str(HDIM)
        fname = 'data/variance'+data_type+save_addon
        fname += '.pkl'
        if not os.path.isfile(fname):
            continue

        # If not computed, use variance.py
        # fname = 'data/variance'+data_type+save_addon+'_rr'
        # fname = 'data/variance'+data_type+save_addon
        with open(fname,'rb') as f:
            res = pickle.load(f)
        h_var_all = res['h_var_all']
        keys      = res['keys']

        ind_rules = [keys.index(rule) for rule in rules]
        h_var_all = h_var_all[:, ind_rules]

        # First only get active units. Total variance across tasks larger than 1e-3
        ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]

        # Temporary: Mimicking biased sampling. Notice the free parameter though.
        # print('Mimicking selective sampling')
        # ind_active = np.where((h_var_all.sum(axis=1) > 1e-3)*(h_var_all[:,0]>1*1e-2))[0]

        h_var_all  = h_var_all[ind_active, :]

        # Normalize by the total variance across tasks
        h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

        # Plot the proportion of variance for the first rule
        data_plot = h_normvar_all[:, 0]
        hist, bins_edge = np.histogram(data_plot, bins=20, range=(0,1))

        # Plot the percentage instead of the total count
        hist = hist/np.sum(hist)

        # Store
        hists.append(hist)
        hdims.append(HDIM)

    # Get median of all histogram
    hists = np.array(hists)
    # hist_low, hist_med, hist_high = np.percentile(hists, [10, 50, 90], axis=0)

    return hists, bins_edge, hdims

def plot_hist_varprop(save_type, rules, hdim_example=None):
    '''
    Plot histogram of proportion of variance for some tasks across units
    :param save_addon:
    :param data_type:
    :param rules: list of rules. Show proportion of variance for the first rule
    :return:
    '''

    hists, bins_edge, hdims = compute_hist_varprop(save_type, rules)

    hist_low, hist_med, hist_high = np.percentile(hists, [10, 50, 90], axis=0)

    # hist_med, bins_edge = np.histogram(data_plots, bins=20, range=(0,1))
    # hist_med = np.array(hist_med)/len(hdims)

    fs = 6
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.2,0.3,0.6,0.5])
    if hdim_example is not None:
        hist_example = hists[hdims.index(hdim_example)]
        ax.bar(bins_edge[:-1], hist_example, width=bins_edge[1]-bins_edge[0],
               color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
    ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_med, color='black', linewidth=1.5)
    # ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_low)
    # ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_high)
    plt.locator_params(nbins=3)
    xlabel = 'VarRatio({:s}, {:s})'.format(rule_name[rules[0]], rule_name[rules[1]])
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylim(bottom=-0.02*hist_med.max())
    ax.set_xlim([-0.1,1.1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
    if save:
        plt.savefig('figure/plot_hist_varprop'+
                rule_name[rules[0]].replace(' ','')+
                rule_name[rules[1]].replace(' ','')+
                save_type+'.pdf', transparent=True)

def plot_hist_varprop_selection(save_type, **kwargs):
    rules_list = [(CHOICE_MOD1, CHOICE_MOD2),
                  (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2),
                  (CHOICE_MOD1, REMAP),
                  (DELAYMATCHGO, DMCGO),
                  (INHGO, GO)]
    for rules in rules_list:
        plot_hist_varprop(save_type=save_type, rules=rules, **kwargs)

def plot_hist_varprop_all(save_type, hdim_example=None):
    '''
    Plot histogram of proportion of variance for some tasks across units
    :param save_addon:
    :param data_type:
    :param rules: list of rules. Show proportion of variance for the first rule
    :return:
    '''
    data_type = 'rule'
    rules = [GO, INHGO, DELAYGO,\
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
        REMAP, INHREMAP, DELAYREMAP,\
        DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]
    figsize = (8, 8)

    # rules = [GO, INHGO, DELAYGO]
    # figsize = (4, 4)

    fs = 6 # fontsize

    f, axarr = plt.subplots(len(rules), len(rules), figsize=figsize)
    for i in range(len(rules)):
        for j in range(len(rules)):
            ax = axarr[i, j]
            if i == 0:
                ax.set_title(rule_name[rules[j]], fontsize=fs, rotation=45, va='bottom')
            if j == 0:
                ax.set_ylabel(rule_name[rules[i]], fontsize=fs, rotation=45, ha='right')

            # if j == i:
            #     ax.axis('off')
            #     continue

            hists, bins_edge, hdims = compute_hist_varprop(save_type, (rules[i], rules[j]))

            hist_low, hist_med, hist_high = np.percentile(hists, [10, 50, 90], axis=0)

            if hdim_example is not None:
                hist_example = hists[hdims.index(hdim_example)]
                ax.bar(bins_edge[:-1], hist_example, width=bins_edge[1]-bins_edge[0],
                       color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')

            ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_med, color='black')
            plt.locator_params(nbins=3)
            # xlabel = r'$\frac{Var({:s})}{[Var({:s})+Var({:s})]}$'.format(rule_name[rules[0]],rule_name[rules[0]],rule_name[rules[1]])
            xlabel = 'VarRatio({:s},{:s})'.format(rule_name[rules[0]], rule_name[rules[1]])
            # ax.set_xlabel(xlabel, fontsize=fs)
            ax.set_ylim(bottom=-0.02*hist_med.max())
            ax.set_xticks([0,1])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xlim([0,1])
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)

    # plt.tight_layout()
    if save:
        plt.savefig('figure/plot_hist_varprop_all'+save_type+'.pdf', transparent=True)

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
    save_addon = 'allrule_weaknoise_400'
    data_type = 'rule'

    rules = [GO, INHGO, DELAYGO,\
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
        REMAP, INHREMAP, DELAYREMAP,\
        DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]


    # compute_variance(save_addon, data_type, rules, random_rotation=True, fast_eval=True)
    # with open('data/variance'+data_type+save_addon+'_rr'+'.pkl','rb') as f:
    #     res = pickle.load(f)
    # h_var_all = res['h_var_all']
    # # First only get active units. Total variance across tasks larger than 1e-3
    # # ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
    # # h_var_all  = h_var_all[ind_active, :]
    #
    # h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T


    save_type = 'allrule_weaknoise'
    # plot_hist_varprop(save_type, rules=[DELAYMATCHGO, DMCGO], hdim_example=400)
    # plot_hist_varprop_all('allrule_weaknoise')
    # plot_hist_varprop_selection(save_type, hdim_example=400)