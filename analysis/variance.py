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
import tensorflow as tf

from task import *
from network import Model
import tools

save = True


def _compute_variance_bymodel(model, sess, rules=None, random_rotation=False):
    """Compute variance for all tasks.

        Args:
            model: network.Model instance
            sess: tensorflow session
            rules: list of rules to compute variance, list of strings
            random_rotation: boolean. If True, rotate the neural activity.
        """
    h_all_byrule = OrderedDict()
    h_all_byepoch = OrderedDict()
    hp = model.hp

    if rules is None:
        rules = hp['rules']
    print(rules)

    n_hidden = hp['n_rnn']

    if random_rotation:
        # Generate random orthogonal matrix
        from scipy.stats import ortho_group
        random_ortho_matrix = ortho_group.rvs(dim=n_hidden)

    for rule in rules:
        trial = generate_trials(rule, hp, 'test', noise_on=False)
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        h = sess.run(model.h, feed_dict=feed_dict)
        if random_rotation:
            h = np.dot(h, random_ortho_matrix)  # randomly rotate

        for e_name, e_time in trial.epochs.items():
            if 'fix' not in e_name:  # Ignore fixation period
                h_all_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1], :,
                                                :]

        # Ignore fixation period
        h_all_byrule[rule] = h[trial.epochs['fix1'][1]:, :, :]

    # Reorder h_all_byepoch by epoch-first
    keys = list(h_all_byepoch.keys())
    # ind_key_sort = np.lexsort(zip(*keys))
    # Using mergesort because it is stable
    ind_key_sort = np.argsort(list(zip(*keys))[1], kind='mergesort')
    h_all_byepoch = OrderedDict(
        [(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

    for data_type in ['rule', 'epoch']:
        if data_type == 'rule':
            h_all = h_all_byrule
        elif data_type == 'epoch':
            h_all = h_all_byepoch
        else:
            raise ValueError

        h_var_all = np.zeros((n_hidden, len(h_all.keys())))
        for i, val in enumerate(h_all.values()):
            # val is Time, Batch, Units
            # Variance across time and stimulus
            # h_var_all[:, i] = val[t_start:].reshape((-1, n_hidden)).var(axis=0)
            # Variance acros stimulus, then averaged across time
            h_var_all[:, i] = val.var(axis=1).mean(axis=0)

        result = {'h_var_all': h_var_all, 'keys': list(h_all.keys())}
        save_name = 'variance_' + data_type
        if random_rotation:
            save_name += '_rr'

        fname = os.path.join(model.model_dir, save_name + '.pkl')
        print('Variance saved at {:s}'.format(fname))
        with open(fname, 'wb') as f:
            pickle.dump(result, f)


def _compute_variance(model_dir, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    model = Model(model_dir, sigma_rec=0)
    with tf.Session() as sess:
        model.restore()
        _compute_variance_bymodel(model, sess, rules, random_rotation)


def compute_variance(model_dir, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    dirs = tools.valid_model_dirs(model_dir)
    for d in dirs:
        _compute_variance(d, rules, random_rotation)


def _compute_hist_varprop(model_dir, rule_pair, random_rotation=False):
    data_type = 'rule'
    assert len(rule_pair) == 2
    assert data_type == 'rule'

    fname = os.path.join(model_dir, 'variance_'+data_type)
    if random_rotation:
        fname += '_rr'
    fname += '.pkl'
    if not os.path.isfile(fname):
        # If not computed, compute now
        compute_variance(model_dir, random_rotation=random_rotation)

    res = tools.load_pickle(fname)
    h_var_all = res['h_var_all']
    keys      = res['keys']

    ind_rules = [keys.index(rule) for rule in rule_pair]
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
    # data_plot = h_normvar_all[:, 0]
    data_plot = (h_var_all[:, 0]-h_var_all[:, 1])/((h_var_all[:, 0]+h_var_all[:, 1]))
    hist, bins_edge = np.histogram(data_plot, bins=20, range=(-1,1))

    # # Plot the percentage instead of the total count
    # hist = hist/np.sum(hist)

    return hist, bins_edge


def compute_hist_varprop(model_dir, rule_pair, random_rotation=False):
    data_type = 'rule'
    assert len(rule_pair) == 2
    assert data_type == 'rule'

    model_dirs = tools.valid_model_dirs(model_dir)

    hists = list()
    for model_dir in model_dirs:
        hist, bins_edge_ = _compute_hist_varprop(model_dir, rule_pair, random_rotation)
        if hist is None:
            continue
        else:
            bins_edge = bins_edge_

        # Store
        hists.append(hist)

    # Get median of all histogram
    hists = np.array(hists)
    # hist_low, hist_med, hist_high = np.percentile(hists, [10, 50, 90], axis=0)

    return hists, bins_edge

def _plot_hist_varprop(hist_plot, bins_edge, rule_pair, hist_example=None,
                       plot_legend=False, figname=None, title=None):
    '''Plot histogram of fractional variance'''
    # Plot the percentage instead of the total count
    hist_plot = hist_plot/np.sum(hist_plot)
    if hist_example is not None:
        hist_example = hist_example/np.sum(hist_example)

    fs = 6
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.2,0.3,0.6,0.5])
    legends = list()
    labels = list()
    if hist_example is not None:
        pass
        bar = ax.bar(bins_edge[:-1], hist_example, width=bins_edge[1]-bins_edge[0],
               color='xkcd:cerulean', edgecolor='none')
        legends.append(bar)
        labels.append('Example network')
    pl, = ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_plot, color='black', linewidth=1.5)
    legends.append(pl)
    labels.append('All networks')
    # ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_low)
    # ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_high)
    plt.locator_params(nbins=3)
    xlabel = 'FTV({:s}, {:s})'.format(rule_name[rule_pair[0]], rule_name[rule_pair[1]])
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylim(bottom=-0.02*hist_plot.max())
    ax.set_xlim([-1.1,1.1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
    if title:
        ax.set_title(title, fontsize=7)
    if plot_legend:
        lg = plt.legend(legends, labels, ncol=1,bbox_to_anchor=(1.1,1.3),
                        fontsize=fs,labelspacing=0.3,loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
    if save:
        if figname is None:
            figname = 'plot_hist_varprop_tmp.pdf'
        plt.savefig(os.path.join('figure', figname), transparent=True)


def plot_hist_varprop(model_dir,
                      rule_pair,
                      plot_example=False,
                      figname_extra=None,
                      **kwargs):
    """
    Plot histogram of proportion of variance for some tasks across units

    Args:
        model_dir: model directory
        rule_pair: tuple of strings, pair of rules
        plot_example: bool
        figname_extra: string or None
    """

    hists, bins_edge = compute_hist_varprop(model_dir, rule_pair)

    hist_low, hist_med, hist_high = np.percentile(hists, [10, 50, 90], axis=0)

    # hist_med, bins_edge = np.histogram(data_plots, bins=20, range=(0,1))
    # hist_med = np.array(hist_med)/len(hdims)

    if plot_example:
        hist_example = hists[0]
    else:
        hist_example = None

    hist_plot = hist_med
    figname = 'plot_hist_varprop' + rule_pair[0] + rule_pair[1]
    figname = figname.replace('*','')
    if figname_extra:
        figname += figname_extra
    _plot_hist_varprop(hist_plot, bins_edge, rule_pair=rule_pair,
                       hist_example=hist_example, figname=figname+'.pdf',
                       **kwargs)


def plot_hist_varprop_selection(model_dir, figname_extra=None):
    rule_pair_list = [('dm1', 'dm2'),
                  ('contextdm1', 'contextdm2'),
                  ('dm1', 'fdanti'),
                  ('dm1', 'contextdm1'),
                  ('fdgo', 'reactgo'),
                  ('delaydm1', 'dm1'),
                  ('dmcgo', 'dmcnogo'),
                  ('contextdm1', 'contextdelaydm1')]
    for rule_pair in rule_pair_list:
        plot_hist_varprop(model_dir=model_dir,
                          rule_pair=rule_pair,
                          plot_legend=(rule_pair==('dm1', 'fdanti')),
                          plot_example=True,
                          figname_extra=figname_extra)


def plot_hist_varprop_all(model_dir, plot_control=True):
    '''
    Plot histogram of proportion of variance for some tasks across units
    :param save_name:
    :param data_type:
    :param rule_pair: list of rule_pair. Show proportion of variance for the first rule
    :return:
    '''

    model_dirs = tools.valid_model_dirs(model_dir)

    hp = tools.load_hp(model_dirs[0])
    rules = hp['rules']

    figsize = (7, 7)

    # For testing
    # rules, figsize = ['fdgo','reactgo','delaygo', 'fdanti', 'reactanti'], (4, 4)

    fs = 6 # fontsize

    f, axarr = plt.subplots(len(rules), len(rules), figsize=figsize)
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.02, top=0.9)

    for i in range(len(rules)):
        for j in range(len(rules)):
            ax = axarr[i, j]
            if i == 0:
                ax.set_title(rule_name[rules[j]], fontsize=fs, rotation=45, va='bottom')
            if j == 0:
                ax.set_ylabel(rule_name[rules[i]], fontsize=fs, rotation=45, ha='right')
            
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            if i == j:
                ax.spines["bottom"].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            hists, bins_edge = compute_hist_varprop(model_dir, (rules[i], rules[j]))
            hist_low, hist_med, hist_high = np.percentile(hists, [10, 50, 90], axis=0)
            hist_med /= hist_med.sum()

            # Control case
            if plot_control:
                hists_ctrl, _ = compute_hist_varprop(model_dir, (rules[i], rules[j]), random_rotation=True)
                _, hist_med_ctrl, _ = np.percentile(hists_ctrl, [10, 50, 90], axis=0)
                hist_med_ctrl /= hist_med_ctrl.sum()
                ax.plot((bins_edge[:-1]+bins_edge[1:])/2,
                        hist_med_ctrl, color='gray', lw=0.75)

            ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist_med, color='black')
            plt.locator_params(nbins=3)

            # ax.set_ylim(bottom=-0.02*hist_med.max())
            ax.set_ylim([-0.01, 0.6])
            print(hist_med.max())
            ax.set_xticks([-1,1])
            ax.set_xticklabels([])
            if i == 0 and j == 1:
                ax.set_yticks([0, 0.6])
                ax.spines["left"].set_visible(True)
            else:
                ax.set_yticks([])
            ax.set_xlim([-1,1])
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
            

    # plt.tight_layout()
    plt.savefig('figure/plot_hist_varprop_all.pdf', transparent=True)

def plot_hist_varprop_selection_cont():
        save_type = 'cont_allrule'
        save_type_end = '_0_1_2intsynmain'
        rules_list = [(CHOICE_MOD1, CHOICE_MOD2),
                      (CHOICEATTEND_MOD1, CHOICEATTEND_MOD2),
                      (CHOICE_MOD1, CHOICEATTEND_MOD1),
                      (CHOICEDELAY_MOD1, CHOICE_MOD1)]
        for rules in rules_list:
            plot_hist_varprop(save_type=save_type, save_type_end=save_type_end, rules=rules,
                              plot_legend=(rules==(CHOICE_MOD1, REACTANTI)), hdim_example=0)

def get_random_rotation_variance(save_name, data_type): ##TODO: Need more work
    # save_name = 'allrule_weaknoise_300'
    # data_type = 'rule'

    # If not computed, use variance.py
    # fname = 'data/variance'+data_type+save_name+'_rr'
    fname = os.path.join('data', 'variance_'+data_type+save_name)
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


def compute_ntasks_selective():
    # Compute the number of tasks each neuron is selective for
    # NOT WELL DEFINED YET
    # DOESN"T REALLY WORK
    with open(os.path.join('data','variance'+data_type+save_name+'_rr'+'.pkl'),'rb') as f:
        res_rr = pickle.load(f)
    h_var_all_rr = res_rr['h_var_all']

    bounds = np.percentile(h_var_all_rr, 97.5, axis=0)

    # bounds = 1e-2

    with open(os.path.join('data','variance'+data_type+save_name+'.pkl'),'rb') as f:
        res = pickle.load(f)
    h_var_all = res['h_var_all']

    # First only get active units. Total variance across tasks larger than 1e-3
    ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
    h_var_all  = h_var_all[ind_active, :]

    h_selective = h_var_all > bounds
    n_selective = h_selective.sum(axis=1)

    hist, bins_edge = np.histogram(n_selective, bins=lren(rules)+1, range=(-0.5,len(rules)+0.5))

    fig = plt.figure(figsize=(3,2.4))
    ax = fig.add_axes([0.2,0.3,0.6,0.5])
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
               color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')


def plot_var_random():
    dist = 'beta'
    n = 10000
    if dist == 'uniform':
        var = np.random.rand(2 * n)
    elif dist == 'beta':
        var = np.random.beta(4, 3, size=(2 * n,))
    elif dist == 'gamma':
        var = np.random.gamma(1, 2, size=(2 * n,))
    elif dist == 'lognormal':
        var = np.random.randn(2 * n) * 1.9 + 0.75
        var = var * (var < 6) + 6.0 * (var >= 6)
        var = np.exp(var)

    frac_var = (var[:n] - var[n:]) / (var[:n] + var[n:])

    plt.figure(figsize=(2, 2))
    plt.hist(var)

    plt.figure(figsize=(2, 2))
    plt.hist(frac_var)


if __name__ == '__main__':
    pass

