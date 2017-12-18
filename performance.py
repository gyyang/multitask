"""
This file contains functions that test the behavior of the model
These functions generally involve some psychometric measurements of the model,
for example performance in decision-making tasks as a function of input strength

These measurements are important as they show whether the network exhibits
some critically important computations, including integration and generalization.
"""

from __future__ import division

import os
import numpy as np
import pickle
import time
import copy
from collections import OrderedDict
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work

import tensorflow as tf

from task import generate_trials, rule_name, get_dist
from network import Model, get_perf
import tools

rule_color={'reactgo'           : sns.xkcd_palette(['green'])[0],
            'delaygo'           : sns.xkcd_palette(['olive'])[0],
            'fdgo'              : sns.xkcd_palette(['forest green'])[0],
            'reactanti'         : sns.xkcd_palette(['mustard'])[0],
            'delayanti'         : sns.xkcd_palette(['tan'])[0],
            'fdanti'            : sns.xkcd_palette(['brown'])[0],
            'dm1'               : sns.xkcd_palette(['lavender'])[0],
            'dm2'               : sns.xkcd_palette(['aqua'])[0],
            'contextdm1'        : sns.xkcd_palette(['bright purple'])[0],
            'contextdm2'        : sns.xkcd_palette(['green blue'])[0],
            'multidm'           : sns.xkcd_palette(['blue'])[0],
            'delaydm1'          : sns.xkcd_palette(['indigo'])[0],
            'delaydm2'          : sns.xkcd_palette(['grey blue'])[0],
            'contextdelaydm1'   : sns.xkcd_palette(['royal purple'])[0],
            'contextdelaydm2'   : sns.xkcd_palette(['dark cyan'])[0],
            'multidelaydm'      : sns.xkcd_palette(['royal blue'])[0],
            'dmsgo'             : sns.xkcd_palette(['red'])[0],
            'dmsnogo'           : sns.xkcd_palette(['rose'])[0],
            'dmcgo'             : sns.xkcd_palette(['orange'])[0],
            'dmcnogo'           : sns.xkcd_palette(['peach'])[0]
            }

save = True

def plot_trainingprogress(save_name, rule_plot=None, save=True):
    # Plot Training Progress
    log = tools.load_log(save_name)

    trials      = log['trials']
    times       = log['times']
    cost_tests  = log['cost_tests']
    perf_tests  = log['perf_tests']

    fs = 6 # fontsize
    fig = plt.figure(figsize=(3.5,2.5))
    d1, d2 = 0.015, 0.35
    ax1 = fig.add_axes([0.15,0.5+d1,   0.5,d2])
    ax2 = fig.add_axes([0.15,0.5-d1-d2,0.5,d2])
    lines = list()
    labels = list()

    x_plot = np.array(trials)/1000.
    if rule_plot == None:
        rule_plot = cost_tests.keys()

    for i, rule in enumerate(rule_plot):
        # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
        line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=rule_color[rule])
        ax2.plot(x_plot, perf_tests[rule],color=rule_color[rule])
        lines.append(line[0])
        labels.append(rule_name[rule])

    if 'cost_ewcs' in log:
        ax1.plot(x_plot, np.log10(log['cost_ewcs']), color='black')

    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)

    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Total trials (1,000)',fontsize=fs, labelpad=2)
    ax2.set_ylabel('Performance',fontsize=fs)
    ax1.set_ylabel(r'$log_{10}$(cost)',fontsize=fs)
    ax1.set_xticklabels([])
    ax1.set_title('Training time {:0.1f} hours'.format(times[-1]/3600.),fontsize=fs)
    ax1.locator_params(axis='y', nbins=3)
    ax2.locator_params(axis='y', nbins=4)
    lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.7,0.5),
                    fontsize=fs,labelspacing=0.3,loc=6)
    plt.setp(lg.get_title(),fontsize=fs)
    if save:
        plt.savefig('figure/Training_Progress'+save_name+'.pdf', transparent=True)
    plt.show()

def plot_performanceprogress(save_name, rule_plot=None, save=True):
    # Plot Training Progress
    log = tools.load_log(save_name)
    config = tools.load_config(save_name)

    trials      = log['trials']
    times       = log['times']
    cost_tests  = log['cost_tests']
    perf_tests  = log['perf_tests']

    fs = 6 # fontsize
    fig = plt.figure(figsize=(3.5,1.2))
    ax = fig.add_axes([0.1,0.25,0.35,0.6])
    lines = list()
    labels = list()

    x_plot = np.array(trials)/1000.
    if rule_plot == None:
        rule_plot = config['rules']

    for i, rule in enumerate(rule_plot):
        # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
        line = ax.plot(x_plot, np.log10(cost_tests[rule]),color=rule_color[rule])
        ax.plot(x_plot, perf_tests[rule],color=rule_color[rule])
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_ylim([0, 1])
    ax.set_xlabel('Total trials (1,000)',fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance',fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=3)
    ax.set_yticks([0,1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    lg = fig.legend(lines, labels, title='Task',ncol=2,bbox_to_anchor=(0.47,0.5),
                    fontsize=fs,labelspacing=0.3,loc=6,frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    if save:
        plt.savefig('figure/Performance_Progress'+save_name+'.pdf', transparent=True)
    plt.show()

def plot_performanceprogress_cont(save_names, save=True, **kwargs):
    save_name1, save_name2 = save_names
    _plot_performanceprogress_cont(save_name1, save=save, **kwargs)
    _plot_performanceprogress_cont(save_name1, save_name2=save_name2, save=save, **kwargs)

def _plot_performanceprogress_cont(save_name, save=True, save_name2=None, dims=None):
    # Plot Training Progress
    log = tools.load_log(save_name)
    config = tools.load_config(save_name)

    trials      = np.array(log['trials'])/1000.
    times       = log['times']
    perf_tests  = log['perf_tests']
    rule_now    = log['rule_now']

    if save_name2 is not None:
        log2 = tools.load_log(save_name2)

        trials2      = np.array(log2['trials'])/1000.
        perf_tests2  = log2['perf_tests']

    fs = 7 # fontsize
    lines = list()
    labels = list()


    rule_train_plot = config['rule_trains']
    rule_test_plot  = config['rules']

    if dims is None:
        nx, ny = 4, 3
    else:
        nx, ny = dims

    print(nx, ny)
    fig, axarr = plt.subplots(nx, ny, figsize=(nx,ny), sharex=True)

    f_rule_color = lambda r : rule_color[r[0]] if hasattr(r, '__iter__') else rule_color[r]

    # for i, rule in enumerate(rule_test_plot):
    #     pass

    for i in range(int(nx*ny)):
        ix, iy = i%nx, int(i/nx)
        print(ix, iy)
        ax = axarr[ix, iy]

        if i >= len(rule_test_plot):
            ax.axis('off')
            continue

        rule = rule_test_plot[i]

        # Plot fills
        trials_rule_prev_end = 0 # end of previous rule training time
        for rule_ in rule_train_plot:
            alpha = 0.2
            if hasattr(rule_, '__iter__'):
                if rule in rule_:
                    lw = 2
                    ec = 'black'
                else:
                    lw = 0.5
                    ec = (0,0,0,0.1)
            else:
                if rule == rule_:
                    lw = 2
                    ec = 'black'
                else:
                    lw = 0.5
                    ec = (0,0,0,0.1)
            trials_rule_now = [trials_rule_prev_end] + \
                              [trials[ii] for ii in range(len(rule_now)) if rule_now[ii]==rule_]
            trials_rule_prev_end = trials_rule_now[-1]
            # ax.fill_between(trials_rule_now, 0, 1, facecolor=f_rule_color(rule_)+(alpha,),
            #                 edgecolor=(0,0,0,0.5), linewidth=lw)
            ax.fill_between(trials_rule_now, 0, 1, facecolor='none',
                            edgecolor=ec, linewidth=0.5)

        # Plot lines
        # line = ax.plot(trials, perf_tests[rule], lw=0.75, color=f_rule_color(rule))
        line = ax.plot(trials, perf_tests[rule], lw=1, color='gray')
        if save_name2 is not None:
            # ax.plot(trials2, perf_tests2[rule], lw=1.5, color=f_rule_color(rule))
            ax.plot(trials2, perf_tests2[rule], lw=1, color='red')
        lines.append(line[0])
        if not hasattr(rule, '__iter__'):
            rule_name_print = rule_name[rule]
        else:
            rule_name_print = ' & '.join([rule_name[r] for r in rule])
        labels.append(rule_name_print)

        ax.tick_params(axis='both', which='major', labelsize=fs)

        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, trials_rule_prev_end])
        ax.set_yticks([0, 1])
        ax.set_xticks([0, np.floor(trials_rule_prev_end/100.)*100])
        if (ix==nx-1) and (iy==0):
            ax.set_xlabel('Total trials (1,000)',fontsize=fs, labelpad=1)
        if i == 0:
            ax.set_ylabel('Performance', fontsize=fs, labelpad=1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('left')

        # ax.set_title(rule_name[rule], fontsize=fs, y=0.87, color=rule_color[rule])
        ax.set_title(rule_name[rule], fontsize=fs, y=0.87, color='black')

    print('Training time {:0.1f} hours'.format(times[-1]/3600.))
    if save:
        # plt.savefig('figure/TrainingCont_Progress'+save_name+'.pdf', transparent=True)
        name = 'TrainingCont_Progress'
        if save_name2 is not None:
            name = name + '2'
        plt.savefig('figure/'+name+'.pdf', transparent=True)
    plt.show()

def get_finalperformance(save_pattern):
    save_names = tools.valid_save_names(save_pattern)

    config = tools.load_config(save_names[0])

    rule_plot = config['rules']

    final_cost, final_perf = OrderedDict(), OrderedDict()
    for rule in rule_plot:
        final_cost[rule] = list()
        final_perf[rule] = list()
    training_time_plot = list()

    # Recording performance and cost for networks
    for save_name in save_names:
        log = tools.load_log(save_name)
        if log is None:
            continue

        cost_tests = log['cost_tests']
        perf_tests = log['perf_tests']

        for rule in rule_plot:
            final_perf[rule] += [float(perf_tests[rule][-1])]
            final_cost[rule] += [float(cost_tests[rule][-1])]
        training_time_plot.append(log['times'][-1])

    return final_cost, final_perf, rule_plot, training_time_plot


def obsolete_plot_finalperformance(save_type, save_type_end=None):

    var_plot, final_cost, final_perf, rule_plot, training_time_plot = \
        get_finalperformance(save_type, save_type_end)

    fig = plt.figure(figsize=(5,3))
    d1, d2 = 0.01, 0.35
    ax1 = fig.add_axes([0.15,0.5+d1,   0.5,d2])
    ax2 = fig.add_axes([0.15,0.5-d1-d2,0.5,d2])
    lines = list()
    labels = list()
    for i, rule in enumerate(rule_plot):
        # line = ax1.plot(x_plot,np.log10(final_cost[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot,final_perf[rule],color=color_rules[i%26])
        line = ax1.plot(var_plot,np.log10(final_cost[rule]),color=rule_color[rule])
        ax2.plot(var_plot,final_perf[rule],color=rule_color[rule])
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax2.plot(var_plot,np.array([final_perf[r] for r in rule_plot]).mean(axis=0),color='black',linewidth=3)

    print('Overall final performance {:0.2f}'.format(np.array([final_perf[r] for r in rule_plot]).mean()))

    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)

    ax2.set_ylim(top=1.05)
    ax2.set_xlabel('Number of Recurrent Units',fontsize=7)
    ax2.set_ylabel('performance',fontsize=7)
    ax1.set_ylabel(r'$log_{10}$(cost)',fontsize=7)
    ax1.set_xticklabels([])
    # ax1.set_title('After {:.1E} trials'.format(n_trial),fontsize=7)
    ax1.locator_params(axis='y', nbins=3)
    ax2.locator_params(axis='y', nbins=4)
    lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.7,0.5),
                    fontsize=7,labelspacing=0.3,loc=6)
    plt.setp(lg.get_title(),fontsize=7)
    plt.savefig('figure/FinalCostPerformance_'+save_type+'.pdf', transparent=True)
    plt.show()

    # Training time as a function of number of recurrent units
    x_plot = np.array(var_plot)/100.
    y_plot = np.array(training_time_plot)/3600.
    fig = plt.figure(figsize=(5,1.5))
    ax  = fig.add_axes([0.15, 0.25, 0.5, 0.7])
    p = np.polyfit(x_plot[x_plot<5], y_plot[x_plot<5], 2) # Temporary
    #p = np.polyfit(x_plot, y_plot, 2)
    ax.plot(x_plot, p[0]*x_plot**2+p[1]*x_plot+p[2], color='black')
    ax.plot(x_plot, y_plot, color=sns.xkcd_palette(['cerulean'])[0])
    ax.text(0.05, 0.7, 'Number of Hours = \n {:0.2f}$(N/100)^2$+{:0.2f}$(N/100)$+{:0.2f}'.format(*p), fontsize=7, transform=ax.transAxes)
    ax.set_xlabel('Number of Recurrent Units $N$ (100)',fontsize=7)
    ax.set_ylabel('Training time (hours)',fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    if save:
        plt.savefig('figure/FinalTrainingTime_'+save_type+'.pdf', transparent=True)
    plt.show()

def plot_finalperformance_cont(save_pattern1, save_pattern2):
    final_cost, final_perf1, rule_plot, training_time_plot = \
        get_finalperformance(save_pattern1)

    final_cost, final_perf2, rule_plot, training_time_plot = \
        get_finalperformance(save_pattern2)

    final_perf_plot1 = np.array(final_perf1.values())
    final_perf_plot2 = np.array(final_perf2.values())

    fig = plt.figure(figsize=(3.0,2.5))
    ax = fig.add_axes([0.2, 0.4, 0.7, 0.4])
    ax.plot(final_perf_plot1, 'o-', color='gray', markersize=3)
    ax.plot(final_perf_plot2, 'o-', color='red', markersize=3)

    ax.plot(final_perf_plot1[:,0], 'o-', color='gray', markersize=3, label='Traditional learning')
    ax.plot(final_perf_plot2[:,0], 'o-', color='red', markersize=3, label='Continual learning')

    tick_names = [rule_name[r] for r in rule_plot]
    plt.xticks(range(len(tick_names)), tick_names,
           rotation=90, va='top', fontsize=6)

    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim([-1, len(rule_plot)])
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('Performance',fontsize=7)
    ax.locator_params(axis='y', nbins=3)
    lg = ax.legend(ncol=1, bbox_to_anchor=(0.5,1.0),
                    fontsize=7, loc=8, frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plt.setp(lg.get_title(),fontsize=7)
    if save:
        plt.savefig('figure/FinalCostPerformanceCont.pdf', transparent=True)
    plt.show()


def get_allperformance(save_pattern, param_list=None):
    # Get all model names that match patterns (strip off .ckpt.meta at the end)
    save_names = tools.valid_save_names(save_pattern)

    final_perfs = dict()
    filenames = dict()

    if param_list is None:
        param_list = ['param_intsyn', 'easy_task', 'activation']

    for save_name in save_names:
        log = tools.load_log(save_name)
        config = tools.load_config(save_name)

        perf_tests = log['perf_tests']

        final_perf = np.mean([float(val[-1]) for val in perf_tests.values()])

        key = tuple([config[p] for p in param_list])
        if key in final_perfs.keys():
            final_perfs[key].append(final_perf)
        else:
            final_perfs[key] = [final_perf]
            filenames[key] = save_name

    for key, val in final_perfs.iteritems():
        final_perfs[key] = np.mean(val)
        print(key),
        print('{:0.3f}'.format(final_perfs[key])),
        print(filenames[key])

def plot_finalperformance_lr():
    # Specialized for varyign learning rate. Can be rewritten, but I'm lazy.
    n_lr = 100
    save_type = 'allrule_weaknoise'
    save_name = save_type+'_lr1'
    log = tools.load_log(save_name)
    cost_tests  = log['cost_tests']
    perf_tests  = log['perf_tests']

    final_cost = {k: [] for k in cost_tests}
    final_perf = {k: [] for k in cost_tests}
    lr_plot = list()
    training_time_plot = list()
    # Recording performance and cost for networks
    for i_lr in range(n_lr):
        save_name = save_type+'_lr'+str(i_lr)
        log = tools.load_log(save_name)
        if log is None:
            continue
        cost_tests = log['cost_tests']
        perf_tests = log['perf_tests']
        # if perf_tests[DMCGO][-1] > 0.1:
        for key in perf_tests.keys():
            final_perf[key] += [float(perf_tests[key][-1])]
            final_cost[key]        += [float(cost_tests[key][-1])]
        # lr_plot.append(config['learning_rate'])
        lr_plot.append(np.logspace(-2,-4,100)[i_lr]) # Temporary
        training_time_plot.append(log['times'][-1])

    n_trial = log['trials'][-1]
    x_plot = np.log10(np.array(lr_plot))
    rule_plot = None
    if rule_plot == None:
        rule_plot = perf_tests.keys()

    fig = plt.figure(figsize=(5,3))
    d1, d2 = 0.01, 0.35
    ax1 = fig.add_axes([0.15,0.5+d1,   0.5,d2])
    ax2 = fig.add_axes([0.15,0.5-d1-d2,0.5,d2])
    lines = list()
    labels = list()
    for i, rule in enumerate(rule_plot):
        line = ax1.plot(x_plot,np.log10(final_cost[rule]),color=color_rules[i%26])
        ax2.plot(x_plot,final_perf[rule],color=color_rules[i%26])
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)

    ax2.set_ylim(top=1.05)
    ax2.set_xlabel('Log (Learning rate)',fontsize=7)
    ax2.set_ylabel('performance',fontsize=7)
    ax1.set_ylabel(r'$log_{10}$(cost)',fontsize=7)
    ax1.set_xticklabels([])
    ax1.set_title('After {:.1E} trials'.format(n_trial),fontsize=7)
    lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.65,0.5),
                    fontsize=7,labelspacing=0.3,loc=6)
    plt.setp(lg.get_title(),fontsize=7)
    plt.savefig('figure/FinalCostPerformance_'+save_type+'_varylr.pdf', transparent=True)
    plt.show()

    # x_plot = np.array(HDIM_plot)/100.
    # y_plot = np.array(training_time_plot)/3600.
    # fig = plt.figure(figsize=(5,1.5))
    # ax  = fig.add_axes([0.15, 0.25, 0.5, 0.7])
    # p = np.polyfit(x_plot[x_plot<5], y_plot[x_plot<5], 2) # Temporary
    # #p = np.polyfit(x_plot, y_plot, 2)
    # ax.plot(x_plot, p[0]*x_plot**2+p[1]*x_plot+p[2], color='black')
    # ax.plot(x_plot, y_plot, color=sns.xkcd_palette(['cerulean'])[0])
    # ax.text(0.05, 0.7, 'Number of Hours = \n {:0.2f}$(N/100)^2$+{:0.2f}$(N/100)$+{:0.2f}'.format(*p), fontsize=7, transform=ax.transAxes)
    # ax.set_xlabel('Number of Recurrent Units $N$ (100)',fontsize=7)
    # ax.set_ylabel('Training time (hours)',fontsize=7)
    # ax.tick_params(axis='both', which='major', labelsize=7)
    # plt.savefig('figure/FinalTrainingTime_'+save_type+'.pdf', transparent=True)
    # plt.show()


################ Psychometric - Varying Coherence #############################

def _psychometric_dm(save_name, rule, params_list, batch_shape):
    """Base function for computing psychometric performance in 2AFC tasks

    Args:
        save_name : model name
        rule : task to analyze
        params_list : a list of parameter dictionaries used for the psychometric mode
        batch_shape : shape of each batch. Each batch should have shape (n_rep, ...)
        n_rep is the number of repetitions that will be averaged over

    Return:
        ydatas: list of performances
    """
    print('Starting psychometric analysis of the {:s} task...'.format(rule_name[rule]))

    model = Model(save_name)
    config = model.config
    with tf.Session() as sess:
        model.restore(sess)

        ydatas = list()
        for params in params_list:

            trial  = generate_trials(rule, config, 'psychometric', params=params)
            y_loc_sample = model.get_y_loc(model.get_y(trial.x))
            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            stim1_locs_ = np.reshape(params['stim1_locs'], batch_shape)
            stim2_locs_ = np.reshape(params['stim2_locs'], batch_shape)

            # Average over the first dimension of each batch
            choose1 = (get_dist(y_loc_sample - stim1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - stim2_locs_) < 0.3*np.pi).sum(axis=0)
            ydatas.append(choose1/(choose1 + choose2))

    return ydatas

def psychometric_choice(save_name, **kwargs):
    rule = 'dm1'
    stim_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.05
    n_stim_loc = 300
    n_stim = len(stim_cohs)
    batch_size = n_stim_loc * n_stim
    batch_shape = (n_stim_loc, n_stim)
    ind_stim_loc, ind_stim = np.unravel_index(range(batch_size),batch_shape)

    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    stim1_strengths = 1 + stim_cohs[ind_stim]
    stim2_strengths = 1 - stim_cohs[ind_stim]

    params_list = list()
    stim_times = [200, 400, 800]
    for stim_time in stim_times:
        params = {'stim1_locs' : stim1_locs,
                  'stim2_locs' : stim2_locs,
                  'stim1_strengths' : stim1_strengths,
                  'stim2_strengths' : stim2_strengths,
                  'stim_time'    : stim_time}

        params_list.append(params)

    xdatas = [stim_cohs*2] * len(stim_times)
    ydatas = _psychometric_dm(save_name, rule, params_list, batch_shape)

    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in stim_times],
                              colors=sns.dark_palette("light blue", 3, input="xkcd"),
                              legtitle='Stim. time (ms)', rule=rule, **kwargs)

def psychometric_delaychoice(save_name, **kwargs):
    rule = 'delaydm1'

    # n_stim_loc = 120
    n_stim_loc = 12
    n_stim = 7
    batch_size = n_stim_loc * n_stim
    batch_shape = (n_stim_loc, n_stim)
    ind_stim_loc, ind_stim = np.unravel_index(range(batch_size),batch_shape)

    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    stim_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.5
    stim1_strengths = 1 + stim_cohs[ind_stim]
    stim2_strengths = 1 - stim_cohs[ind_stim]

    stim1_ons = 200
    stim1_offs = 400

    dtars = [200,800,3200] # stim1 offset and stim2 onset time difference
    # dtars = [2800,3000,3200] # stim1 offset and stim2 onset time difference
    params_list = list()
    for dtar in dtars:
        stim2_ons  = stim1_offs + dtar
        stim2_offs = stim2_ons + 200
        params = {'stim1_locs'    : stim1_locs,
                  'stim2_locs'    : stim2_locs,
                  'stim1_strengths' : stim1_strengths,
                  'stim2_strengths' : stim2_strengths,
                  'stim1_ons'     : stim1_ons,
                  'stim1_offs'    : stim1_offs,
                  'stim2_ons'     : stim2_ons,
                  'stim2_offs'    : stim2_offs,
                  }
        params_list.append(params)

    xdatas = [stim_cohs*2] * len(dtars)
    ydatas = _psychometric_dm(save_name, rule, params_list, batch_shape)

    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in dtars],
                              colors=sns.dark_palette("light blue", 3, input="xkcd"),
                              legtitle='Delay (ms)',rule=rule, **kwargs)

def psychometric_choiceattend(save_name, **kwargs):
    psychometric_choiceattend_(save_name, 'contextdm1', **kwargs)
    psychometric_choiceattend_(save_name, 'contextdm2', **kwargs)

def psychometric_choiceattend_(save_name, rule, **kwargs):
    stim_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.05

    n_stim_loc = 100 # increase repeat by increasing this
    n_stim = len(stim_cohs)
    batch_size = n_stim_loc * n_stim**2
    batch_shape = (n_stim_loc,n_stim,n_stim)
    ind_stim_loc, ind_stim_mod1, ind_stim_mod2 = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc

    # print('Using only one target location now')
    # stim1_locs = np.zeros(len(ind_stim_loc))

    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    params = {'stim1_locs' : stim1_locs,
              'stim2_locs' : stim2_locs,
              'stim1_mod1_strengths' : 1 + stim_cohs[ind_stim_mod1],
              'stim2_mod1_strengths' : 1 - stim_cohs[ind_stim_mod1],
              'stim1_mod2_strengths' : 1 + stim_cohs[ind_stim_mod2],
              'stim2_mod2_strengths' : 1 - stim_cohs[ind_stim_mod2],
              'stim_time'    : 800}


    prop1s = _psychometric_dm(save_name, rule, [params], batch_shape)[0]

    xdatas = [stim_cohs*2]*2
    ydatas = [prop1s.mean(axis=k) for k in [1,0]]

    labels = ['Attend', 'Ignore'] if rule=='contextdm1' else ['Ignore', 'Attend']

    plot_psychometric_choice(xdatas, ydatas,
                              labels=labels,
                              colors=sns.color_palette("Set2",2),
                              legtitle='Modality',rule=rule, **kwargs)

def psychometric_choiceint(save_name, **kwargs):
    rule = 'multidm'
    stim_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.05
    n_stim_loc = 500 # increase repeat by increasing this
    n_stim = len(stim_cohs)
    batch_size = n_stim_loc * n_stim
    batch_shape = (n_stim_loc,n_stim)
    ind_stim_loc, ind_stim1_strength = np.unravel_index(range(batch_size),batch_shape)

    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    stim1_strengths = 1 + stim_cohs[ind_stim1_strength]
    stim2_strengths = 1 - stim_cohs[ind_stim1_strength]

    mod_strengths = [(1,0), (0,1), (1,1)]
    params_list = list()
    for mod_strength in mod_strengths:
        params = {'stim1_locs' : stim1_locs,
                  'stim2_locs' : stim2_locs,
                  'stim1_mod1_strengths' : stim1_strengths*mod_strength[0],
                  'stim2_mod1_strengths' : stim2_strengths*mod_strength[0],
                  'stim1_mod2_strengths' : stim1_strengths*mod_strength[1],
                  'stim2_mod2_strengths' : stim2_strengths*mod_strength[1],
                  'stim_time'    : 400}
        params_list.append(params)

    xdatas = [stim_cohs*2] * len(mod_strengths)
    ydatas = _psychometric_dm(save_name, rule, params_list, batch_shape)

    fits = plot_psychometric_choice(xdatas, ydatas,
                                    labels=['1 only', '2 only', 'both'],
                                    colors=sns.color_palette("Set2",3),
                                    legtitle='Modality', rule=rule, **kwargs)
    sigmas = [fit[1] for fit in fits]
    print('Fit sigmas:')
    print(sigmas)

def psychometric_intrepro(save_name):
    with Run(save_name, fast_eval=fast_eval) as R:

        n_stim_loc = 360
        # intervals = [700]
        # intervals = [500, 600, 700, 800, 900, 1000]
        intervals = np.linspace(500, 1000, 10)
        mean_sample_intervals = list()
        for interval in intervals:
            batch_size = n_stim_loc
            stim_mod1_locs  = 2*np.pi*np.arange(n_stim_loc)/n_stim_loc

            params = {'stim_mod1_locs'  : stim_mod1_locs,
                      'interval'       : interval}

            task  = generate_onebatch(INTREPRO, R.config, 'psychometric', params=params)
            h_test = R.f_h(task.x)
            y = R.f_y(h_test)

            sample_intervals = list() # sampled interval test
            for i_batch in range(batch_size):
                try: ##TODO: Temporary solution
                    # Setting the threshold can be tricky, but doesn't impact the overall results
                    sample_interval = np.argwhere(y[:,i_batch,0]<0.3)[0]-task.epochs['stim2'][1]
                    sample_intervals.append(sample_interval)
                except IndexError:
                    # print i_batch
                    pass
            mean_sample_intervals.append(np.mean(sample_intervals))

        fig = plt.figure(figsize=(2,1.5))
        ax = fig.add_axes([0.25,0.25,0.65,0.65])
        ax.plot(intervals, mean_sample_intervals, 'o-', markersize=3.5, color=sns.xkcd_palette(['blue'])[0])
        ax.plot(intervals, intervals, color=sns.xkcd_palette(['black'])[0])
        plt.xlim([intervals[0]-50,intervals[-1]+50])
        plt.ylim([intervals[0]-50,intervals[-1]+50])
        plt.xlabel('Sample Interval (ms)',fontsize=7)
        plt.ylabel('Production interval (ms)',fontsize=7)
        plt.title('Rule '+rule_name[INTREPRO], fontsize=7)
        plt.locator_params(nbins=5)
        ax.tick_params(axis='both', which='major', labelsize=7)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/analyze_'+rule_name[INTREPRO].replace(' ','')+'_performance.pdf', transparent=True)
        plt.show()

def plot_psychometric_choice(xdatas, ydatas, labels, colors, **kwargs):
    """
    Standard function for plotting the psychometric curves

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    """
    fs = 6
    fig = plt.figure(figsize=(1.8,1.3))
    ax = fig.add_axes([0.25,0.25,0.65,0.65])
    fits = list()
    for i in range(len(xdatas)):
        # Analyze performance of the choice tasks
        cdf_gaussian = lambda x, mu, sigma : stats.norm.cdf(x, mu, sigma)

        xdata = xdatas[i]
        ydata = ydatas[i]
        ax.plot(xdata, ydata, 'o', markersize=3.5, color=colors[i])

        try:
            x_plot = np.linspace(xdata[0],xdata[-1],100)
            (mu,sigma), _ = curve_fit(cdf_gaussian, xdata, ydata, bounds=([-0.5,0.001],[0.5,10]))
            fits.append((mu,sigma))
            ax.plot(x_plot, cdf_gaussian(x_plot,mu,sigma), label=labels[i],
                    linewidth=1, color=colors[i])
        except:
            pass

    plt.xlabel('Stim. 1 - Stim. 2',fontsize=fs)
    plt.ylim([-0.05,1.05])
    plt.xlim([xdata[0]*1.1,xdata[-1]*1.1])
    plt.yticks([0,0.5,1])
    if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
        plt.yticks([0,0.5,1],['','',''])
    else:
        plt.ylabel('P(choice 1)',fontsize=fs)
    plt.title(rule_name[kwargs['rule']], fontsize=fs)
    plt.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    if len(xdatas)>1:
        if len(kwargs['legtitle'])>10:
            loc = (0.0, 0.5)
        else:
            loc = (0.0, 0.5)
        leg = plt.legend(title=kwargs['legtitle'],fontsize=fs,frameon=False,
                         loc=loc,labelspacing=0.3)
        plt.setp(leg.get_title(),fontsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    figname = 'figure/analyze_'+rule_name[kwargs['rule']].replace(' ','') + '_performance'
    if 'figname_append' in kwargs:
        figname += kwargs['figname_append']

    if save:
        plt.savefig(figname+'.pdf', transparent=True)
    plt.show()
    return fits

def psychometric_choicefamily_2D(save_name, rule, lesion_units=None,
                                 n_coh=8, n_stim_loc=20, coh_range=0.1):
    # Generate task parameters for choice tasks
    # coh_range = 0.2
    # coh_range = 0.05
    cohs = np.linspace(-coh_range, coh_range, n_coh)

    batch_size = n_stim_loc * n_coh**2
    batch_shape = (n_stim_loc,n_coh,n_coh)
    ind_stim_loc, ind_stim_mod1, ind_stim_mod2 = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    stim_mod1_cohs = cohs[ind_stim_mod1]
    stim_mod2_cohs = cohs[ind_stim_mod2]

    params_dict = dict()
    params_dict[CHOICE_MOD1] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_strengths' : 1 + stim_mod1_cohs, # Just use mod 1 value
          'stim2_strengths' : 1 - stim_mod1_cohs,
          'stim_time'    : 800
          }
    params_dict[CHOICE_MOD2] = params_dict[CHOICE_MOD1]

    params_dict[CHOICEATTEND_MOD1] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
          'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
          'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
          'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
          'stim_time'    : 800
          }

    params_dict[CHOICEATTEND_MOD2] = params_dict[CHOICEATTEND_MOD1]

    params_dict[CHOICEDELAYATTEND_MOD1] = params_dict[CHOICEATTEND_MOD1]
    params_dict[CHOICEDELAYATTEND_MOD1]['stim_time'] = 800
    params_dict[CHOICEDELAYATTEND_MOD2] = params_dict[CHOICEDELAYATTEND_MOD1]

    params_dict[CHOICE_INT] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
          'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
          'stim1_mod2_strengths' : 1 + stim_mod1_cohs, # Same as Mod 1
          'stim2_mod2_strengths' : 1 - stim_mod1_cohs,
          'stim_time'    : 800
          }

    params_dict[CHOICEDELAYATTEND_MOD1] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
          'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
          'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
          'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
          'stim_time'    : 800
          }

    with Run(save_name, lesion_units=lesion_units, fast_eval=True) as R:

        params = params_dict[rule]
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)

        # print('Using temporary rule setup')
        # task  = generate_onebatch(rule, R.config, 'psychometric', params=params,
        #                           add_rule=[CHOICEDELAY_MOD2, CHOICEATTEND_MOD2, CHOICE_INT],
        #                           rule_strength=[0., 1., -0.])

        y_sample = R.f_y_from_x(task.x)
        y_sample_loc = R.f_y_loc(y_sample)

    perf = get_perf(y_sample, task.y_loc)
    print('Performance {:0.3f}'.format(np.mean(perf)))

    # Compute the overall performance.
    # Importantly, discard trials where no decision was made to one of the choices
    loc_cor = task.y_loc[-1] # last time point, correct locations
    loc_err = (loc_cor+np.pi)%(2*np.pi)
    choose_cor = (get_dist(y_sample_loc[-1] - loc_cor) < 0.3*np.pi).sum()
    choose_err = (get_dist(y_sample_loc[-1] - loc_err) < 0.3*np.pi).sum()
    perf = choose_cor/(choose_cor+choose_err)

    # Compute the proportion of choosing choice 1, while maintaining the batch_shape
    stim1_locs_ = np.reshape(stim1_locs, batch_shape)
    stim2_locs_ = np.reshape(stim2_locs, batch_shape)

    y_sample_loc = np.reshape(y_sample_loc[-1], batch_shape)
    choose1 = (get_dist(y_sample_loc - stim1_locs_) < 0.3*np.pi).sum(axis=0)
    choose2 = (get_dist(y_sample_loc - stim2_locs_) < 0.3*np.pi).sum(axis=0)
    prop1s = choose1/(choose1 + choose2)

    return perf, prop1s, cohs

def _plot_psychometric_choicefamily_2D(prop1s, cohs, rule, title=None, **kwargs):
    n_coh = len(cohs)

    fs = 6
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    im = ax.imshow(prop1s, cmap='BrBG', origin='lower',
                   aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Mod 2 coh.', fontsize=fs, labelpad=-3)
    plt.xticks([0, n_coh-1], [cohs[0], cohs[-1]],
               rotation=0, va='center', fontsize=fs)
    if 'ylabel' in kwargs and kwargs['ylabel']==False:
        plt.yticks([])
    else:
        ax.set_ylabel('Mod 1 coh.', fontsize=fs, labelpad=-3)
        plt.yticks([0, n_coh-1], [cohs[0], cohs[-1]],
                   rotation=0, va='center', fontsize=fs)
    if title is not None:
        plt.title(title, fontsize=fs)
    ax.tick_params('both', length=0)
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set_visible(False)

    if 'colorbar' in kwargs and kwargs['colorbar']==False:
        pass
    else:
        ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Prop. of choice 1', fontsize=fs, labelpad=-3)
        plt.tick_params(axis='both', which='major', labelsize=fs)

    if save:
        if 'save_name' not in kwargs:
            save_name = rule_name[rule].replace(' ','')+'_perf2D.pdf'
        else:
            save_name = kwargs['save_name']
        plt.savefig(os.path.join('figure', save_name), transparent=True)

    plt.show()

def plot_psychometric_choicefamily_2D(save_name, rule, **kwargs):
    perf, prop1s, cohs = psychometric_choicefamily_2D(save_name, rule, **kwargs)
    _plot_psychometric_choicefamily_2D(prop1s, cohs, rule)


################ Psychometric - Varying Stim Time #############################

def compute_choicefamily_varytime(save_name, rule):
    assert rule in ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm']
    print('Starting vary time analysis of the {:s} task...'.format(rule_name[rule]))
    n_stim_loc = 3000
    # n_stim_loc = 1000
    # n_stim_loc = 100

    n_coh = 4
    batch_size = n_stim_loc * n_coh
    batch_shape = (n_stim_loc,n_coh)
    ind_stim_loc, ind_stim = np.unravel_index(range(batch_size),batch_shape)

    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    # if rule == CHOICE_INT:
    #     stim_str_range = 0.02
    # else:
    #     stim_str_range = 0.04
    stim_str_range = 0.04

    cohs = stim_str_range*(2**np.arange(n_coh))/(2**(n_coh-1))
    stim1_strengths = 1 + cohs[ind_stim]
    stim2_strengths = 1 - cohs[ind_stim]

    params_list = list()
    stim_times = np.logspace(np.log10(200), np.log10(1500), 8, dtype=int)
    for stim_time in stim_times:
        if rule in ['dm1', 'dm2']:
            params = {'stim1_locs' : stim1_locs,
                      'stim2_locs' : stim2_locs,
                      'stim1_strengths' : stim1_strengths,
                      'stim2_strengths' : stim2_strengths,
                      'stim_time'    : stim_time}

        elif rule == 'multidm':
            params = {'stim1_locs' : stim1_locs,
                      'stim2_locs' : stim2_locs,
                      'stim1_mod1_strengths' : 1 + cohs[ind_stim],
                      'stim2_mod1_strengths' : 1 - cohs[ind_stim],
                      'stim1_mod2_strengths' : 1 + cohs[ind_stim],
                      'stim2_mod2_strengths' : 1 - cohs[ind_stim],
                      'stim_time'    : stim_time}

        elif rule in ['contextdm1', 'contextdm2']:
            if rule == 'contextdm1':
                att = '1'
                ign = '2'
            else:
                att = '2'
                ign = '1'
            params = {'stim1_locs' : stim1_locs,
                      'stim2_locs' : stim2_locs,
                      'stim1_mod'+att+'_strengths' : 1 + cohs[ind_stim],
                      'stim2_mod'+att+'_strengths' : 1 - cohs[ind_stim],
                      'stim1_mod'+ign+'_strengths' : np.ones(batch_size),
                      'stim2_mod'+ign+'_strengths' : np.ones(batch_size),
                      'stim_time'    : stim_time}


        params_list.append(params)

    ydatas = _psychometric_dm(save_name, rule, params_list, batch_shape)

    xdatas = [stim_times] * n_coh
    ydatas = np.array(ydatas).T

    result = {'xdatas' : xdatas, 'ydatas' : ydatas, 'cohs' : cohs}

    savename = 'data/varytime_'+rule_name[rule].replace(' ','') +save_name
    with open(savename+'.pkl','wb') as f:
        pickle.dump(result, f)

def plot_choicefamily_varytime(save_name, rule):
    assert rule in ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm']
    savename = 'data/varytime_'+rule_name[rule].replace(' ','') +save_name

    with open(savename+'.pkl','rb') as f:
        result = pickle.load(f)

    xdatas = result['xdatas']
    ydatas = result['ydatas']
    cohs   = result['cohs']
    stim_times = xdatas[0]
    n_coh  = len(xdatas)


    # Plot how the threshold varies with stimulus duration
    weibull = lambda x, a, b : 1 - 0.5*np.exp(-(x/a)**b)
    xdata = cohs

    alpha_fits = list()
    for i in range(len(stim_times)):
        ydata = ydatas[:, i]
        res = minimize(lambda param: np.sum((weibull(xdata, param[0], param[1])-ydata)**2),
                       [0.1, 1], bounds=([1e-3,1],[1e-5,10]), method='L-BFGS-B')
        alpha, beta = res.x
        alpha_fits.append(alpha)

    perfect_int = lambda x, b: -0.5*x+b
    b, _ = curve_fit(perfect_int, np.log10(stim_times), np.log10(alpha_fits))

    fs = 7
    fig = plt.figure(figsize=(2.5,1.5))
    ax = fig.add_axes([0.2,0.25,0.4,0.6])
    ax.plot(np.log10(stim_times), np.log10(alpha_fits), 'o-', color='black', label='model', markersize=3)
    ax.plot(np.log10(stim_times), -0.5*np.log10(stim_times)+b, color='red', label='perfect int.')
    ax.set_xlabel('Stimulus duration (ms)', fontsize=fs)
    ax.set_ylabel('Discrim. thr. (x0.01)', fontsize=fs)
    ax.set_xticks(np.log10(np.array([200,400,800,1600])))
    ax.set_xticklabels(['200','400','800','1600'])
    ax.set_yticks(np.log10(np.array([0.005, 0.01, 0.02, 0.04])))
    ax.set_yticklabels(['0.5','1','2', '4'])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_title(rule_name[rule], fontsize=fs)
    leg = plt.legend(fontsize=fs,frameon=False,bbox_to_anchor=[1,1], loc=2)
    # plt.locator_params(axis='y', nbins=5)
    figname = 'varytime2_'+rule_name[rule].replace(' ','')
    # figname = figname + save_name
    if save:
        plt.savefig('figure/'+figname+'.pdf', transparent=True)


    # Chronometric curve
    figname = 'varytime_'+rule_name[rule].replace(' ','')
    # figname = figname + save_name
    plot_psychometric_varytime(xdatas, ydatas, figname,
                              labels=['{:0.3f}'.format(t) for t in 2*cohs],
                              colors=sns.dark_palette("light blue", n_coh, input="xkcd"),
                              legtitle='Stim. 1 - Stim. 2',rule=rule)


def psychometric_delaychoice_varytime(save_name, rule):
    n_stim_loc = 100
    n_stim = 3
    batch_size = n_stim_loc * n_stim
    batch_shape = (n_stim_loc,n_stim)
    ind_stim_loc, ind_stim = np.unravel_index(range(batch_size),batch_shape)

    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    stim_str_range = 0.2
    cohs = stim_str_range*(2**np.arange(n_stim))/(2**(n_stim-1))
    stim1_strengths = 1 + cohs[ind_stim]
    stim2_strengths = 2 - stim1_strengths

    stim1_ons = 200
    stim1_offs = 500
    # stim1 offset and stim2 onset time difference
    dtars = np.logspace(np.log10(100), np.log10(5000), 5, dtype=int)
    # dtars = np.array([400,600,1000,1400,2000]) - 500

    params_list = list()
    for dtar in dtars:
        stim2_ons  = stim1_offs + dtar
        stim2_offs = stim2_ons + 200
        params = {'stim1_locs'    : stim1_locs,
                  'stim2_locs'    : stim2_locs,
                  'stim1_strengths' : stim1_strengths,
                  'stim2_strengths' : stim2_strengths,
                  'stim1_ons'     : stim1_ons,
                  'stim1_offs'    : stim1_offs,
                  'stim2_ons'     : stim2_ons,
                  'stim2_offs'    : stim2_offs,
                  }
        params_list.append(params)

    xdatas = [dtars] * n_stim
    ydatas = _psychometric_dm(save_name, rule, params_list, batch_shape)
    xdatas, ydatas = np.array(xdatas), np.array(ydatas).T

    figname = 'varytime_'+rule_name[rule].replace(' ','')
    plot_psychometric_varytime(xdatas, ydatas, figname,
                               labels=['{:0.3f}'.format(t) for t in 2*cohs],
                               colors=sns.dark_palette("light blue", n_stim, input="xkcd"),
                               legtitle='Stim. 1 - Stim. 2',rule=rule,
                               xlabel='Delay time (ms)')

def plot_psychometric_varytime(xdatas, ydatas, figname, labels, colors, **kwargs):
    """
    Standard function for plotting the psychometric curves
    Here the stimulus-present time is varied

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    """
    fs = 7
    fig = plt.figure(figsize=(2.5,1.5))
    ax = fig.add_axes([0.2,0.25,0.4,0.6])
    for i in range(len(xdatas)):
        xdata = xdatas[i]
        ydata = ydatas[i]
        ax.plot(xdata, ydata, 'o-', color=colors[i], label=labels[i], markersize=3)

    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    else:
        xlabel = 'Stim. Time (ms)'
    plt.xlabel(xlabel,fontsize=fs)
    plt.ylim([0.45,1.05])
    plt.yticks([0.5,0.75,1])
    if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
        plt.yticks([0.5,1],['',''])
    else:
        plt.ylabel('Performance',fontsize=fs)
    plt.title(rule_name[kwargs['rule']], fontsize=fs)
    plt.locator_params(axis='x',nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=fs)

    bbox_to_anchor = (1.0, 1)
    leg = plt.legend(title=kwargs['legtitle'],fontsize=fs,frameon=False,
                     bbox_to_anchor=bbox_to_anchor,labelspacing=0.2,loc=2)
    plt.setp(leg.get_title(),fontsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save:
        plt.savefig('figure/'+figname+'.pdf', transparent=True)
    plt.show()

################ Psychometric - Varying Stim Loc ##############################

def psychometric_choice_varyloc(save_name, **kwargs):
    print('Starting standard analysis of the CHOICE task...')
    with Run(save_name, fast_eval=fast_eval) as R:
        n_rep = 100
        n_stim = 5
        n_stim_loc = 36
        batch_size = n_rep * n_stim * n_stim_loc
        batch_shape = (n_rep,n_stim,n_stim_loc)
        ind_rep, ind_stim, ind_stim_loc = np.unravel_index(range(batch_size),batch_shape)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

        stim_str_range = 0.04
        cohs = stim_str_range*np.arange(n_stim)/(n_stim-1)
        stim1_strengths = 1 + cohs[ind_stim]
        stim2_strengths = 2 - stim1_strengths

        params = {'stim1_locs' : stim1_locs,
                  'stim2_locs' : stim2_locs,
                  'stim1_strengths' : stim1_strengths,
                  'stim2_strengths' : stim2_strengths,
                  'stim_time'    : 600}

        task  = generate_onebatch(CHOICE_MOD1, R.config, 'psychometric', params=params)
        y_sample = R.f_y_from_x(task.x)
        y_sample_loc = R.f_y_loc(y_sample)

        stim1_locs_ = np.reshape(stim1_locs, batch_shape)
        stim2_locs_ = np.reshape(stim2_locs, batch_shape)

        y_sample_loc = np.reshape(y_sample_loc[-1], batch_shape)
        choose1 = (get_dist(y_sample_loc - stim1_locs_) < 0.3*np.pi).sum(axis=0)
        choose2 = (get_dist(y_sample_loc - stim2_locs_) < 0.3*np.pi).sum(axis=0)
        ydatas = choose1/(choose1 + choose2)

    xdatas = [2*np.pi*np.arange(n_stim_loc)/n_stim_loc] * n_stim

    plot_psychometric_varyloc(xdatas, ydatas,
                              labels=['{:0.3f}'.format(t) for t in 2*cohs],
                              colors=sns.dark_palette("light blue", n_stim, input="xkcd"),
                              legtitle='Tar1 - Tar2',rule=CHOICE_MOD1, **kwargs)

def plot_psychometric_varyloc(xdatas, ydatas, labels, colors, **kwargs):
    """
    Standard function for plotting the psychometric curves
    Here the stimulus-present time is varied

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    """
    fig = plt.figure(figsize=(2,1.5))
    ax = fig.add_axes([0.25,0.25,0.65,0.65])
    for i in range(len(xdatas)):
        xdata = xdatas[i]
        ydata = ydatas[i]
        ax.plot(xdata, ydata, 'o-', color=colors[i], label=labels[i], markersize=3)

    plt.xlabel('Stim. Time (ms)',fontsize=7)
    plt.ylim([-0.05,1.05])
    plt.yticks([0, 0.5, 1])
    if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
        plt.yticks([0, 0.5,1],['',''])
    else:
        plt.ylabel('Performance',fontsize=6)
    plt.title(rule_name[kwargs['rule']], fontsize=6)
    plt.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=6)

    bbox_to_anchor = (1.0, 0.6)
    leg = plt.legend(title=kwargs['legtitle'],fontsize=6,frameon=False,
                     bbox_to_anchor=bbox_to_anchor,labelspacing=0.2)
    plt.setp(leg.get_title(),fontsize=6)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    figname = 'figure/analyze_'+rule_name[kwargs['rule']].replace(' ','') + '_varyloc'
    if 'figname_append' in kwargs:
        figname += kwargs['figname_append']

    if save:
        plt.savefig(figname+'.pdf', transparent=True)
    # plt.show()

################ Psychometric - Delay Matching Tasks ##########################

def psychometric_delaymatching(save_name, rule):
    with Run(save_name, fast_eval=True) as R:
        psychometric_delaymatching_fromsession(R, rule)

def psychometric_delaymatching_fromsession(R, rule):
    # Input is a Run session
    n_rep = 1
    n_stim_loc = 10 # increase repeat by increasing this
    batch_size = n_rep * n_stim_loc**2
    batch_shape = (n_rep, n_stim_loc,n_stim_loc)
    ind_rep, ind_stim_loc1, ind_stim_loc2 = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    stim1_locs = 2*np.pi*ind_stim_loc1/n_stim_loc
    stim2_locs = 2*np.pi*ind_stim_loc2/n_stim_loc

    params = {'stim1_locs' : stim1_locs,
              'stim2_locs' : stim2_locs}

    # rule = DMSGO
    # rule = DMCGO
    task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
    y_sample = R.f_y_from_x(task.x)

    if rule in [DMSGO, DMCGO]:
        match_response = y_sample[-1, :, 0] < 0.5 # Last time point, fixation unit, match if go
    elif rule in [DMSNOGO, DMCNOGO]:
        match_response = y_sample[-1, :, 0] > 0.5
    match_response = match_response.reshape(batch_shape)
    match_response = match_response.mean(axis=0)

    kwargs = dict()
    fs = 6
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    im = ax.imshow(match_response, cmap='BrBG', origin='lower',
                   aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('Test loc.', fontsize=fs, labelpad=-3)
    plt.xticks([0, n_stim_loc-1], ['0', '360'],
               rotation=0, va='center', fontsize=fs)
    if 'ylabel' in kwargs and kwargs['ylabel']==False:
        plt.yticks([])
    else:
        ax.set_ylabel('Sample loc.', fontsize=fs, labelpad=-3)
        plt.yticks([0, n_stim_loc-1], [0, 360],
                   rotation=0, va='center', fontsize=fs)
    # plt.title(rule_name[rule] + '\n' + lesion_group_name, fontsize=fs)
    ax.tick_params('both', length=0)
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set_visible(False)

    if 'colorbar' in kwargs and kwargs['colorbar']==False:
        pass
    else:
        ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Prop. of match', fontsize=fs, labelpad=-3)
        plt.tick_params(axis='both', which='major', labelsize=fs)

    # plt.savefig('figure/'+rule_name[rule].replace(' ','')+
    #             '_perf2D_lesion'+str(lesion_group)+
    #             self.save_name+'.pdf', transparent=True)
    plt.show()

################ Psychometric - Anti Tasks ####################################

def psychometric_goantifamily_2D(save_name, rule, title=None, **kwargs):
    n_rep = 20
    n_stim_loc = 20 # increase repeat by increasing this
    batch_size = n_rep * n_stim_loc
    batch_shape = (n_rep, n_stim_loc)
    ind_rep, ind_stim_loc = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    stim_locs = 2*np.pi*ind_stim_loc/n_stim_loc

    if rule in [REACTGO, REACTANTI]:
        params = {'stim_locs' : stim_locs}
    elif rule in [FDGO, FDANTI]:
        params = {'stim_locs' : stim_locs,
                  'stim_time' : 1000}
    elif rule in [DELAYGO, DELAYANTI]:
        params = {'stim_locs' : stim_locs,
                  'stim_ons'  : 500,
                  'stim_offs' : 800,
                  'delay_time' : 1000}
    else:
        raise ValueError('Not supported rule value')

    with Run(save_name, fast_eval=True) as R:
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
        # response locations at last time points
        y_hat_loc = R.f_y_loc_from_x(task.x)[-1]

    y_hat_loc = np.reshape(y_hat_loc, batch_shape)
    stim_locs_ = np.reshape(stim_locs, batch_shape)[0,:]
    bins = np.concatenate((stim_locs_, np.array([2*np.pi])))
    responses = np.zeros((n_stim_loc, n_stim_loc))

    # Looping over input locations
    for i in range(n_stim_loc):
        hist, bins_edge = np.histogram(y_hat_loc[:,i], bins=bins)
        responses[:,i] = hist/n_rep


    fs = 6
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    im = ax.imshow(responses, cmap='hot', origin='lower',
                   aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('input loc.', fontsize=fs, labelpad=-3)
    plt.xticks([0, n_stim_loc-1], ['0', '360'],
               rotation=0, va='center', fontsize=fs)
    ax.set_ylabel('output loc.', fontsize=fs, labelpad=-3)
    plt.yticks([0, n_stim_loc-1], ['0', '360'],
               rotation=0, va='center', fontsize=fs)
    if title is not None:
        ax.set_title(title, fontsize=fs)
    ax.tick_params('both', length=0)
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set_visible(False)

    if 'colorbar' in kwargs and kwargs['colorbar']==False:
        pass
    else:
        ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
        cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Prop. of responses', fontsize=fs, labelpad=-3)
        plt.tick_params(axis='both', which='major', labelsize=fs)

    if save:
        if 'save_name' not in kwargs:
            save_name = rule_name[rule].replace(' ','')+'_perf2D.pdf'
        else:
            save_name = kwargs['save_name']
        plt.savefig(os.path.join('figure', save_name), transparent=True)

    plt.show()



if __name__ == '__main__':
    pass
    save_name = '0_256migrate'

    # plot_trainingprogress(save_name)
    # plot_performanceprogress(save_name)
    
    # plot_trainingprogress_cont('cont_allrule_2_0_0_1intsynmain')
    # plot_trainingprogress_cont('cont_allrule_4_0_1_1intsynthu')

    # plot_finalperformance('allrule_softplus', '_300test')
    # plot_finalperformance('cont_allrule', '_1_0_1intsynthu')
    # plot_finalperformance('oicdmconly_strongnoise')
    # plot_finalperformance_lr()

    ################## Continual Learning Performance #########################
    # get_allperformance('cont_allrule', 'intsynrelu')
    # plot_performanceprogress_cont(('cont_allrule_4_0_0_0intsynrelu',
    #                                'cont_allrule_4_0_1_1intsynrelu'))
    # plot_finalperformance_cont('cont_allrule', '_0_0_0intsynrelu', '_0_1_1intsynrelu')

    ################## Psychometric Performance ###############################
    psychometric_choice(save_name)
    # psychometric_choiceattend(save_name, no_ylabel=True)
