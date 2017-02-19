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
from task import *
from network import get_perf
from run import Run

rule_color={GO                : sns.xkcd_palette(['green'])[0],
            DELAYGO           : sns.xkcd_palette(['olive'])[0],
            INHGO             : sns.xkcd_palette(['light green'])[0],
            REMAP             : sns.xkcd_palette(['mustard'])[0],
            DELAYREMAP        : sns.xkcd_palette(['tan'])[0],
            INHREMAP          : sns.xkcd_palette(['yellow'])[0],
            CHOICE_MOD1       : sns.xkcd_palette(['lavender'])[0],
            CHOICE_MOD2       : sns.xkcd_palette(['aqua'])[0],
            CHOICEATTEND_MOD1 : sns.xkcd_palette(['bright purple'])[0],
            CHOICEATTEND_MOD2 : sns.xkcd_palette(['cyan'])[0],
            CHOICE_INT        : sns.xkcd_palette(['blue'])[0],
            CHOICEDELAY_MOD1  : sns.xkcd_palette(['violet'])[0],
            CHOICEDELAY_MOD2  : sns.xkcd_palette(['teal'])[0],
            DMSGO             : sns.xkcd_palette(['red'])[0],
            DMSNOGO           : sns.xkcd_palette(['rose'])[0],
            DMCGO             : sns.xkcd_palette(['orange'])[0],
            DMCNOGO           : sns.xkcd_palette(['peach'])[0]
            }


# If True, will evaluate the network with larger time steps
fast_eval = True

save = False

def plot_trainingprogress(save_addon, rule_plot=None, save=True):
    # Plot Training Progress
    with open('data/config'+save_addon+'.pkl', 'rb') as f:
        config = pickle.load(f)

    trials      = config['trials']
    times       = config['times']
    cost_tests  = config['cost_tests']
    perf_tests  = config['perf_tests']

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

    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax2.tick_params(axis='both', which='major', labelsize=fs)

    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('Total trials (1,000)',fontsize=fs, labelpad=2)
    ax2.set_ylabel('performance',fontsize=fs)
    ax1.set_ylabel(r'$log_{10}$(cost)',fontsize=fs)
    ax1.set_xticklabels([])
    ax1.set_title('Training time {:0.1f} hours'.format(times[-1]/3600.),fontsize=fs)
    ax1.locator_params(axis='y', nbins=3)
    ax2.locator_params(axis='y', nbins=4)
    lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.7,0.5),
                    fontsize=fs,labelspacing=0.3,loc=6)
    plt.setp(lg.get_title(),fontsize=fs)
    if save:
        plt.savefig('figure/Training_Progress'+config['save_addon']+'.pdf', transparent=True)
    plt.show()

def plot_finalperformance(save_type):
    HDIMs = range(1000)
    # Initialization. Dictionary comprehension.
    for HDIM in HDIMs:
        save_addon = save_type+'_'+str(HDIM)
        fname = 'data/config'+save_addon+'.pkl'
        if os.path.isfile(fname):
            break

    with open('data/config'+save_addon+'.pkl','rb') as f:
        config = pickle.load(f)
    cost_tests  = config['cost_tests']
    perf_tests  = config['perf_tests']

    final_cost = {k: [] for k in cost_tests}
    final_perf = {k: [] for k in cost_tests}
    HDIM_plot = list()
    training_time_plot = list()
    # Recording performance and cost for networks

    for HDIM in HDIMs:
        save_addon = save_type+'_'+str(HDIM)
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue
        with open(fname,'rb') as f:
            config = pickle.load(f)
        cost_tests = config['cost_tests']
        perf_tests = config['perf_tests']
        # if perf_tests[DMCGO][-1] > 0.1:
        for key in perf_tests.keys():
            final_perf[key] += [float(perf_tests[key][-1])]
            final_cost[key]        += [float(cost_tests[key][-1])]
        HDIM_plot.append(HDIM)
        training_time_plot.append(config['times'][-1])

    n_trial = config['trials'][-1]
    x_plot = np.array(HDIM_plot)
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
        # line = ax1.plot(x_plot,np.log10(final_cost[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot,final_perf[rule],color=color_rules[i%26])
        line = ax1.plot(x_plot,np.log10(final_cost[rule]),color=rule_color[rule])
        ax2.plot(x_plot,final_perf[rule],color=rule_color[rule])
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)

    ax2.set_ylim(top=1.05)
    ax2.set_xlabel('Number of Recurrent Units',fontsize=7)
    ax2.set_ylabel('performance',fontsize=7)
    ax1.set_ylabel(r'$log_{10}$(cost)',fontsize=7)
    ax1.set_xticklabels([])
    ax1.set_title('After {:.1E} trials'.format(n_trial),fontsize=7)
    ax1.locator_params(axis='y', nbins=3)
    ax2.locator_params(axis='y', nbins=4)
    lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.65,0.5),
                    fontsize=7,labelspacing=0.3,loc=6)
    plt.setp(lg.get_title(),fontsize=7)
    plt.savefig('figure/FinalCostPerformance_'+save_type+'.pdf', transparent=True)
    plt.show()

    x_plot = np.array(HDIM_plot)/100.
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

def plot_finalperformance_lr():
    # Specialized for varyign learning rate. Can be rewritten, but I'm lazy.
    n_lr = 100
    save_type = 'allrule_weaknoise'
    save_addon = save_type+'_lr1'
    with open('data/config'+save_addon+'.pkl','rb') as f:
        config = pickle.load(f)
    cost_tests  = config['cost_tests']
    perf_tests  = config['perf_tests']

    final_cost = {k: [] for k in cost_tests}
    final_perf = {k: [] for k in cost_tests}
    lr_plot = list()
    training_time_plot = list()
    # Recording performance and cost for networks
    for i_lr in range(n_lr):
        save_addon = save_type+'_lr'+str(i_lr)
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue
        with open(fname,'rb') as f:
            config = pickle.load(f)
        cost_tests = config['cost_tests']
        perf_tests = config['perf_tests']
        # if perf_tests[DMCGO][-1] > 0.1:
        for key in perf_tests.keys():
            final_perf[key] += [float(perf_tests[key][-1])]
            final_cost[key]        += [float(cost_tests[key][-1])]
        # lr_plot.append(config['learning_rate'])
        lr_plot.append(np.logspace(-2,-4,100)[i_lr]) # Temporary
        training_time_plot.append(config['times'][-1])

    n_trial = config['trials'][-1]
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

def psychometric_choice(save_addon, **kwargs):
    print('Starting standard analysis of the CHOICE task...')
    with Run(save_addon, fast_eval=fast_eval) as R:

        tar_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.1
        n_tar_loc = 300
        n_tar = len(tar_cohs)
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar1_strengths = 1 + tar_cohs[ind_tar]
        tar2_strengths = 1 - tar_cohs[ind_tar]

        ydatas = list()
        tar_times = [100, 400, 1600]
        for tar_time in tar_times:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_strengths' : tar1_strengths,
                      'tar2_strengths' : tar2_strengths,
                      'tar_time'    : tar_time}

            task  = generate_onebatch(CHOICE_MOD1, R.config, 'psychometric', params=params)
            y_loc_sample = R.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            ydatas.append(choose1/(choose1 + choose2))

        xdatas = [tar_cohs*2] * len(tar_times)

    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in tar_times],
                              colors=sns.dark_palette("light blue", 3, input="xkcd"),
                              legtitle='Stim. time (ms)',rule=CHOICE_MOD1, **kwargs)

def psychometric_delaychoice(save_addon, **kwargs):
    print('Starting standard analysis of the CHOICEDELAY task...')
    with Run(save_addon, fast_eval=fast_eval) as R:
        n_tar_loc = 120
        n_tar = 7
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.75
        tar1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar/(n_tar-1)
        tar2_strengths = 2 - tar1_strengths

        tar1_ons = 200
        tar1_offs = 400

        dtars = [200,1500,3000] # tar1 offset and tar2 onset time difference
        # dtars = [2800,3000,3200] # tar1 offset and tar2 onset time difference
        ydatas = list()
        for dtar in dtars:
            tar2_ons  = tar1_offs + dtar
            tar2_offs = tar2_ons + 200
            params = {'tar1_locs'    : tar1_locs,
                      'tar2_locs'    : tar2_locs,
                      'tar1_strengths' : tar1_strengths,
                      'tar2_strengths' : tar2_strengths,
                      'tar1_ons'     : tar1_ons,
                      'tar1_offs'    : tar1_offs,
                      'tar2_ons'     : tar2_ons,
                      'tar2_offs'    : tar2_offs,
                      }

            task  = generate_onebatch(CHOICEDELAY_MOD1, R.config, 'psychometric', params=params)
            y_loc_sample = R.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            print(choose1)
            print(choose2)
            ydatas.append(choose1/(choose1 + choose2))

        xdatas = [tar_str_range*(-1+2*np.arange(n_tar)/(n_tar-1))] * len(dtars)

    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in dtars],
                              colors=sns.dark_palette("light blue", 3, input="xkcd"),
                              legtitle='Delay (ms)',rule=CHOICEDELAY_MOD1, **kwargs)

def psychometric_choiceattend(save_addon, **kwargs):
    psychometric_choiceattend_(save_addon, CHOICEATTEND_MOD1, **kwargs)
    psychometric_choiceattend_(save_addon, CHOICEATTEND_MOD2, **kwargs)

def psychometric_choiceattend_(save_addon, rule, **kwargs):
    print('Starting standard analysis of the {:s} task...'.format(rule_name[rule]))
    with Run(save_addon, fast_eval=fast_eval) as R:

        print('Using temporary rule setup')
        from run import replacerule
        rule_X = np.array([CHOICE_INT, CHOICE_MOD1])
        # beta = np.array([1,0])
        # beta = np.array([-1,2])
        beta = np.array([0,1])
        # beta = np.array([0,2])
        # beta = np.array([-4,5])
        beta = replacerule(R, rule, rule_X, beta)

        from task import get_dist

        tar_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.2
        # tar_cohs = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])*0.5

        n_tar_loc = 100 # increase repeat by increasing this
        n_tar = len(tar_cohs)
        batch_size = n_tar_loc * n_tar**2
        batch_shape = (n_tar_loc,n_tar,n_tar)
        ind_tar_loc, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

        # Looping target location
        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_mod1_cohs = tar_cohs[ind_tar_mod1]
        tar_mod2_cohs = tar_cohs[ind_tar_mod2]

        params = {'tar1_locs' : tar1_locs,
                  'tar2_locs' : tar2_locs,
                  'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
                  'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
                  'tar1_mod2_strengths' : 1 + tar_mod2_cohs,
                  'tar2_mod2_strengths' : 1 - tar_mod2_cohs,
                  'tar_time'    : 800}

        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
        y_sample = R.f_y_from_x(task.x)
        y_sample_loc = R.f_y_loc(y_sample)
        perf = get_perf(y_sample, task.y_loc)
        print('Performance {:0.3f}'.format(np.mean(perf)))


        tar1_locs_ = np.reshape(tar1_locs, batch_shape)
        tar2_locs_ = np.reshape(tar2_locs, batch_shape)

        y_sample_loc = np.reshape(y_sample_loc[-1], batch_shape)
        choose1 = (get_dist(y_sample_loc - tar1_locs_) < 0.3*np.pi).sum(axis=0)
        choose2 = (get_dist(y_sample_loc - tar2_locs_) < 0.3*np.pi).sum(axis=0)
        prop1s = choose1/(choose1 + choose2)

        xdatas = [tar_cohs*2]*2
        ydatas = [prop1s.mean(axis=k) for k in [1,0]]

        labels = ['Attend', 'Ignore'] if rule==CHOICEATTEND_MOD1 else ['Ignore', 'Attend']

        plot_psychometric_choice(xdatas, ydatas,
                                  labels=labels,
                                  colors=sns.color_palette("Set2",2),
                                  legtitle='Modality',rule=rule, **kwargs)

def psychometric_choiceint(save_addon, **kwargs):
    print('Starting standard analysis of the CHOICEINT task...')
    with Run(save_addon, fast_eval=fast_eval) as R:

        tar_cohs = np.array([-0.5, -0.15, -0.05, 0, 0.05, 0.15, 0.5])*0.1
        n_tar_loc = 500 # increase repeat by increasing this
        n_tar = len(tar_cohs)
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar1_strength = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)


        tar1_strengths = 1 + tar_cohs[ind_tar1_strength]
        tar2_strengths = 1 - tar_cohs[ind_tar1_strength]

        xdatas = list()
        ydatas = list()
        for mod_strength in [(1,0), (0,1), (1,1)]:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_mod1_strengths' : tar1_strengths*mod_strength[0],
                      'tar2_mod1_strengths' : tar2_strengths*mod_strength[0],
                      'tar1_mod2_strengths' : tar1_strengths*mod_strength[1],
                      'tar2_mod2_strengths' : tar2_strengths*mod_strength[1],
                      'tar_time'    : 800}

            task  = generate_onebatch(CHOICE_INT, R.config, 'psychometric', params=params)
            y_loc_sample = R.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            prop1s = choose1/(choose1 + choose2)

            xdatas.append(tar_cohs*2)
            ydatas.append(prop1s)

        fits = plot_psychometric_choice(
            xdatas,ydatas, labels=['1 only', '2 only', 'both'],
            colors=sns.color_palette("Set2",3),
            legtitle='Modality',rule=CHOICE_INT, **kwargs)
        sigmas = [fit[1] for fit in fits]
        print('Fit sigmas:')
        print(sigmas)

def psychometric_intrepro(save_addon):
    with Run(save_addon, fast_eval=fast_eval) as R:

        n_tar_loc = 360
        # intervals = [700]
        # intervals = [500, 600, 700, 800, 900, 1000]
        intervals = np.linspace(500, 1000, 10)
        mean_sample_intervals = list()
        for interval in intervals:
            batch_size = n_tar_loc
            tar_mod1_locs  = 2*np.pi*np.arange(n_tar_loc)/n_tar_loc

            params = {'tar_mod1_locs'  : tar_mod1_locs,
                      'interval'       : interval}

            task  = generate_onebatch(INTREPRO, R.config, 'psychometric', params=params)
            h_test = R.f_h(task.x)
            y = R.f_y(h_test)

            sample_intervals = list() # sampled interval test
            for i_batch in range(batch_size):
                try: ##TODO: Temporary solution
                    # Setting the threshold can be tricky, but doesn't impact the overall results
                    sample_interval = np.argwhere(y[:,i_batch,0]<0.3)[0]-task.epochs['tar2'][1]
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
    '''
    Standard function for plotting the psychometric curves

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    '''
    fig = plt.figure(figsize=(2,1.5))
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

    plt.xlabel('Target 1 - Target 2',fontsize=7)
    plt.ylim([-0.05,1.05])
    plt.xlim([xdata[0]*1.1,xdata[-1]*1.1])
    plt.yticks([0,0.5,1])
    if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
        plt.yticks([0,0.5,1],['','',''])
    else:
        plt.ylabel('Proportion of choice 1',fontsize=7)
    plt.title(rule_name[kwargs['rule']], fontsize=7)
    plt.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=7)

    if len(xdatas)>1:
        if len(kwargs['legtitle'])>10:
            bbox_to_anchor = (0.6, 1.1)
        else:
            bbox_to_anchor = (0.5, 1.1)
        leg = plt.legend(title=kwargs['legtitle'],fontsize=7,frameon=False,
                         bbox_to_anchor=bbox_to_anchor,labelspacing=0.3)
        plt.setp(leg.get_title(),fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    savename = 'figure/analyze_'+rule_name[kwargs['rule']].replace(' ','') + '_performance'
    if 'savename_append' in kwargs:
        savename += kwargs['savename_append']

    if save:
        plt.savefig(savename+'.pdf', transparent=True)
    plt.show()
    return fits

def psychometric_choicefamily_2D(save_addon, rule, lesion_units=None,
                                 n_coh=8, n_tar_loc=20, coh_range=0.2):
    # Generate task parameters for choice tasks
    # coh_range = 0.2
    # coh_range = 0.05
    cohs = np.linspace(-coh_range, coh_range, n_coh)

    batch_size = n_tar_loc * n_coh**2
    batch_shape = (n_tar_loc,n_coh,n_coh)
    ind_tar_loc, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
    tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

    tar_mod1_cohs = cohs[ind_tar_mod1]
    tar_mod2_cohs = cohs[ind_tar_mod2]

    params_dict = dict()
    params_dict[CHOICE_MOD1] = \
         {'tar1_locs' : tar1_locs,
          'tar2_locs' : tar2_locs,
          'tar1_strengths' : 1 + tar_mod1_cohs, # Just use mod 1 value
          'tar2_strengths' : 1 - tar_mod1_cohs,
          'tar_time'    : 800
          }
    params_dict[CHOICE_MOD2] = params_dict[CHOICE_MOD1]

    params_dict[CHOICEATTEND_MOD1] = \
         {'tar1_locs' : tar1_locs,
          'tar2_locs' : tar2_locs,
          'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
          'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
          'tar1_mod2_strengths' : 1 + tar_mod2_cohs,
          'tar2_mod2_strengths' : 1 - tar_mod2_cohs,
          'tar_time'    : 800
          }

    params_dict[CHOICEATTEND_MOD2] = params_dict[CHOICEATTEND_MOD1]

    params_dict[CHOICEDELAYATTEND_MOD1] = params_dict[CHOICEATTEND_MOD1]
    params_dict[CHOICEDELAYATTEND_MOD1]['tar_time'] = 00
    params_dict[CHOICEDELAYATTEND_MOD2] = params_dict[CHOICEDELAYATTEND_MOD1]

    params_dict[CHOICE_INT] = \
         {'tar1_locs' : tar1_locs,
          'tar2_locs' : tar2_locs,
          'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
          'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
          'tar1_mod2_strengths' : 1 + tar_mod1_cohs, # Same as Mod 1
          'tar2_mod2_strengths' : 1 - tar_mod1_cohs,
          'tar_time'    : 800
          }

    params_dict[CHOICEDELAYATTEND_MOD1] = \
         {'tar1_locs' : tar1_locs,
          'tar2_locs' : tar2_locs,
          'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
          'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
          'tar1_mod2_strengths' : 1 + tar_mod2_cohs,
          'tar2_mod2_strengths' : 1 - tar_mod2_cohs,
          'tar_time'    : 800
          }

    with Run(save_addon, lesion_units=lesion_units, fast_eval=True) as R:

        params = params_dict[rule]
        # task  = generate_onebatch(rule, R.config, 'psychometric', params=params)

        print('Using temporary rule setup')
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params,
                                  add_rule=[CHOICEDELAY_MOD2, CHOICEATTEND_MOD2, CHOICE_INT],
                                  rule_strength=[0., 1., -0.])

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
    tar1_locs_ = np.reshape(tar1_locs, batch_shape)
    tar2_locs_ = np.reshape(tar2_locs, batch_shape)

    y_sample_loc = np.reshape(y_sample_loc[-1], batch_shape)
    choose1 = (get_dist(y_sample_loc - tar1_locs_) < 0.3*np.pi).sum(axis=0)
    choose2 = (get_dist(y_sample_loc - tar2_locs_) < 0.3*np.pi).sum(axis=0)
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

def plot_psychometric_choicefamily_2D(save_addon, rule, **kwargs):
    perf, prop1s, cohs = psychometric_choicefamily_2D(save_addon, rule, **kwargs)
    _plot_psychometric_choicefamily_2D(prop1s, cohs, rule)


################ Psychometric - Varying Stim Time #############################

def compute_choicefamily_varytime(save_addon, rule):
    assert rule in [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
    print('Starting vary time analysis of the {:s} task...'.format(rule_name[rule]))
    with Run(save_addon, fast_eval=fast_eval) as R:
        n_tar_loc = 3000
        n_coh = 3
        batch_size = n_tar_loc * n_coh
        batch_shape = (n_tar_loc,n_coh)
        ind_tar_loc, ind_tar = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        if rule == CHOICE_INT:
            tar_str_range = 0.02
        else:
            tar_str_range = 0.04

        cohs = tar_str_range*(2**np.arange(n_coh))/(2**(n_coh-1))
        tar1_strengths = 1 + cohs[ind_tar]
        tar2_strengths = 2 - tar1_strengths

        ydatas = list()
        tar_times = np.logspace(np.log10(200), np.log10(1500), 8, dtype=int)
        for tar_time in tar_times:
            if rule in [CHOICE_MOD1, CHOICE_MOD2]:
                params = {'tar1_locs' : tar1_locs,
                          'tar2_locs' : tar2_locs,
                          'tar1_strengths' : tar1_strengths,
                          'tar2_strengths' : tar2_strengths,
                          'tar_time'    : tar_time}

            elif rule == CHOICE_INT:
                params = {'tar1_locs' : tar1_locs,
                          'tar2_locs' : tar2_locs,
                          'tar1_mod1_strengths' : 1 + cohs[ind_tar],
                          'tar2_mod1_strengths' : 1 - cohs[ind_tar],
                          'tar1_mod2_strengths' : 1 + cohs[ind_tar],
                          'tar2_mod2_strengths' : 1 - cohs[ind_tar],
                          'tar_time'    : tar_time}

            elif rule in [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]:
                if rule == CHOICEATTEND_MOD1:
                    att = '1'
                    ign = '2'
                else:
                    att = '2'
                    ign = '1'
                params = {'tar1_locs' : tar1_locs,
                          'tar2_locs' : tar2_locs,
                          'tar1_mod'+att+'_strengths' : 1 + cohs[ind_tar],
                          'tar2_mod'+att+'_strengths' : 1 - cohs[ind_tar],
                          'tar1_mod'+ign+'_strengths' : np.ones(batch_size),
                          'tar2_mod'+ign+'_strengths' : np.ones(batch_size),
                          'tar_time'    : tar_time}


            task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
            y_loc_sample = R.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            ydatas.append(choose1/(choose1 + choose2))

    xdatas = [tar_times] * n_coh
    ydatas = np.array(ydatas).T

    result = {'xdatas' : xdatas, 'ydatas' : ydatas, 'cohs' : cohs}

    savename = 'data/varytime_'+rule_name[rule].replace(' ','') +save_addon
    with open(savename+'.pkl','wb') as f:
        pickle.dump(result, f)

def plot_choicefamily_varytime(save_addon, rule):
    assert rule in [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
    savename = 'data/varytime_'+rule_name[rule].replace(' ','') +save_addon

    with open(savename+'.pkl','rb') as f:
        result = pickle.load(f)

    xdatas = result['xdatas']
    ydatas = result['ydatas']
    cohs   = result['cohs']
    tar_times = xdatas[0]
    n_coh  = len(xdatas)


    # Plot how the threshold varies with stimulus duration
    weibull = lambda x, a, b : 1 - 0.5*np.exp(-(x/a)**b)
    xdata = cohs

    alpha_fits = list()
    for i in range(len(tar_times)):
        ydata = ydatas[:, i]
        res = minimize(lambda param: np.sum((weibull(xdata, param[0], param[1])-ydata)**2),
                       [0.1, 1], bounds=([1e-3,1],[1e-5,10]), method='L-BFGS-B')
        alpha, beta = res.x
        alpha_fits.append(alpha)

    perfect_int = lambda x, b: -0.5*x+b
    b, _ = curve_fit(perfect_int, np.log10(tar_times), np.log10(alpha_fits))

    fs = 7
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.2, 0.3, 0.7, 0.6])
    ax.plot(np.log10(tar_times), np.log10(alpha_fits), 'o-', color='black', label='model', markersize=3)
    ax.plot(np.log10(tar_times), -0.5*np.log10(tar_times)+b, color='red', label='perfect int.')
    ax.set_xlabel('Stimulus duration (ms)', fontsize=fs)
    ax.set_ylabel('Log threshold', fontsize=fs)
    ax.set_xticks(np.log10(np.array([200,400,800,1600])))
    ax.set_xticklabels(['200','400','800','1600'])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_title(rule_name[rule], fontsize=fs)
    leg = plt.legend(fontsize=fs,frameon=False,bbox_to_anchor=[1,1])
    plt.locator_params(axis='y', nbins=5)
    savename = 'varytime2_'+rule_name[rule].replace(' ','') +save_addon
    if save:
        plt.savefig('figure/'+savename+'.pdf', transparent=True)


    # Chronometric curve
    savename = 'varytime_'+rule_name[rule].replace(' ','') +save_addon
    plot_psychometric_varytime(xdatas, ydatas, savename,
                              labels=['{:0.3f}'.format(t) for t in 2*cohs],
                              colors=sns.dark_palette("light blue", n_coh, input="xkcd"),
                              legtitle='Tar1 - Tar2',rule=rule)


def psychometric_delaychoice_varytime(save_addon, **kwargs):
    print('Starting standard analysis of the DELAY CHOICE task...')
    with Run(save_addon, fast_eval=fast_eval) as R:
        n_tar_loc = 300
        n_tar = 1
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.2
        cohs = tar_str_range*(2**np.arange(n_tar))/(2**(n_tar-1))
        tar1_strengths = 1 + cohs[ind_tar]
        tar2_strengths = 2 - tar1_strengths

        tar1_ons = 200
        tar1_offs = 500
        # tar1 offset and tar2 onset time difference
        dtars = np.logspace(np.log10(100), np.log10(5000), 5, dtype=int)
        # dtars = np.array([400,600,1000,1400,2000]) - 500
        ydatas = list()
        for dtar in dtars:
            tar2_ons  = tar1_offs + dtar
            tar2_offs = tar2_ons + 200
            params = {'tar1_locs'    : tar1_locs,
                      'tar2_locs'    : tar2_locs,
                      'tar1_strengths' : tar1_strengths,
                      'tar2_strengths' : tar2_strengths,
                      'tar1_ons'     : tar1_ons,
                      'tar1_offs'    : tar1_offs,
                      'tar2_ons'     : tar2_ons,
                      'tar2_offs'    : tar2_offs,
                      }

            task  = generate_onebatch(CHOICEDELAY_MOD1, R.config, 'psychometric', params=params)
            # y_loc_sample = R.f_y_loc_from_x(task.x)
            y_sample = R.f_y_from_x(task.x)
            y_loc_sample = R.f_y_loc(y_sample)
            perf = get_perf(y_sample, task.y_loc)
            print('Performance {:0.3f}'.format(np.mean(perf)))

            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            ydatas.append(choose1/(choose1 + choose2))

        xdatas = [dtars] * n_tar
        ydatas = np.array(ydatas).T


    plot_psychometric_varytime(xdatas, ydatas,
                               labels=['{:0.3f}'.format(t) for t in 2*cohs],
                               colors=sns.dark_palette("light blue", n_tar, input="xkcd"),
                               legtitle='Tar1 - Tar2',rule=CHOICEDELAY_MOD1,
                               xlabel='Delay time (ms)', **kwargs)

def plot_psychometric_varytime(xdatas, ydatas, savename, labels, colors, **kwargs):
    '''
    Standard function for plotting the psychometric curves
    Here the stimulus-present time is varied

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    '''
    fs = 7
    fig = plt.figure(figsize=(2,1.5))
    ax = fig.add_axes([0.25,0.25,0.65,0.65])
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

    bbox_to_anchor = (1.0, 0.55)
    leg = plt.legend(title=kwargs['legtitle'],fontsize=fs,frameon=False,
                     bbox_to_anchor=bbox_to_anchor,labelspacing=0.2)
    plt.setp(leg.get_title(),fontsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save:
        plt.savefig('figure/'+savename+'.pdf', transparent=True)
    # plt.show()

################ Psychometric - Varying Stim Loc ##############################

def psychometric_choice_varyloc(save_addon, **kwargs):
    print('Starting standard analysis of the CHOICE task...')
    with Run(save_addon, fast_eval=fast_eval) as R:
        n_rep = 100
        n_tar = 5
        n_tar_loc = 36
        batch_size = n_rep * n_tar * n_tar_loc
        batch_shape = (n_rep,n_tar,n_tar_loc)
        ind_rep, ind_tar, ind_tar_loc = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.04
        cohs = tar_str_range*np.arange(n_tar)/(n_tar-1)
        tar1_strengths = 1 + cohs[ind_tar]
        tar2_strengths = 2 - tar1_strengths

        params = {'tar1_locs' : tar1_locs,
                  'tar2_locs' : tar2_locs,
                  'tar1_strengths' : tar1_strengths,
                  'tar2_strengths' : tar2_strengths,
                  'tar_time'    : 600}

        task  = generate_onebatch(CHOICE_MOD1, R.config, 'psychometric', params=params)
        y_sample = R.f_y_from_x(task.x)
        y_sample_loc = R.f_y_loc(y_sample)

        tar1_locs_ = np.reshape(tar1_locs, batch_shape)
        tar2_locs_ = np.reshape(tar2_locs, batch_shape)

        y_sample_loc = np.reshape(y_sample_loc[-1], batch_shape)
        choose1 = (get_dist(y_sample_loc - tar1_locs_) < 0.3*np.pi).sum(axis=0)
        choose2 = (get_dist(y_sample_loc - tar2_locs_) < 0.3*np.pi).sum(axis=0)
        ydatas = choose1/(choose1 + choose2)

    xdatas = [2*np.pi*np.arange(n_tar_loc)/n_tar_loc] * n_tar

    plot_psychometric_varyloc(xdatas, ydatas,
                              labels=['{:0.3f}'.format(t) for t in 2*cohs],
                              colors=sns.dark_palette("light blue", n_tar, input="xkcd"),
                              legtitle='Tar1 - Tar2',rule=CHOICE_MOD1, **kwargs)

def plot_psychometric_varyloc(xdatas, ydatas, labels, colors, **kwargs):
    '''
    Standard function for plotting the psychometric curves
    Here the stimulus-present time is varied

    xdatas, ydatas, labels, and colors are all lists. Each list contains
    properties for each curve.
    '''
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
    savename = 'figure/analyze_'+rule_name[kwargs['rule']].replace(' ','') + '_varyloc'
    if 'savename_append' in kwargs:
        savename += kwargs['savename_append']

    if save:
        plt.savefig(savename+'.pdf', transparent=True)
    # plt.show()

################ Psychometric - Delay Matching Tasks ##########################

def psychometric_delaymatching(save_addon, rule):
    with Run(save_addon, fast_eval=True) as R:
        psychometric_delaymatching_fromsession(R, rule)

def psychometric_delaymatching_fromsession(R, rule):
    # Input is a Run session
    n_rep = 1
    n_tar_loc = 10 # increase repeat by increasing this
    batch_size = n_rep * n_tar_loc**2
    batch_shape = (n_rep, n_tar_loc,n_tar_loc)
    ind_rep, ind_tar_loc1, ind_tar_loc2 = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    tar1_locs = 2*np.pi*ind_tar_loc1/n_tar_loc
    tar2_locs = 2*np.pi*ind_tar_loc2/n_tar_loc

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs}

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
    plt.xticks([0, n_tar_loc-1], ['0', '360'],
               rotation=0, va='center', fontsize=fs)
    if 'ylabel' in kwargs and kwargs['ylabel']==False:
        plt.yticks([])
    else:
        ax.set_ylabel('Sample loc.', fontsize=fs, labelpad=-3)
        plt.yticks([0, n_tar_loc-1], [0, 360],
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
    #             self.save_addon+'.pdf', transparent=True)
    plt.show()

################ Psychometric - Anti Tasks ####################################

def psychometric_goantifamily_2D(save_addon, rule, title=None, **kwargs):
    n_rep = 20
    n_tar_loc = 20 # increase repeat by increasing this
    batch_size = n_rep * n_tar_loc
    batch_shape = (n_rep, n_tar_loc)
    ind_rep, ind_tar_loc = np.unravel_index(range(batch_size),batch_shape)

    # Looping target location
    tar_locs = 2*np.pi*ind_tar_loc/n_tar_loc

    if rule in [GO, REMAP]:
        params = {'tar_locs' : tar_locs}
    elif rule in [INHGO, INHREMAP]:
        params = {'tar_locs' : tar_locs,
                  'tar_time' : 1000}
    elif rule in [DELAYGO, DELAYREMAP]:
        params = {'tar_locs' : tar_locs,
                  'tar_ons'  : 500,
                  'tar_offs' : 800,
                  'delay_time' : 1000}
    else:
        raise ValueError('Not supported rule value')

    with Run(save_addon, fast_eval=True) as R:
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
        # response locations at last time points
        y_hat_loc = R.f_y_loc_from_x(task.x)[-1]

    y_hat_loc = np.reshape(y_hat_loc, batch_shape)
    tar_locs_ = np.reshape(tar_locs, batch_shape)[0,:]
    bins = np.concatenate((tar_locs_, np.array([2*np.pi])))
    responses = np.zeros((n_tar_loc, n_tar_loc))

    # Looping over input locations
    for i in range(n_tar_loc):
        hist, bins_edge = np.histogram(y_hat_loc[:,i], bins=bins)
        responses[:,i] = hist/n_rep


    fs = 6
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    im = ax.imshow(responses, cmap='hot', origin='lower',
                   aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel('input loc.', fontsize=fs, labelpad=-3)
    plt.xticks([0, n_tar_loc-1], ['0', '360'],
               rotation=0, va='center', fontsize=fs)
    ax.set_ylabel('output loc.', fontsize=fs, labelpad=-3)
    plt.yticks([0, n_tar_loc-1], ['0', '360'],
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
    # plot_trainingprogress('allrule_tanh_340')
    # plot_trainingprogress('oicdmconly_strongnoise_200')
    # plot_finalperformance('allrule_relu')
    # plot_finalperformance('oicdmconly_strongnoise')
    # plot_finalperformance_lr()
    
    # psychometric_choice('allrule_weaknoise_400')
    # psychometric_choiceattend('allrule_weaknoise_400', no_ylabel=True)
    # psychometric_choiceattend_varytime('tf_withrecnoise_400')
    # psychometric_choiceint('allrule_weaknoise_400', no_ylabel=True)
    # psychometric_delaychoice('tf_withrecnoise_400')

    # save_addon = 'delaychoiceonly_weaknoise_140'
    # save_addon = 'allrule_weaknoise_360' # This works in all three rules
    # save_addon = 'allrule_weaknoise_480' # This works as well
    # save_addon = 'allrule_weaknoise_500'
    # save_addon = 'allrule_weaknoise_400'
    save_addon = 'allrule_softplus_380'
    # save_addon = 'choicefamily_softplus_220'
    # save_addon = 'attendonly_weaknoise_500'
    # for rule in [CHOICEATTEND_MOD1]:
    for rule in [CHOICE_MOD1, CHOICEATTEND_MOD1, CHOICE_INT]:
        pass
        # compute_choicefamily_varytime(save_addon, rule)
        # plot_choicefamily_varytime(save_addon, rule)

    for rule in [CHOICEDELAYATTEND_MOD2]:
        pass
        # plot_psychometric_choicefamily_2D(save_addon, rule, n_tar_loc=20, coh_range = 0.6)

    # psychometric_choiceattend_(save_addon, CHOICEATTEND_MOD1)

    # compute_psychometric_choice_varytime(save_addon, savename_append=save_addon)
    # plot_psychometric_choice_varytime(savename_append=save_addon)
    # compute_psychometric_choiceattend_varytime_(save_addon, CHOICEATTEND_MOD1, savename_append=save_addon)
    # plot_psychometric_choiceattend_varytime_(CHOICEATTEND_MOD1, savename_append=save_addon)
    # psychometric_choiceattend_varytime_(save_addon, CHOICEATTEND_MOD1, savename_append=save_addon)
    # psychometric_choiceattend_varytime('attendonly_weaknoise_200')
    # psychometric_choiceint_varytime('allrule_weaknoise_200')
    # psychometric_delaychoice_varytime(save_addon, savename_append=save_addon)

    for rule in [DMSGO, DMSNOGO, DMCGO, DMCNOGO]:
        pass
        # psychometric_delaymatching(save_addon, rule)

    save_addon = 'allrule_softplus_300'
    for rule in [GO, INHGO, DELAYGO, REMAP, INHREMAP, DELAYREMAP]:
        psychometric_goantifamily_2D(save_addon, rule)

