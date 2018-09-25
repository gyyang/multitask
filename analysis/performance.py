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
from collections import OrderedDict
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

import tensorflow as tf

from task import generate_trials, rule_name, get_dist
from network import Model
import tools


_rule_color = {
    'reactgo': 'green',
            'delaygo': 'olive',
            'fdgo': 'forest green',
            'reactanti': 'mustard',
            'delayanti': 'tan',
            'fdanti': 'brown',
            'dm1': 'lavender',
            'dm2': 'aqua',
            'contextdm1': 'bright purple',
            'contextdm2': 'green blue',
            'multidm': 'blue',
            'delaydm1': 'indigo',
            'delaydm2': 'grey blue',
            'contextdelaydm1': 'royal purple',
            'contextdelaydm2': 'dark cyan',
            'multidelaydm': 'royal blue',
            'dmsgo': 'red',
            'dmsnogo': 'rose',
            'dmcgo': 'orange',
            'dmcnogo': 'peach'
            }

rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}

save = True
THETA = 0.3 * np.pi

# From sns.dark_palette("light blue", 3, input="xkcd")
BLUES = [np.array([0.13333333, 0.13333333, 0.13333333, 1.        ]),
         np.array([0.3597078 , 0.47584775, 0.56246059, 1.        ]),
         np.array([0.58431373, 0.81568627, 0.98823529, 1.        ])]


def plot_performanceprogress(model_dir, rule_plot=None):
    # Plot Training Progress
    log = tools.load_log(model_dir)
    hp = tools.load_hp(model_dir)

    trials = log['trials']

    fs = 6 # fontsize
    fig = plt.figure(figsize=(3.5,1.2))
    ax = fig.add_axes([0.1,0.25,0.35,0.6])
    lines = list()
    labels = list()

    x_plot = np.array(trials)/1000.
    if rule_plot == None:
        rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
        # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
        line = ax.plot(x_plot, np.log10(log['cost_'+rule]),
                       color=rule_color[rule])
        ax.plot(x_plot, log['perf_'+rule], color=rule_color[rule])
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
    plt.savefig('figure/Performance_Progresss.pdf', transparent=True)
    plt.show()


def plot_performanceprogress_cont(model_dirs, save=True):
    model_dir1, model_dir2 = model_dirs
    _plot_performanceprogress_cont(model_dir1, save=save)
    _plot_performanceprogress_cont(model_dir1, model_dir2, save=save)


def _plot_performanceprogress_cont(model_dir, model_dir2=None, save=True):
    # Plot Training Progress
    log = tools.load_log(model_dir)
    hp = tools.load_hp(model_dir)

    trials = np.array(log['trials'])/1000.
    times = log['times']
    rule_now = log['rule_now']

    if model_dir2 is not None:
        log2 = tools.load_log(model_dir2)
        trials2 = np.array(log2['trials'])/1000.

    fs = 7  # fontsize
    lines = list()
    labels = list()

    rule_train_plot = hp['rule_trains']
    rule_test_plot = hp['rules']

    nx, ny = 4, 2
    fig, axarr = plt.subplots(nx, ny, figsize=(3, 3), sharex=True)
    for i in range(int(nx*ny)):
        ix, iy = i % nx, int(i / nx)
        ax = axarr[ix, iy]

        if i >= len(rule_test_plot):
            ax.axis('off')
            continue

        rule = rule_test_plot[i]

        # Plot fills
        trials_rule_prev_end = 0  # end of previous rule training time
        for rule_ in rule_train_plot:
            if rule == rule_:
                ec = 'black'
            else:
                ec = (0, 0, 0, 0.1)
            trials_rule_now = [trials_rule_prev_end] + [
                    trials[ii] for ii in range(len(rule_now))
                    if rule_now[ii] == rule_]
            trials_rule_prev_end = trials_rule_now[-1]
            ax.fill_between(trials_rule_now, 0, 1, facecolor='none',
                            edgecolor=ec, linewidth=0.5)

        # Plot lines
        line = ax.plot(trials, log['perf_'+rule], lw=1, color='gray')
        if model_dir2 is not None:
            ax.plot(trials2, log2['perf_'+rule], lw=1, color='red')
        lines.append(line[0])
        if isinstance(rule, str):
            rule_name_print = rule_name[rule]
        else:
            rule_name_print = ' & '.join([rule_name[r] for r in rule])
        labels.append(rule_name_print)

        ax.tick_params(axis='both', which='major', labelsize=fs)

        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, trials_rule_prev_end])
        ax.set_yticks([0, 1])
        ax.set_xticks([0, np.floor(trials_rule_prev_end / 100.) * 100])
        if (ix == nx-1) and (iy == 0):
            ax.set_xlabel('Total trials (1,000)', fontsize=fs, labelpad=1)
        if i == 0:
            ax.set_ylabel('Performance', fontsize=fs, labelpad=1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(rule_name[rule], fontsize=fs, y=0.87, color='black')

    print('Training time {:0.1f} hours'.format(times[-1]/3600.))
    if save:
        name = 'TrainingCont_Progress'
        if model_dir2 is not None:
            name = name + '2'
        plt.savefig('figure/'+name+'.pdf', transparent=True)
    plt.show()


def get_finalperformance(model_dirs):
    """Get lists of final performance."""
    hp = tools.load_hp(model_dirs[0])

    rule_plot = hp['rules']

    final_cost, final_perf = OrderedDict(), OrderedDict()
    for rule in rule_plot:
        final_cost[rule] = list()
        final_perf[rule] = list()
    training_time_plot = list()

    # Recording performance and cost for networks
    for model_dir in model_dirs:
        log = tools.load_log(model_dir)
        if log is None:
            continue

        for rule in rule_plot:
            final_perf[rule] += [float(log['perf_'+rule][-1])]
            final_cost[rule] += [float(log['cost_'+rule][-1])]
        training_time_plot.append(log['times'][-1])

    return final_cost, final_perf, rule_plot, training_time_plot


def plot_finalperformance_cont(model_dirs1, model_dirs2):
    final_cost, final_perf1, rule_plot, training_time_plot = \
        get_finalperformance(model_dirs1)

    final_cost, final_perf2, rule_plot, training_time_plot = \
        get_finalperformance(model_dirs2)

    final_perf_plot1 = np.array(list(final_perf1.values()))
    final_perf_plot2 = np.array(list(final_perf2.values()))

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


def get_allperformance(model_dirs, param_list=None):
    # Get all model names that match patterns (strip off .ckpt.meta at the end)
    model_dirs = tools.valid_model_dirs(model_dirs)

    final_perfs = dict()
    filenames = dict()

    if param_list is None:
        param_list = ['param_intsyn', 'easy_task', 'activation']

    for model_dir in model_dirs:
        log = tools.load_log(model_dir)
        hp = tools.load_hp(model_dir)

        perf_tests = log['perf_tests']

        final_perf = np.mean([float(val[-1]) for val in perf_tests.values()])

        key = tuple([hp[p] for p in param_list])
        if key in final_perfs.keys():
            final_perfs[key].append(final_perf)
        else:
            final_perfs[key] = [final_perf]
            filenames[key] = model_dir

    for key, val in final_perfs.items():
        final_perfs[key] = np.mean(val)
        print(key),
        print('{:0.3f}'.format(final_perfs[key])),
        print(filenames[key])


################ Psychometric - Varying Coherence #############################
def _psychometric_dm(model_dir, rule, params_list, batch_shape):
    """Base function for computing psychometric performance in 2AFC tasks

    Args:
        model_dir : model name
        rule : task to analyze
        params_list : a list of parameter dictionaries used for the psychometric mode
        batch_shape : shape of each batch. Each batch should have shape (n_rep, ...)
        n_rep is the number of repetitions that will be averaged over

    Return:
        ydatas: list of performances
    """
    print('Starting psychometric analysis of the {:s} task...'.format(rule_name[rule]))

    model = Model(model_dir)
    hp = model.hp
    with tf.Session() as sess:
        model.restore()

        ydatas = list()
        for params in params_list:

            trial  = generate_trials(rule, hp, 'psychometric', params=params)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            y_loc_sample = sess.run(model.y_hat_loc, feed_dict=feed_dict)
            y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)

            stim1_locs_ = np.reshape(params['stim1_locs'], batch_shape)
            stim2_locs_ = np.reshape(params['stim2_locs'], batch_shape)

            # Average over the first dimension of each batch
            choose1 = (get_dist(y_loc_sample - stim1_locs_) < THETA).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - stim2_locs_) < THETA).sum(axis=0)
            ydatas.append(choose1/(choose1 + choose2))

    return ydatas


def psychometric_choice(model_dir, **kwargs):
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
        params = {'stim1_locs': stim1_locs,
                  'stim2_locs': stim2_locs,
                  'stim1_strengths': stim1_strengths,
                  'stim2_strengths': stim2_strengths,
                  'stim_time': stim_time}

        params_list.append(params)

    xdatas = [stim_cohs*2] * len(stim_times)
    ydatas = _psychometric_dm(model_dir, rule, params_list, batch_shape)

    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in stim_times],
                              colors=BLUES,
                              legtitle='Stim. time (ms)', rule=rule, **kwargs)


def psychometric_delaychoice(model_dir, **kwargs):
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
        params = {'stim1_locs' : stim1_locs,
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
    ydatas = _psychometric_dm(model_dir, rule, params_list, batch_shape)

    plot_psychometric_choice(xdatas, ydatas,
                              labels=[str(t) for t in dtars],
                              colors=BLUES,
                              legtitle='Delay (ms)',rule=rule, **kwargs)


def psychometric_choiceattend(model_dir, **kwargs):
    psychometric_choiceattend_(model_dir, 'contextdm1', **kwargs)
    psychometric_choiceattend_(model_dir, 'contextdm2', **kwargs)


def psychometric_choiceattend_(model_dir, rule, **kwargs):
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


    prop1s = _psychometric_dm(model_dir, rule, [params], batch_shape)[0]

    xdatas = [stim_cohs*2]*2
    ydatas = [prop1s.mean(axis=k) for k in [1,0]]

    labels = ['Attend', 'Ignore'] if rule=='contextdm1' else ['Ignore', 'Attend']

    # from sns.color_palette("Set2",2)
    colors = [(0.4, 0.7607843137254902, 0.6470588235294118),
              (0.9882352941176471, 0.5529411764705883, 0.3843137254901961)]

    plot_psychometric_choice(xdatas, ydatas,
                              labels=labels,
                              colors=colors,
                              legtitle='Modality',rule=rule, **kwargs)


def psychometric_choiceint(model_dir, **kwargs):
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
    ydatas = _psychometric_dm(model_dir, rule, params_list, batch_shape)

    # sns.color_palette("Set2",3)
    colors = [(0.4, 0.7607843137254902, 0.6470588235294118),
              (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
              (0.5529411764705883, 0.6274509803921569, 0.796078431372549)]

    fits = plot_psychometric_choice(xdatas, ydatas,
                                    labels=['1 only', '2 only', 'both'],
                                    colors=colors,
                                    legtitle='Modality', rule=rule, **kwargs)
    sigmas = [fit[1] for fit in fits]
    print('Fit sigmas:')
    print(sigmas)


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
    plt.xticks([xdata[0], 0, xdata[-1]])
    if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
        plt.yticks([0,0.5,1],['','',''])
    else:
        plt.ylabel('P(choice 1)',fontsize=fs)
    plt.title(rule_name[kwargs['rule']], fontsize=fs, y=0.95)
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


def psychometric_choicefamily_2D(model_dir, rule, lesion_units=None,
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
    params_dict['dm1'] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_strengths' : 1 + stim_mod1_cohs, # Just use mod 1 value
          'stim2_strengths' : 1 - stim_mod1_cohs,
          'stim_time'    : 800
          }
    params_dict['dm2'] = params_dict['dm1']

    params_dict['contextdm1'] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
          'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
          'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
          'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
          'stim_time'    : 800
          }

    params_dict['contextdm2'] = params_dict['contextdm1']

    params_dict['contextdelaydm1'] = params_dict['contextdm1']
    params_dict['contextdelaydm1']['stim_time'] = 800
    params_dict['contextdelaydm2'] = params_dict['contextdelaydm1']

    params_dict['multidm'] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
          'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
          'stim1_mod2_strengths' : 1 + stim_mod1_cohs, # Same as Mod 1
          'stim2_mod2_strengths' : 1 - stim_mod1_cohs,
          'stim_time'    : 800
          }

    params_dict['contextdelaydm1'] = \
         {'stim1_locs' : stim1_locs,
          'stim2_locs' : stim2_locs,
          'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
          'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
          'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
          'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
          'stim_time'    : 800
          }

    model = Model(model_dir)
    hp = model.hp
    with tf.Session() as sess:
        model.restore()
        model.lesion_units(sess, lesion_units)

        params = params_dict[rule]
        trial = generate_trials(rule, hp, 'psychometric',
                                params=params)
        feed_dict = tools.gen_feed_dict(model, trial, hp)
        y_sample, y_loc_sample = sess.run([model.y_hat, model.y_hat_loc],
                                          feed_dict=feed_dict)

    # Compute the overall performance.
    # Importantly, discard trials where no decision was made
    loc_cor = trial.y_loc[-1]  # last time point, correct locations
    loc_err = (loc_cor+np.pi)%(2*np.pi)
    choose_cor = (get_dist(y_loc_sample[-1] - loc_cor) < THETA).sum()
    choose_err = (get_dist(y_loc_sample[-1] - loc_err) < THETA).sum()
    perf = choose_cor/(choose_cor+choose_err)

    # Compute the proportion of choosing choice 1 and maintain the batch_shape
    stim1_locs_ = np.reshape(stim1_locs, batch_shape)
    stim2_locs_ = np.reshape(stim2_locs, batch_shape)

    y_loc_sample = np.reshape(y_loc_sample[-1], batch_shape)
    choose1 = (get_dist(y_loc_sample - stim1_locs_) < THETA).sum(axis=0)
    choose2 = (get_dist(y_loc_sample - stim2_locs_) < THETA).sum(axis=0)
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


def plot_psychometric_choicefamily_2D(model_dir, rule, **kwargs):
    perf, prop1s, cohs = psychometric_choicefamily_2D(model_dir, rule, **kwargs)
    _plot_psychometric_choicefamily_2D(prop1s, cohs, rule)


################ Psychometric - Varying Stim Time #############################
def compute_choicefamily_varytime(model_dir, rule):
    assert rule in ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm']
    print('Starting vary time analysis of the {:s} task...'.format(rule_name[rule]))
    n_stim_loc = 3000
    # n_stim_loc = 100  # for quick debugging

    n_coh = 4
    batch_size = n_stim_loc * n_coh
    batch_shape = (n_stim_loc,n_coh)
    ind_stim_loc, ind_stim = np.unravel_index(range(batch_size),batch_shape)

    stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
    stim2_locs = (stim1_locs+np.pi)%(2*np.pi)

    # if rule == 'multidm':
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

    ydatas = _psychometric_dm(model_dir, rule, params_list, batch_shape)

    xdatas = [stim_times] * n_coh
    ydatas = np.array(ydatas).T

    result = {'xdatas' : xdatas, 'ydatas' : ydatas, 'cohs' : cohs}

    savename = os.path.join(model_dir, 'varytime_'+rule)
    with open(savename+'.pkl','wb') as f:
        pickle.dump(result, f)


def plot_choicefamily_varytime(model_dir, rule):
    import seaborn as sns
    assert rule in ['dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm']
    savename = os.path.join(model_dir, 'varytime_' + rule + '.pkl')

    try:
        result = tools.load_pickle(savename)
    except FileNotFoundError:
        raise FileNotFoundError('Run performance.compute_choicefamily_varytime first.')

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
    # figname = figname + model_dir
    if save:
        plt.savefig('figure/'+figname+'.pdf', transparent=True)


    # Chronometric curve
    figname = 'varytime_'+rule_name[rule].replace(' ','')
    # figname = figname + model_dir
    plot_psychometric_varytime(xdatas, ydatas, figname,
                              labels=['{:0.3f}'.format(t) for t in 2*cohs],
                              colors=sns.dark_palette("light blue", n_coh, input="xkcd"),
                              legtitle='Stim. 1 - Stim. 2',rule=rule)


def psychometric_delaychoice_varytime(model_dir, rule):
    import seaborn as sns
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
    ydatas = _psychometric_dm(model_dir, rule, params_list, batch_shape)
    xdatas, ydatas = np.array(xdatas), np.array(ydatas).T

    figname = 'varytime_'+rule
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


if __name__ == '__main__':
    pass
