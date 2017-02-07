"""
Analysis of the OIC and DMC tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
from task import *
from run import Run

save = True # TEMP

def gen_taskparams():
    a = 12
    n_tar_loc = 2 * a
    batch_size = n_tar_loc

    tar1_locs = np.concatenate(((0.1+0.8*np.arange(a)/a),(1.1+0.8*np.arange(a)/a)))*np.pi
    tar1_cats = (tar1_locs<np.pi).astype(int) # Category of target 1

    params = dict()
    params[OIC] = {
        'tar1_locs' : tar1_locs,
        'tar1_cats' : tar1_cats, # This is actually not used by task
        'tar2_locs' : [0]     * batch_size,
        'tar3_locs' : [np.pi] * batch_size
        }

    params[DMC] = {
        'tar1_locs' : tar1_locs,
        'tar1_cats' : tar1_cats, # This is actually not used by task
        'tar2_locs' : tar1_locs, # Doesn't matter for current analysis
        }

    return params

save_addon = 'oicdmconly_strongnoise_30'
rules = [OIC, DMC]

# Analyzing the sample period, so not iterating over second stimulus or target locations

params = gen_taskparams()
h_samples = dict()

with Run(save_addon, fast_eval=True) as R:
    config = R.config
    for rule in rules:
        task = generate_onebatch(rule, config, 'psychometric', params=params[rule])
        h_samples[rule] = R.f_h(task.x)


####################### Plotting neuronal activities ##########################
if True:
    tar1_cats = params[OIC]['tar1_cats'] # Should be the same for both tasks

    fs = 7

    # time_ind = np.arange(int(1500/config['dt']))
    # Show same time range
    # t_plot = time_ind*config['dt']/1000.

    # for plot_ind in range(0, 20):
    for plot_ind in range(config['HDIM']):

        # ymax = np.max([np.max(h_samples[rule][time_ind, :, plot_ind]) for rule in rules])
        ymax = np.max([np.max(h_samples[rule][:, :, plot_ind]) for rule in rules])

        fig = plt.figure(figsize=(3,1.5))
        for i, rule in enumerate(rules):
            # Show full range
            t_plot = np.arange(h_samples[rule].shape[0])*config['dt']/1000.
            ax = fig.add_axes([.2+i*0.4, .2, .3, .7])

            _ = ax.plot(t_plot, h_samples[rule][:, tar1_cats==1, plot_ind], color='blue')
            _ = ax.plot(t_plot, h_samples[rule][:, tar1_cats==0, plot_ind], color='red')
            # _ = ax.plot(t_plot, h_samples[rule][:, tar1_cats==1, plot_ind][time_ind], color='blue')
            # _ = ax.plot(t_plot, h_samples[rule][:, tar1_cats==0, plot_ind][time_ind], color='red')
            ax.set_title(rule_name[rule] + ' Unit {:d}'.format(plot_ind), fontsize=fs)
            ax.set_xlabel('Time (s)', fontsize=fs)
            if i == 0:
                ax.set_ylabel('Activity (a.u.)', fontsize=fs)
            plt.locator_params(nbins=3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
            ax.set_ylim([0, ymax*1.2])


############################ Classification ###################################
if False:
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    # clf = AdaBoostClassifier()
    # clf = GaussianNB()
    clf = LinearDiscriminantAnalysis()
    # clf = QuadraticDiscriminantAnalysis()
    # clf = DecisionTreeClassifier(max_depth=5)
    # clf = KNeighborsClassifier(n_neighbors=7)
    # clf = SVC(kernel='rbf', probability=True)
    # clf = SVC(kernel="linear", C=0.025)
    # clf = SVC(kernel="linear", C=1.0)
    # clf = SVC(C=20.)
    # clf = LinearSVC()
    # clf = LinearSVC(multi_class='crammer_singer')

    for rule_train in rules: # Rule used to train classifier
    # rule_train = DMC # Rule used to train classifier
    
        prop_train = 0.5 # Proportion of data used for training classifiers
        batch_size = h_samples[OIC].shape[1]
        n_train = int(batch_size*prop_train)
        n_rep = 30 # Number of repetition for cross-validation
    
        prop_subsample = 0.05
        n_subsample = int(config['HDIM'] * prop_subsample)
    
        # Store mean classification performance in time
        preds_intime = {rule : [] for rule in rules}
        time_inds = np.arange(150)
        for time_ind in time_inds:
            # Get mean prediction on hold-one-out data
            X_oic = h_samples[OIC][time_ind] # (n_samples, n_features) = (n_trials, n_neurons)
            X_dmc = h_samples[DMC][time_ind] # (n_samples, n_features) = (n_trials, n_neurons)
            y = params[rule_train]['tar1_cats'] # categories, should be the same for both rules
    
            preds = {rule : [] for rule in rules} # Store hold-one-out results
    
            # Looping over hold-out data points
            for i in range(n_rep):
                # Get random trials
                ind_shuffle = np.arange(batch_size)
                np.random.shuffle(ind_shuffle)
                ind_train = ind_shuffle[:n_train]
                ind_test  = ind_shuffle[n_train:]
    
                # Get random units
                ind_shuffle = np.arange(config['HDIM'])
                np.random.shuffle(ind_shuffle)
                ind_subsample = ind_shuffle[:n_subsample]
    
                X_train = h_samples[rule_train][time_ind, ind_train, :][:, ind_subsample]
                clf.fit(X_train, y[ind_train])
    
                for rule_test in rules:
                    X_test = h_samples[rule_test][time_ind, ind_test, :][:, ind_subsample]
                    y_test = params[rule_test]['tar1_cats'][ind_test]
    
                    preds[rule_test].append(np.mean(y_test==clf.predict(X_test)))
    
    
            for rule in rules:
                preds_intime[rule].append(np.mean(preds[rule]))
    
    
        colors = ['black', 'red']
        fs = 7
    
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_axes([.2, .2, .7, .7])
        for i, rule in enumerate(rules):
            _ = ax.plot(time_inds*config['dt']/1000., preds_intime[rule],
                        color=colors[i], label=rule_name[rule])
    
        lg = ax.legend(title='test rule',
                       fontsize=fs, ncol=1, bbox_to_anchor=(1.0,0.5),
                       loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        ax.set_title('Trained on ' + rule_name[rule_train], fontsize=fs)
        ax.set_xlabel('Time (s)', fontsize=fs)
        ax.set_ylabel('Classification performance', fontsize=fs)
        plt.locator_params(nbins=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='major', labelsize=fs, length=2)