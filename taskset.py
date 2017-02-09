"""
Task set analysis (temporary name)
Analyze how state-space of stimulus-averaged activity
"""

from __future__ import division

import os
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from task import *
from run import Run, plot_singleneuron_intime


class TaskSetAnalysis(object):
    def __init__(self, save_addon, fast_eval=True):
        ########################## Running the network ################################
        data_type = 'rule'
        rules = [GO, INHGO, DELAYGO,\
                CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
                CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
                REMAP, INHREMAP, DELAYREMAP,\
                DMSGO, DMSNOGO, DMCGO, DMCNOGO]

        # rules = [CHOICE_MOD1, CHOICE_MOD2]
        # rules = [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
        # rules = [GO, INHGO, DELAYGO, REMAP, INHREMAP, DELAYREMAP]
        # rules = [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2]
        # rules = [DMSGO, DMSNOGO]
        # rules = [DMCGO, DMCNOGO]
        ########################## Running the network ################################
        n_rules = len(rules)

        h_all_byrule  = OrderedDict()
        h_all_byepoch = OrderedDict()
        with Run(save_addon, sigma_rec=0, fast_eval=fast_eval) as R:
            config = R.config
            nx, nh, ny = config['shape']

            for rule in rules:
                task = generate_onebatch(rule=rule, config=config, mode='test')
                h = R.f_h(task.x)

                t_start = int(500/config['dt']) # Important: Ignore the initial transition

                # Average across stimulus conditions
                h_all_byrule[rule] = np.mean(h[t_start:, :, :], axis=1)

                for e_name, e_time in task.epochs.iteritems():
                    if 'fix' not in e_name:
                    # if ('fix' not in e_name) and ('go' not in e_name):
                        # Take epoch
                        # h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[0]:e_time[1],:,:], axis=1)
                        # Take last time point from epoch
                        h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[1]-1,:,:], axis=0)

        self.rules = rules
        self.h_all_byrule  = h_all_byrule
        self.h_all_byepoch = h_all_byepoch

    def plot_tmp(self):

        ########################## Dimensionality reduction ###########################

        # Concatenate across rules to create dataset
        data = np.concatenate(self.h_all_byepoch.values(), axis=0)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=5, whiten=False)
        pca.fit(data)


        from performance import color_rules
        fs = 7
        pcs = (0, 1)

        lines = list()
        labels = list()
        fig = plt.figure(figsize=(3.5,2.5))
        ax = fig.add_axes([0.1, 0.2, 0.5, 0.7])
        for rule in self.rules:
            data1 = pca.transform(self.h_all_byrule[rule])
            line = ax.plot(data1[:,pcs[0]], data1[:,pcs[1]], '-', color=color_rules[rule], linewidth=1)
            ax.plot(data1[0,pcs[0]], data1[0,pcs[1]], 's', color=color_rules[rule], markersize=3)
            lines.append(line[0])
            labels.append(rule_name[rule])

            # data1 = pca.transform(h_all_byepoch[(rule, 'go1')])
            # ax.plot(data1[:,pcs[0]], data1[:,pcs[1]], '-', color=color_rules[rule], linewidth=2)

        lg = ax.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(1.0,0.5),
                            fontsize=fs,labelspacing=0.3,loc=6)
        plt.setp(lg.get_title(),fontsize=fs)
        # plt.savefig('figure/temp_taskset{:d}{:d}.pdf'.format(pcs[0], pcs[1]), transparent=True)