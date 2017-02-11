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

save = True

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

                # Average across stimulus conditions
                h = h.mean(axis=1)

                dt_new = 50
                every_t = int(dt_new/config['dt'])

                t_start = int(500/config['dt']) # Important: Ignore the initial transition
                # Average across stimulus conditions
                h_all_byrule[rule] = h[t_start:, :][::every_t,...]

                for e_name, e_time in task.epochs.iteritems():
                    if 'fix' in e_name:
                        continue

                    # if ('fix' not in e_name) and ('go' not in e_name):
                    # Take epoch
                    h_all_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1],:][::every_t,...]
                    # Take last time point from epoch
                    # h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[0]:e_time[1],:,:][-1], axis=1)

        self.rules = rules
        self.h_all_byrule  = h_all_byrule
        self.h_all_byepoch = h_all_byepoch
        self.save_addon = save_addon

    @staticmethod
    def filter(h, rules=None, epochs=None, non_rules=None, non_epochs=None, get_lasttimepoint=False):
        # h should be a dictionary
        # get a new dictionary containing keys from the list of rules and epochs
        # And avoid epochs from non_rules and non_epochs
        # h_new = OrderedDict([(key, val) for key, val in h.iteritems() if key[1] in epochs])
        h_new = OrderedDict()
        for key in h:
            rule, epoch = key

            include_key = True
            if rules is not None:
                include_key = include_key and (rule in rules)

            if epochs is not None:
                include_key = include_key and (epoch in epochs)

            if non_rules is not None:
                include_key = include_key and (rule not in non_rules)

            if non_epochs is not None:
                include_key = include_key and (epoch not in non_epochs)

            if include_key:
                if get_lasttimepoint:
                    h_new[key] = h[key][np.newaxis, -1, :]
                else:
                    h_new[key] = h[key]

        return h_new


    def plot_taskspace(self, rules=None, epochs=None, dim_reduction_type='MDS',
                       plot_text=True):
        # Plot tasks in space

        # Only get last time points for each epoch
        h = self.filter(self.h_all_byepoch, epochs=epochs, rules=rules, get_lasttimepoint=True)

        # Concatenate across rules to create dataset
        data = np.concatenate(h.values(), axis=0)
        data = data.astype(dtype='float64')

        if dim_reduction_type == 'PCA':
            from sklearn.decomposition import PCA
            model = PCA(n_components=5)

        elif dim_reduction_type == 'MDS':
            from sklearn.manifold import MDS
            model = MDS(n_components=2, metric=True, random_state=0)

        elif dim_reduction_type == 'TSNE':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=2)

        elif dim_reduction_type == 'IsoMap':
            from sklearn.manifold import Isomap
            model = Isomap(n_components=2)

        # Transform data
        data_trans = model.fit_transform(data)

        # Package back to dictionary
        h_trans = OrderedDict()
        i_start = 0
        for key, val in h.iteritems():
            i_end = i_start + val.shape[0]
            h_trans[key] = data_trans[i_start:i_end, :]
            i_start = i_end


        shape_mapping = {'tar1' : 's',
                         'tar2' : 's',
                         'delay1' : 'v',
                         'delay2' : 'd',
                         'go1'  : 'o',
                         'fix1' : 'p'}

        from performance import rule_color

        fs = 7 # fontsize
        dim0, dim1 = (0, 1) # plot dimensions

        texts = list()

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for key, val in h_trans.iteritems():
            rule, epoch = key
            ax.plot(val[-1,dim0], val[-1,dim1], shape_mapping[epoch],
                    color=rule_color[rule], mec=np.array(rule_color[rule])*0.5, mew=1.0)

            if plot_text:
                texts.append(ax.text(val[-1,dim0], val[-1,dim1], rule_name[rule],
                                     fontsize=6, color=np.array(rule_color[rule])*0.5))

            if 'fix' not in epoch:
                ax.plot(val[:,dim0], val[:,dim1], color=rule_color[rule], alpha=0.5)

        ax.set_xlabel(dim_reduction_type + ' dimension {:d}'.format(dim0+1), fontsize=fs)
        ax.set_ylabel(dim_reduction_type + ' dimension {:d}'.format(dim1+1), fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        # plt.locator_params(nbins=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if plot_text:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

        save_name = 'taskspace'+dim_reduction_type

        if epochs is not None:
            save_name = save_name + ''.join(epochs)

        if save:
            plt.savefig(os.path.join('figure', save_name+'.pdf'), transparent=True)


def plot_taskspaces():
    save_addon = 'allrule_weaknoise_400'
    tsa = TaskSetAnalysis(save_addon)
    tsa.plot_taskspace(epochs=['tar1', 'delay1', 'go1'])
    tsa.plot_taskspace(epochs=['tar1'])

    # rules = [INHGO, INHREMAP, DELAYGO, DELAYREMAP]
    # rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]
    # rules = [DMCGO, DMCNOGO, DMSGO, DMSNOGO]
    # rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT, CHOICEDELAY_MOD1, CHOICEDELAY_MOD2]
    # rules = None

    # epochs = ['tar1']
    # epochs = ['tar1', 'delay1', 'go1']
    # epochs = None

# plot_taskspaces()