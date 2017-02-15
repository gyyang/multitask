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

save = False

def get_dim(h, zero_mean=False):
    # Get effective dimension
    # h : (n_samples, n_features)

    # Abbott, Rajan, Sompolinsky 2011
    # N_eff = (\sum \lambda_i)^2 / \sum (\lambda_i^2)
    # \lambda_i is the i-th eigenvalue

    if zero_mean:
        h = h - h.mean(axis=0) # Do not change it in-place

    _, s, _ = np.linalg.svd(h)
    l = s**2 # get eigenvalues
    N_eff = (np.sum(l)**2) / np.sum(l**2)

    return N_eff

class TaskSetAnalysis(object):
    def __init__(self, save_addon, fast_eval=True):
        ########################## Running the network ################################
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

        # Stimulus-averaged traces
        h_stimavg_byrule  = OrderedDict()
        h_stimavg_byepoch = OrderedDict()
        # Last time points of epochs
        h_lastt_byepoch   = OrderedDict()

        with Run(save_addon, sigma_rec=0, fast_eval=fast_eval) as R:
            config = R.config
            nx, nh, ny = config['shape']

            for rule in rules:
                task = generate_onebatch(rule=rule, config=config, mode='test')
                h = R.f_h(task.x)

                # Average across stimulus conditions
                h_stimavg = h.mean(axis=1)

                dt_new = 50
                every_t = int(dt_new/config['dt'])

                t_start = int(500/config['dt']) # Important: Ignore the initial transition
                # Average across stimulus conditions
                h_stimavg_byrule[rule] = h_stimavg[t_start:, :][::every_t,...]

                for e_name, e_time in task.epochs.iteritems():
                    if 'fix' in e_name:
                        continue

                    # if ('fix' not in e_name) and ('go' not in e_name):
                    # Take epoch
                    h_stimavg_byepoch[(rule, e_name)] = h_stimavg[e_time[0]:e_time[1],:][::every_t,...]
                    # Take last time point from epoch
                    # h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[0]:e_time[1],:,:][-1], axis=1)

                    h_lastt_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1],:,:][-1]

        self.rules = rules
        self.h_stimavg_byrule  = h_stimavg_byrule
        self.h_stimavg_byepoch = h_stimavg_byepoch
        self.h_lastt_byepoch   = h_lastt_byepoch
        self.save_addon = save_addon

    @staticmethod
    def filter(h, rules=None, epochs=None, non_rules=None, non_epochs=None,
               get_lasttimepoint=False, get_timeaverage=False, **kwargs):
        # h should be a dictionary
        # get a new dictionary containing keys from the list of rules and epochs
        # And avoid epochs from non_rules and non_epochs
        # h_new = OrderedDict([(key, val) for key, val in h.iteritems() if key[1] in epochs])

        if get_lasttimepoint:
            print('Analyzing last time points of epochs')
        if get_timeaverage:
            print('Analyzing time-averaged activities of epochs')

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
                elif get_timeaverage:
                    h_new[key] = np.mean(h[key], axis=0, keepdims=True)
                else:
                    h_new[key] = h[key]

        return h_new


    def plot_taskspace(self, rules=None, epochs=None, dim_reduction_type='MDS',
                       plot_text=True, color_by_feature=False, feature=None,
                       figsize=(4,4), markersize=5, plot_label=True,
                       plot_special_point=True, **kwargs):
        # Plot tasks in space

        # Only get last time points for each epoch
        h = self.filter(self.h_stimavg_byepoch, epochs=epochs, rules=rules, **kwargs)

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


        shape_mapping = {'tar1' : 'o',
                         'tar2' : 'o',
                         'delay1' : 'v',
                         'delay2' : 'd',
                         'go1'  : 's',
                         'fix1' : 'p'}

        from performance import rule_color

        fs = 6 # fontsize
        dim0, dim1 = (0, 1) # plot dimensions

        texts = list()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.05, 0.05, 0.75, 0.75])
        for key, val in h_trans.iteritems():
            rule, epoch = key

            if color_by_feature:
                color = 'red' if feature in rule_features[rule] else 'black'
                color = np.array(sns.xkcd_palette([color])[0])
            else:
                # Default coloring by rule_color
                color = np.array(rule_color[rule])


            ax.plot(val[-1,dim0], val[-1,dim1], shape_mapping[epoch],
                    color=color, mec=color*1, mew=1.0, ms=markersize)

            if plot_text:
                texts.append(ax.text(val[-1,dim0], val[-1,dim1], rule_name[rule],
                                     fontsize=6, color=color*0.5))

            if 'fix' not in epoch:
                ax.plot(val[:,dim0], val[:,dim1], color=color, alpha=0.5)

        if plot_label:
            ax.set_xlabel(dim_reduction_type + ' dim. {:d}'.format(dim0+1), fontsize=fs)
            ax.set_ylabel(dim_reduction_type + ' dim. {:d}'.format(dim1+1), fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        # plt.locator_params(nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.1)
        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')

        # Plot special points:
        if rules == [INHGO, INHREMAP, DELAYGO, DELAYREMAP]:
            special_point = -(h[(INHREMAP,'tar1')] - h[(DELAYREMAP,'tar1')]) + h[(INHGO,'tar1')]
        elif rules == [DMCGO, DMCNOGO, DMSGO, DMSNOGO]:
            special_point = -(h[(DMSGO,'tar1')] - h[(DMSNOGO,'tar1')]) + h[(DMCGO,'tar1')]
        elif rules == [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]:
            special_point = np.concatenate(
                ((h[(CHOICEATTEND_MOD1,'tar1')] + h[(CHOICE_INT,'tar1')])/2,
                (h[(CHOICEATTEND_MOD2,'tar1')] + h[(CHOICE_INT,'tar1')])/2), axis=0)
        else:
            plot_special_point = False

        if plot_special_point:
            special_point_trans = model.transform(special_point)
            ax.plot(special_point_trans[:,dim0], special_point_trans[:,dim1], '*',
                    color=sns.xkcd_palette(['black'])[0], markersize=4)

        if color_by_feature:
            ax.set_title('{:s} vs. others'.format(feature_names[feature]), fontsize=fs)

        if plot_text:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

        save_name = 'taskspace'+dim_reduction_type

        if epochs is not None:
            save_name = save_name + ''.join(epochs)

        if color_by_feature:
            save_name = save_name + '_' + feature_names[feature]

        if 'save_append' in kwargs:
            save_name = save_name + kwargs['save_append']

        if save:
            plt.savefig(os.path.join('figure', save_name+'.pdf'), transparent=True)


    def compute_dim(self, **kwargs):
        # compute dimensions of each epoch
        print('Computing dimensions of rule/epochs')
        self.dim_lastt_byepoch = OrderedDict()
        for key, val in self.h_lastt_byepoch.iteritems():
            self.dim_lastt_byepoch[key] = get_dim(val, **kwargs)

    def compute_dim_pair(self, **kwargs):
        # Compute dimension of each pair of epochs, and the dimension ratio

        print('Computing dimensions of pairs of rule/epochs')

        self.dimpair_lastt_byepoch = OrderedDict()
        self.dimpairratio_lastt_byepoch = OrderedDict()

        for key1, val1 in self.h_lastt_byepoch.iteritems():
            for key2, val2 in self.h_lastt_byepoch.iteritems():

                #TODO: TEMP
                val1 = val1 - val1.mean(axis=0)
                val2 = val2 - val2.mean(axis=0)

                h_pair = np.concatenate((val1, val2), axis=0)

                dim_pair = get_dim(h_pair, **kwargs)
                dim1, dim2 = self.dim_lastt_byepoch[key1], self.dim_lastt_byepoch[key2]

                self.dimpair_lastt_byepoch[(key1, key2)] = dim_pair
                self.dimpairratio_lastt_byepoch[(key1, key2)] = dim_pair/(dim1 + dim2)


def plot_taskspaces(save_addon, **kwargs):
    tsa = TaskSetAnalysis(save_addon)
    # tsa.plot_taskspace(epochs=['tar1', 'delay1', 'go1'], **kwargs)

    # tsa.plot_taskspace(epochs=['tar1'], plot_text=True, figsize=(3.5,3.5), **kwargs)

    # for feature in features:
    #     tsa.plot_taskspace(epochs=['tar1'], plot_text=False, color_by_feature=True,
    #                        feature=feature, figsize=(1.0,1.0), markersize=2, plot_label=False, **kwargs)

    rules_list = [[INHGO, INHREMAP, DELAYGO, DELAYREMAP],
                  [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT],
                  [DMCGO, DMCNOGO, DMSGO, DMSNOGO]]
    for i, rules in enumerate(rules_list):
        tsa.plot_taskspace(rules=rules, epochs=['tar1'], plot_text=True,
                           save_append='type'+str(i), figsize=(1.5,1.5),
                           markersize=3, plot_label=False, dim_reduction_type='PCA', **kwargs)

    # epochs = ['tar1', 'delay1', 'go1']

def plot_dim():
    save_addon = 'allrule_weaknoise_400'
    tsa = TaskSetAnalysis(save_addon)
    tsa.compute_dim()
    
    epoch_names = tsa.dim_lastt_byepoch.keys()
    dims = tsa.dim_lastt_byepoch.values()
    
    ind_sort = np.argsort(dims)
    tick_names = [rule_name[epoch_names[i][0]] +' '+ epoch_names[i][1] for i in ind_sort]
    dims = [dims[i] for i in ind_sort]
    
    fig = plt.figure(figsize=(1.5,5))
    ax = fig.add_axes([0.6,0.15,0.35,0.8])
    ax.plot(dims,range(len(dims)), 'o-', color=sns.xkcd_palette(['cerulean'])[0], markersize=3)
    plt.yticks(range(len(dims)),tick_names, rotation=0, ha='right', fontsize=6)
    plt.ylim([-0.5, len(dims)-0.5])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlabel('Task Dim.', fontsize=7, labelpad=1)
    plt.locator_params(axis='x',nbins=3)
    plt.savefig('figure/temp.pdf', transparent=True)
    plt.show()

def plot_dimpair():
    save_addon = 'allrule_weaknoise_400'
    tsa = TaskSetAnalysis(save_addon)
    tsa.compute_dim()
    tsa.compute_dim_pair()


    epoch_names = tsa.h_lastt_byepoch.keys()

    # Arbitrarily define sort order for epochs
    epoch_map = dict(zip(['tar1', 'tar2', 'delay1', 'delay2', 'go1'], range(5)))
    epoch_names_forsort = [(en[0], epoch_map[en[1]]) for en in epoch_names]
    ind_sort = np.lexsort(zip(*epoch_names_forsort)) # sort epoch_names first by epoch then by rule
    epoch_names = [epoch_names[i] for i in ind_sort]


    dimratio_pair_matrix = np.zeros((len(epoch_names), len(epoch_names)))
    for i1, key1 in enumerate(epoch_names):
        for i2, key2 in enumerate(epoch_names):
            dimratio_pair_matrix[i1, i2] = tsa.dimpairratio_lastt_byepoch[(key1, key2)]


    figsize = (5,5)
    rect = [0.2,0.2,0.7,0.7]
    tick_names = [rule_name[en[0]] +' '+ en[1] for en in epoch_names]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    # cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
    im = ax.imshow(2-2*dimratio_pair_matrix, aspect='equal', cmap='hot',
                   vmin=0,vmax=1.0,interpolation='nearest',origin='lower')

    if len(tick_names)<20:
        tick_fontsize = 7
    elif len(tick_names)<30:
        tick_fontsize = 6
    else:
        tick_fontsize = 5

    _ = plt.xticks(range(len(tick_names)), tick_names,
               rotation=90, ha='left', fontsize=tick_fontsize)
    _ = plt.yticks(range(len(tick_names)), tick_names,
               rotation=0, va='center', fontsize=tick_fontsize)

    cax = fig.add_axes([rect[0]+rect[2]+0.05, rect[1], 0.05, rect[3]])
    cb = plt.colorbar(im, cax=cax, ticks=[0,0.5,1])
    cb.set_label('Similarity', fontsize=7, labelpad=3)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig('figure/temp.pdf',transparent=True)

save_addon = 'allrule_weaknoise_400'
plot_taskspaces(save_addon, get_lasttimepoint=True)
# plot_taskspaces(save_addon, get_timeaverage=True)

# plot_dim()
# plot_dimpair()



def temp():
    save_addon = 'allrule_weaknoise_400'
    tsa = TaskSetAnalysis(save_addon)

    epochs = ['tar1']
    rules = None

    h = tsa.filter(tsa.h_stimavg_byepoch, epochs=epochs, rules=rules, get_lasttimepoint=True)

    h_keys = h.keys()
    n_epochs = len(h_keys)

    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::"""
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    shuffle_angle = True
    h2 = OrderedDict()
    for key in h_keys:
        h2[key] = h[key][-1]
        if shuffle_angle:
            h2[key] = np.random.permutation(h2[key])

    vec_diffs = list()
    for i in range(n_epochs-1):
        for j in range(i+1, n_epochs):
            hi = h2[h_keys[i]]
            hj = h2[h_keys[j]]

            vec_diffs.append(hi-hj)

    n_diff = len(vec_diffs)
    tmps = list()

    from scipy.stats import linregress

    for i in range(n_diff-1):
        for j in range(i+1, n_diff):
            # tmps.append(angle_between(vec_diffs[i], vec_diffs[j]))
            slope, intercept, r_value, p_value, std_err = linregress(vec_diffs[i],vec_diffs[j])
            tmps.append(r_value)

    _ = plt.hist(tmps)

