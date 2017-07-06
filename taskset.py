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
import time
import seaborn.apionly as sns

from task import *
from run import Run, plot_singleneuron_intime

save = True

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
    def __init__(self, save_addon, rules=None, fast_eval=True):
        ########################## Running the network ################################
        if rules is None:
            # Default value
            rules = range(N_RULE)

        # rules = [CHOICE_MOD1, CHOICE_MOD2]
        # rules = [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
        # rules = [REACTGO, FDGO, DELAYGO, ANTI, FDANTI, DELAYANTI]
        # rules = [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2]
        # rules = [DMSGO, DMSNOGO]
        # rules = [DMCGO, DMCNOGO]
        # rules = [FDGO, FDANTI, DELAYGO, DELAYANTI]
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

                # dt_new = 50
                # every_t = int(dt_new/config['dt'])

                t_start = int(500/config['dt']) # Important: Ignore the initial transition
                # Average across stimulus conditions
                h_stimavg_byrule[rule] = h_stimavg[t_start:, :]

                for e_name, e_time in task.epochs.iteritems():
                    if 'fix' in e_name:
                        continue

                    # if ('fix' not in e_name) and ('go' not in e_name):
                    # Take epoch
                    e_time_start = e_time[0]-1 if e_time[0]>0 else 0
                    h_stimavg_byepoch[(rule, e_name)] = h_stimavg[e_time_start:e_time[1],:]
                    # Take last time point from epoch
                    # h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[0]:e_time[1],:,:][-1], axis=1)
                    h_lastt_byepoch[(rule, e_name)] = h[e_time[1],:,:]

        self.rules = rules
        self.h_stimavg_byrule  = h_stimavg_byrule
        self.h_stimavg_byepoch = h_stimavg_byepoch
        self.h_lastt_byepoch   = h_lastt_byepoch
        self.save_addon = save_addon

    @staticmethod
    def filter(h, rules=None, epochs=None, non_rules=None, non_epochs=None,
               get_lasttimepoint=True, get_timeaverage=False, **kwargs):
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


    def compute_taskspace(self, rules=None, epochs=None, dim_reduction_type='MDS', **kwargs):
        # Only get last time points for each epoch
        h = self.filter(self.h_stimavg_byepoch, epochs=epochs, rules=rules, **kwargs)

        # Concatenate across rules to create dataset
        data = np.concatenate(h.values(), axis=0)
        data = data.astype(dtype='float64')

        # First reduce dimension to dimension of data points
        from sklearn.decomposition import PCA
        n_comp = int(np.min([data.shape[0], data.shape[1]])-1)
        model = PCA(n_components=n_comp)
        data = model.fit_transform(data)

        if dim_reduction_type == 'PCA':
            model = PCA(n_components=2)

        elif dim_reduction_type == 'MDS':
            from sklearn.manifold import MDS
            model = MDS(n_components=2, metric=True, random_state=0)

        elif dim_reduction_type == 'TSNE':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=2)

        elif dim_reduction_type == 'IsoMap':
            from sklearn.manifold import Isomap
            model = Isomap(n_components=2)

        else:
            raise ValueError('Unknown dim_reduction_type')

        # Transform data
        data_trans = model.fit_transform(data)

        # Package back to dictionary
        h_trans = OrderedDict()
        i_start = 0
        for key, val in h.iteritems():
            i_end = i_start + val.shape[0]
            h_trans[key] = data_trans[i_start:i_end, :]
            i_start = i_end

        return h_trans

    def compute_and_plot_taskspace(self,
               rules=None, epochs=None, **kwargs):

        h_trans = self.compute_taskspace(rules=rules, epochs=epochs, **kwargs)
        self.plot_taskspace(h_trans, **kwargs)

    def plot_taskspace(self, h_trans, epochs=None, dim_reduction_type='MDS',
                       plot_text=True, color_by_feature=False, feature=None,
                       figsize=(4,4), markersize=5, plot_label=True,
                       plot_special_point=False, plot_arrow=False, **kwargs):
        # Plot tasks in space

        shape_mapping = {'stim1' : 'o',
                         'stim2' : 'o',
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

        # if plot_arrow:
        #     if rules == [FDGO, FDANTI, DELAYGO, DELAYANTI]:
        #         arrow_starts = [h[(FDANTI,'stim1')], h[(FDGO,'stim1')]]
        #         arrow_ends   = [h[(DELAYANTI,'stim1')],
        #                         -(h[(FDANTI,'stim1')] - h[(DELAYANTI,'stim1')]) + h[(FDGO,'stim1')]]
        #     elif rules == [DMCGO, DMCNOGO, DMSGO, DMSNOGO]:
        #         arrow_starts = [h[(DMSGO,'stim1')], h[(DMCGO,'stim1')]]
        #         arrow_ends   = [h[(DMSNOGO,'stim1')],
        #                         -(h[(DMSGO,'stim1')] - h[(DMSNOGO,'stim1')]) + h[(DMCGO,'stim1')]]
        #
        #     elif rules == [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]:
        #         arrow_starts = [h[(CHOICE_INT,'stim1')], h[(CHOICE_INT,'stim1')]]
        #         arrow_ends   = [h[(CHOICEATTEND_MOD1,'stim1')], h[(CHOICEATTEND_MOD2,'stim1')]]
        #     else:
        #         ValueError('Arrows not provided')
        #
        #     for arrow_start, arrow_end in zip(arrow_starts, arrow_ends):
        #         arrow_start = model.transform(arrow_start)
        #         arrow_end   = model.transform(arrow_end)
        #
        #         ax.annotate("", xy=arrow_start[-1,:2], xytext=arrow_end[-1,:2],
        #             arrowprops=dict(arrowstyle="<-", ec='gray'))


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
        # if plot_special_point:
        #     if rules == [FDGO, FDANTI, DELAYGO, DELAYANTI]:
        #         special_point = -(h[(FDANTI,'stim1')] - h[(DELAYANTI,'stim1')]) + h[(FDGO,'stim1')]
        #
        #     elif rules == [DMCGO, DMCNOGO, DMSGO, DMSNOGO]:
        #         special_point = -(h[(DMSGO,'stim1')] - h[(DMSNOGO,'stim1')]) + h[(DMCGO,'stim1')]
        #
        #     elif rules == [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]:
        #         special_point = np.concatenate(
        #             ((h[(CHOICEATTEND_MOD1,'stim1')] + h[(CHOICE_INT,'stim1')])/2,
        #             (h[(CHOICEATTEND_MOD2,'stim1')] + h[(CHOICE_INT,'stim1')])/2), axis=0)
        #
        #     else:
        #         ValueError('Special points not provided')
        #
        # if plot_special_point:
        #     assert dim_reduction_type == 'PCA'
        #     special_point_trans = model.transform(special_point)
        #     ax.plot(special_point_trans[:,dim0], special_point_trans[:,dim1], '*',
        #             color=sns.xkcd_palette(['black'])[0], markersize=4)


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
        plt.show()


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
    epoch_map = dict(zip(['stim1', 'stim2', 'delay1', 'delay2', 'go1'], range(5)))
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

def temp_quantify_composition():
    save_addon = 'allrule_weaknoise_360'
    tsa = TaskSetAnalysis(save_addon)

    epochs = ['stim1']
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

    shuffle = False
    h2 = OrderedDict()
    for key in h_keys:
        h2[key] = h[key][-1]
        if shuffle:
            # h2[key] = np.random.permutation(h2[key])
            h2[key] = h2[key] * np.random.uniform(0.5,1.5)

    vec_diffs = list()
    vec_ids = list()
    for i in range(n_epochs):
        for j in range(n_epochs):
            if i != j:
                hi = h2[h_keys[i]]
                hj = h2[h_keys[j]]

                vec_diffs.append(hi-hj)
                vec_ids.append([i, j])

    n_diff = len(vec_diffs)


    from scipy.stats import linregress
    from numpy.linalg import norm

    #==============================================================================
    # tmps = list()
    # tmp_ids = list()
    # for i in range(n_diff-1):
    #     for j in range(i+1, n_diff):
    #         # tmps.append(angle_between(vec_diffs[i], vec_diffs[j]))
    #
    #         # slope, intercept, r_value, p_value, std_err = linregress(vec_diffs[i],vec_diffs[j])
    #         # tmps.append(p_value)
    #
    #         # tmps.append(norm(vec_diffs[i]-vec_diffs[j]))
    #         tmps.append(norm(vec_diffs[i]-vec_diffs[j])/np.sqrt(norm(vec_diffs[i])*norm(vec_diffs[i])))
    #         tmp_ids.append(vec_ids[i] + vec_ids[j])
    # tmps = np.array(tmps)
    # # _ = plt.hist(tmps, bins=200)
    #
    #
    # hist, bins_edge = np.histogram(tmps, bins=200)
    # plt.plot((bins_edge[:-1]+bins_edge[1:])/2, hist)
    # # plt.plot((bins_edge0[:-1]+bins_edge0[1:])/2, hist0, 'red')
    #
    #
    # ind_sort = np.argsort(tmps)
    # for i in ind_sort[:100]:
    #     print('')
    #     print('{:0.4f}'.format(tmps[i])),
    #     for j in tmp_ids[i]:
    #         print('{:15s}'.format(rule_name[h_keys[j][0]])),
    #
    #
    # aa = h2[(FDGO,'stim1')]-h2[(FDANTI,'stim1')]
    # bb = h2[(DELAYGO,'stim1')]-h2[(DELAYANTI,'stim1')]
    #
    # aa = h2[(DMCGO,'stim1')]-h2[(DMCNOGO,'stim1')]
    # bb = h2[(DMSGO,'stim1')]-h2[(DMSNOGO,'stim1')]
    #
    # aa = h2[(CHOICE_INT,'stim1')]-h2[(CHOICE_MOD2,'stim1')]
    # bb = h2[(CHOICE_INT,'stim1')]-h2[(CHOICEATTEND_MOD2,'stim1')]
    #==============================================================================


    tmps = list()
    tmp_ids = list()
    for i in range(n_epochs):
        ki = h_keys[i]
        hi = h2[ki]
        for j in range(i+1, n_epochs):
            kj = h_keys[j]
            hj = h2[kj]
            for k in range(j+1, n_epochs):
                kk = h_keys[k]
                hk = h2[kk]
                for l in range(k+1, n_epochs):
                    kl = h_keys[l]
                    hl = h2[kl]
                    hij = hi - hj
                    hkl = hk - hl
                    tmp = norm(hij - hkl)/np.sqrt(norm(hij)*norm(hkl))
                    tmps.append(tmp)
                    # tmp_ids.append([h_keys[m][0] for m in [i,j,k,l]])
                    tmp_ids.append([ki[0],kj[0],kk[0],kl[0]])

    ind_sort = np.argsort(tmps)
    for i in ind_sort[:10]:
        print('')
        print('{:0.4f}'.format(tmps[i])),
        for j in tmp_ids[i]:
            print('{:15s}'.format(rule_name[j])),

    for i in ind_sort[:10]:
        rules = tmp_ids[i]
        tsa.plot_taskspace(rules=rules, epochs=['stim1'], plot_text=True, figsize=(1.5,1.5),
                           markersize=3, plot_label=False, dim_reduction_type='PCA', get_lasttimepoint=True)

def plot_weight_rule_PCA(save_addon):
    # save_addon = 'allrule_weaknoise_480'
    with Run(save_addon, sigma_rec=0) as R:
        w_in  = R.w_in # for later sorting
        w_out = R.w_out
        config = R.config
    nx, nh, ny = config['shape']
    n_ring = config['N_RING']

    rules_list = [[FDGO, FDANTI, DELAYGO, DELAYANTI],
                  [CHOICEDELAY_INT, CHOICEATTEND_MOD1, CHOICE_INT, CHOICEDELAYATTEND_MOD1],
                  [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2],
                  [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]]

    for rules in rules_list:
        w_rules = w_in[:, 2*n_ring+1+np.array(rules)]

        from sklearn.decomposition import PCA
        model = PCA(n_components=5)

        # Transform data
        data_trans = model.fit_transform(w_rules.T)

        plt.figure(figsize=(2,2))
        for i, rule in enumerate(rules):
            plt.scatter(data_trans[i,0],data_trans[i,1])
            plt.text(data_trans[i,0],data_trans[i,1], rule_name[rule])
            plt.axis('equal')
        plt.show()


def compute_taskspace_obsolete(save_addon=None, save_type=None, save_type_end=None, setup=1):
    # save_type = 'allrule_softplus'
    # save_type_end = 'largeinput'
    # setup = 2
    if setup == 1:
        rules = [FDGO, FDANTI, DELAYGO, DELAYANTI]
    elif setup == 2:
        rules = [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    elif setup == 3:
        rules = [DMSGO, DMCGO, DMSNOGO, DMCNOGO]
    elif setup == 4:
        rules = [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
    elif setup == 5:
        rules = [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT, CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,
             CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT, CHOICE_MOD1, CHOICE_MOD2,]
    elif setup == 6:
        rules = [FDGO, DELAYGO, CHOICEATTEND_MOD1, CHOICEDELAYATTEND_MOD1]

    if save_addon is not None:
        tsa = TaskSetAnalysis(save_addon, rules=rules)
        h_trans = tsa.compute_taskspace(rules=rules, epochs=['stim1'],
                                        dim_reduction_type='PCA', setup=setup)
        return h_trans

    vars = range(0,1000)
    h_trans_all = dict()
    i = 0
    for var in vars:
        save_addon = save_type+'_'+str(var)
        if save_type_end is not None:
            save_addon = save_addon + save_type_end
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue
        i += 1

        tsa = TaskSetAnalysis(save_addon, rules=rules)
        h_trans = tsa.compute_taskspace(rules=rules, epochs=['stim1'],
                                        dim_reduction_type='PCA', setup=setup)
        if i == 1:
            for key, val in h_trans.iteritems():
                h_trans_all[key] = val
        else:
            for key, val in h_trans.iteritems():
                h_trans_all[key] = np.concatenate((h_trans_all[key], val), axis=0)
    return h_trans_all

def compute_taskspace(save_addon, setup, restore=False):
    if setup == 1:
        rules = [FDGO, FDANTI, DELAYGO, DELAYANTI]
    elif setup == 2:
        rules = [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    elif setup == 3:
        rules = [DMSGO, DMCGO, DMSNOGO, DMCNOGO]
    elif setup == 4:
        rules = [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
    elif setup == 5:
        rules = [CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT, CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,
             CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT, CHOICE_MOD1, CHOICE_MOD2,]
    elif setup == 6:
        rules = [FDGO, DELAYGO, CHOICEATTEND_MOD1, CHOICEDELAYATTEND_MOD1]

    fname = 'taskset{:d}_space_'.format(setup)+save_addon+'.pkl'
    fname = os.path.join('data', fname)

    if restore and os.path.isfile(fname):
        print('Reloading results from '+fname)
        with open(fname, 'rb') as f:
            h_trans = pickle.load(f)

    else:
        tsa = TaskSetAnalysis(save_addon, rules=rules)
        h_trans = tsa.compute_taskspace(rules=rules, epochs=['stim1'],
                                        dim_reduction_type='PCA', setup=setup)
        with open(fname, 'wb') as f:
            pickle.dump(h_trans, f)
        print('Results stored at : '+fname)

    return h_trans


def _plot_taskspace(h_trans, save_name='temp', plot_example=False, lxy=None,
                    plot_arrow=True, **kwargs):
    from performance import rule_color
    figsize = (1.7,1.7)
    fs = 7 # fontsize
    dim0, dim1 = (0, 1) # plot dimensions
    i_example = 0 # index of the example to plot

    texts = list()

    maxv0, maxv1 = -1, -1

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.2, 0.2, 0.65, 0.65])

    for key, val in h_trans.iteritems():
        rule, epoch = key
        # Default coloring by rule_color
        color = np.array(rule_color[rule])

        if plot_example:
            xplot, yplot = val[i_example,dim0], val[i_example,dim1]
        else:
            xplot, yplot = val[:,dim0], val[:,dim1]

        ax.plot(xplot, yplot, 'o',
                color=color, mec=color*1, mew=1.0, ms=2)


        xtext = np.mean(val[:,dim0])
        if np.mean(val[:,dim1])>0:
            ytext = np.max(val[:,dim1])
            va = 'bottom'
        else:
            ytext = np.min(val[:,dim1])
            va = 'top'

        texts.append(ax.text(xtext*1.1, ytext*1.1, rule_name[rule],
                             fontsize=6, color=color,
                             horizontalalignment='center', verticalalignment=va))

        maxv0 = np.max([maxv0, np.max(abs(val[:,dim0]))])
        maxv1 = np.max([maxv1, np.max(abs(val[:,dim1]))])

    if kwargs['setup'] == 1:
        arrow_starts = [h_trans[(FDGO,'stim1')], h_trans[(FDANTI,'stim1')]]
        arrow_ends   = [h_trans[(DELAYGO,'stim1')], h_trans[(DELAYANTI,'stim1')]]
    elif kwargs['setup'] == 2:
        arrow_starts = [h_trans[(CHOICEATTEND_MOD1,'stim1')], h_trans[(CHOICEDELAYATTEND_MOD1,'stim1')]]
        arrow_ends   = [h_trans[(CHOICEATTEND_MOD2,'stim1')], h_trans[(CHOICEDELAYATTEND_MOD2,'stim1')]]
    else:
        plot_arrow = False

    if plot_arrow:
        for arrow_start, arrow_end in zip(arrow_starts, arrow_ends):
            if plot_example:
                a_start = arrow_start[i_example,[dim0, dim1]]
                a_end = arrow_end[i_example,[dim0, dim1]]
            else:
                a_start = arrow_start[:,[dim0, dim1]].mean(axis=0)
                a_end = arrow_end[:,[dim0, dim1]].mean(axis=0)
            ax.annotate("", xy=a_start, xytext=a_end,
                arrowprops=dict(arrowstyle="<-", ec='gray'))

    if lxy is None:
        lx = np.ceil(maxv0)
        ly = np.ceil(maxv1)
    else:
        lx, ly = lxy

    ax.tick_params(axis='both', which='major', labelsize=fs)
    # plt.locator_params(nbins=3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.margins(0.1)
    # plt.axis('equal')
    plt.xlim([-lx,lx])
    plt.ylim([-ly,ly])
    ax.plot([0,0], [-ly,ly], '--', color='gray')
    ax.plot([-lx,lx], [0,0], '--', color='gray')
    ax.set_xticks([-lx,lx])
    ax.set_yticks([-ly,ly])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if kwargs['setup'] == 1:
        pc_name = 'rPC'
    else:
        pc_name = 'PC'
    ax.set_xlabel(pc_name+' {:d}'.format(dim0+1), fontsize=fs, labelpad=-5)
    ax.set_ylabel(pc_name+' {:d}'.format(dim1+1), fontsize=fs, labelpad=-5)

    if save:
        plt.savefig(os.path.join('figure', save_name+'.pdf'), transparent=True)
    plt.show()

    return (lx, ly)

def plot_taskspace(save_addon, setup=1, restore=True):
    h_trans = compute_taskspace(save_addon, setup, restore=restore)
    save_name = 'taskset{:d}_space_'.format(setup)+save_addon
    _plot_taskspace(h_trans, save_name, setup=setup)

def plot_taskspace_group(save_type, save_type_end=None, setup=1, restore=True, flip_sign=True):

    save_addons, _ = get_valid_saveaddons(save_type, save_type_end)

    h_trans_all = OrderedDict()
    i = 0
    for save_addon in save_addons:
        h_trans = compute_taskspace(save_addon, setup, restore=restore)

        if flip_sign:
            if setup != 1:
                # # The first data point should have all positive coordinate values
                signs = ((h_trans.values()[0]>0)*2.-1)
                for key, val in h_trans.iteritems():
                    h_trans[key] = val*signs
            else:
                # When PC1 and PC2 capture similar variances, allow for a rotation
                # rotation_matrix, clock wise
                get_angle = lambda vec : np.arctan2(vec[1], vec[0])
                theta = get_angle(h_trans.values()[0][0])
                # theta = 0
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]])

                for key, val in h_trans.iteritems():
                    h_trans[key] = np.dot(val, rot_mat)

                if get_angle(h_trans.values()[1][0]) < 0:
                    for key, val in h_trans.iteritems():
                        h_trans[key] = val*np.array([1, -1])

        if i == 0:
            for key, val in h_trans.iteritems():
                h_trans_all[key] = val
        else:
            for key, val in h_trans.iteritems():
                h_trans_all[key] = np.concatenate((h_trans_all[key], val), axis=0)
        i += 1

    save_name = 'taskset{:d}_space_'.format(setup)+save_type
    if save_type_end is not None:
        save_name = save_name + save_type_end

    lxy = _plot_taskspace(h_trans_all, save_name, setup=setup)
    save_name = save_name + '_example'
    lxy = _plot_taskspace(h_trans_all, save_name, setup=setup, plot_example=True, lxy=lxy)

def run_network_replacerule(save_addon, rule, replace_rule, rule_strength):
    '''
    Run the network but with replaced rule input weight
    :param rule: the rule to run
    :param rule_X: A numpy array of rules, whose values will be used to replace
    :param beta: the weights for each rule_X vector used.
    If beta='fit', use the best linear fit

    The rule input connection will be replaced by
    sum_i rule_connection(rule_X_i) * beta_i
    '''
    from network import get_perf

    with Run(save_addon, fast_eval=True) as R:
        config = R.config

        # Get performance
        batch_size_test = 1000
        n_rep = 20
        batch_size_test_rep = int(batch_size_test/n_rep)
        perf_rep = list()
        for i_rep in range(n_rep):
            task = generate_onebatch(rule, config, 'random', batch_size=batch_size_test_rep,
                                     replace_rule=replace_rule, rule_strength=rule_strength)
            h = R.f_h(task.x)
            y_hat = R.f_y(h)
            perf = get_perf(y_hat, task.y_loc)
            perf_rep.append(perf.mean())

    return np.mean(perf_rep), rule_strength

def replace_rule_name(replace_rule, rule_strength):
    # little helper function
    name = ''
    counter = 0
    for r, b in zip(replace_rule, rule_strength):
        if b != 0:

            if b == 1:
                if counter==0:
                    prefix = ''
                else:
                    prefix = '+'
            elif b == -1:
                prefix = '-'
            else:
                prefix = '{:+d}'.format(b)
            name += prefix + rule_name[r] + '\n'
            counter += 1
    # get rid of the last \n
    name = name[:-1]
    return name

def compute_replacerule_performance(save_addon, setup, restore=False):
    #Compute the performance of one task given a replaced rule input

    if setup == 1:
        rule = DELAYANTI
        replace_rule = np.array([DELAYANTI, FDANTI, DELAYGO, FDGO])

        rule_strengths = \
            [[1,0,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,1,1,-1]]

    elif setup == 2:
        rule = CHOICEDELAYATTEND_MOD1
        replace_rule = np.array([CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2,
                                 CHOICEATTEND_MOD1, CHOICEATTEND_MOD2])

        rule_strengths = \
            [[1,0,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,0,1,0],
             [0,1,1,-1]]

    elif setup == 3:
        rule = DMSGO
        replace_rule = np.array([DMSGO, DMCGO, DMSNOGO, DMCNOGO])
        rule_strengths = \
            [[1,0,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,1,1,-1]]

    else:
        raise ValueError('Unknown setup value')

    fname = 'taskset{:d}_perf_'.format(setup)+save_addon+'.pkl'
    fname = os.path.join('data', fname)

    if restore and os.path.isfile(fname):
        print('Reloading results from '+fname)
        with open(fname, 'rb') as f:
            r = pickle.load(f)
        perfs, rule, names = r['perfs'], r['rule'], r['names']

    else:
        perfs = list()
        names = list()
        for rule_strength in rule_strengths:
            perf, _ = run_network_replacerule(save_addon, rule, replace_rule, rule_strength)
            perfs.append(perf)
            names.append(replace_rule_name(replace_rule, rule_strength))

        perfs = np.array(perfs)
        print(perfs)

        results = {'perfs':perfs, 'rule':rule, 'names':names}
        with open(fname, 'wb') as f:
            pickle.dump(results, f)
        print('Results stored at : '+fname)

    return perfs, rule, names

def _plot_replacerule_performance(perfs, rule, names, setup, perfs_all=None, save_name=None):
    save = True

    fs = 7
    width = 0.3
    fig = plt.figure(figsize=(1.6,2.2))
    ax = fig.add_axes([0.55,0.05,0.35,0.7])
    b0 = ax.barh(np.arange(len(perfs))-width/2, perfs[::-1],
           height=width, color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
    if perfs_all is not None:
        n_net = perfs_all.shape[0]
        for i in range(len(perfs)):
            ax.plot(perfs_all[:,-i-1], [i]*n_net, 'o',
                    color=sns.xkcd_palette(['cerulean'])[0], alpha=0.3, markersize=2)
    ax.set_yticks(np.arange(len(perfs)))
    ax.set_yticklabels(names[::-1], rotation=0, horizontalalignment='right')
    ax.set_ylabel('Rule input', fontsize=fs, labelpad=3)
    # ax.set_ylabel('performance', fontsize=fs)
    title = 'Performance on\n'+rule_name[rule]
    if perfs_all is not None:
        title = title + ' (n={:d})'.format(n_net)
    ax.set_title(title, fontsize=fs, y=1.13)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.grid(True)
    ax.set_xticks([0,0.5,1.0])
    ax.set_xlim([0,1.05])
    ax.set_ylim([-0.5,len(perfs)-0.5])
    if save_name is None:
        save_name = 'taskset{:d}_perf_'.format(setup)
    if save:
        plt.savefig(os.path.join('figure', save_name+'.pdf'), transparent=True)
    plt.show()

def plot_replacerule_performance(save_addon, setup, perfs_all=None, save_name=None, restore=True):
    perfs, rule, names = compute_replacerule_performance(save_addon, setup, restore)
    _plot_replacerule_performance(perfs, rule, names, setup, perfs_all, save_name)


def plot_replacerule_performance_group(save_type, save_type_end=None, setup=1, restore=True):
    save_addons, _ = get_valid_saveaddons(save_type, save_type_end)

    perfs_plot = list()
    for save_addon in save_addons:
        perfs, rule, names = compute_replacerule_performance(save_addon, setup, restore)
        perfs_plot.append(perfs)

    perfs_plot = np.array(perfs_plot)
    perfs_median = np.median(perfs_plot, axis=0)

    save_name = 'taskset{:d}_perf_'.format(setup)+save_type
    if save_type_end is not None:
        save_name = save_name + save_type_end

    print(perfs_median)
    _plot_replacerule_performance(perfs_median, rule, names, setup, perfs_all=perfs_plot, save_name=save_name)


if __name__ == '__main__':
    # save_type = 'allrule_softplus'
    # save_type_end = 'largeinput'
    # setup = 1
    # compute_and_plot_replacerule_performance_type(save_type, setup, save_type_end=None)

    save_addon = 'allrule_softplus_2_200tasksetmon'
    # plot_taskspaces(save_addon, get_lasttimepoint=True)
    # plot_weight_rule_PCA(save_addon)
    # for setup in range(7):

    # plot_dim()
    # plot_dimpair()

    # save_addon = 'choicefamily_softplus_340'
    # rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2]
    # rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]
    # rules = [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

    save_addon = 'allrule_relu_0main'
    # rules = [CHOICE_INT, CHOICEDELAY_INT, CHOICEATTEND_MOD1, CHOICEDELAYATTEND_MOD1]
    # tsa = TaskSetAnalysis(save_addon, rules=rules)
    # tsa.plot_taskspace(rules=rules,
    #                    epochs=['stim1'], plot_text=True,
    #                            save_append='type'+str(4), figsize=(1.5,1.5),
    #                            markersize=3, plot_label=False, dim_reduction_type='PCA',
    #                            plot_special_point=False, plot_arrow=False, get_lasttimepoint=True)
    # plot_weight_rule_PCA(save_addon)


    ######################### Plotting task space for all tasks ##################
    save_addon = 'allrule_softplus_0_256paper'
    tsa = TaskSetAnalysis(save_addon)
    tsa.compute_and_plot_taskspace(epochs=['stim1'], dim_reduction_type='PCA')
    # tsa.compute_and_plot_taskspace(epochs=['stim1', 'delay1', 'go1'], dim_reduction_type='MDS')

    
    # for feature in features:
    #     tsa.compute_and_plot_taskspace(epochs=['stim1'], plot_text=False, color_by_feature=True,
    #                        feature=feature, figsize=(1.0,1.0), markersize=2, plot_label=False)

    ##################### Plotting task space with selected tasks #################
    ################# & Plotting performance with replaced rule #################

    st, ste = 'allrule_softplus', '_256paper'
    # setups = [1] # Go, Anti family
    # setups = [2] # Ctx DM family
    setups = [1, 2]
    # setups = [3]
    for setup in setups:
        pass
        # plot_taskspace_group(save_type=st, save_type_end=ste, setup=setup, restore=True)
        # plot_replacerule_performance_group(save_type=st, save_type_end=ste, setup=setup, restore=True)

    #################### Plotting trajectories in task space #####################
    save_addon = 'allrule_softplus_0_300mainfri'
#==============================================================================
#     rules = []
#     rules+= [REACTGO, FDGO, DELAYGO]
#     rules+= [ANTI, FDANTI, DELAYANTI]
#     # rules += [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2]
#     # rules+= [CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
#     # rules+= [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT]
#     tsa = TaskSetAnalysis(save_addon, rules=rules)
#     h_trans = tsa.compute_taskspace(rules=rules, epochs=None,
#                                     dim_reduction_type='PCA', get_lasttimepoint=False)
# 
#     from performance import rule_color
#     dim0, dim1 = (0, 1) # plot dimensions
# 
#     plt.figure()
#     for key, val in h_trans.iteritems():
#         rule, epoch = key
#         if epoch in ['stim1', 'stim2']:
#             lw = 1
#         elif epoch in ['delay1', 'delay2']:
#             lw = 0.5
#         elif epoch == 'go1':
#             lw = 2
#         else:
#             raise NotImplementedError()
#         color = np.array(rule_color[rule])
#         plt.plot(val[:, dim0], val[:, dim1], color=color, lw=lw)
#==============================================================================

