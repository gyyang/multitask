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
        # rules = [INHGO, INHREMAP, DELAYGO, DELAYREMAP]
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
                       plot_special_point=False, plot_arrow=False, **kwargs):
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

        if plot_arrow:
            if rules == [INHGO, INHREMAP, DELAYGO, DELAYREMAP]:
                arrow_starts = [h[(INHREMAP,'tar1')], h[(INHGO,'tar1')]]
                arrow_ends   = [h[(DELAYREMAP,'tar1')],
                                -(h[(INHREMAP,'tar1')] - h[(DELAYREMAP,'tar1')]) + h[(INHGO,'tar1')]]
            elif rules == [DMCGO, DMCNOGO, DMSGO, DMSNOGO]:
                arrow_starts = [h[(DMSGO,'tar1')], h[(DMCGO,'tar1')]]
                arrow_ends   = [h[(DMSNOGO,'tar1')],
                                -(h[(DMSGO,'tar1')] - h[(DMSNOGO,'tar1')]) + h[(DMCGO,'tar1')]]

            elif rules == [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]:
                arrow_starts = [h[(CHOICE_INT,'tar1')], h[(CHOICE_INT,'tar1')]]
                arrow_ends   = [h[(CHOICEATTEND_MOD1,'tar1')], h[(CHOICEATTEND_MOD2,'tar1')]]
            else:
                ValueError('Arrows not provided')

            for arrow_start, arrow_end in zip(arrow_starts, arrow_ends):
                arrow_start = model.transform(arrow_start)
                arrow_end   = model.transform(arrow_end)

                ax.annotate("", xy=arrow_start[-1,:2], xytext=arrow_end[-1,:2],
                    arrowprops=dict(arrowstyle="<-", ec='gray'))


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
        if plot_special_point:
            if rules == [INHGO, INHREMAP, DELAYGO, DELAYREMAP]:
                special_point = -(h[(INHREMAP,'tar1')] - h[(DELAYREMAP,'tar1')]) + h[(INHGO,'tar1')]

            elif rules == [DMCGO, DMCNOGO, DMSGO, DMSNOGO]:
                special_point = -(h[(DMSGO,'tar1')] - h[(DMSNOGO,'tar1')]) + h[(DMCGO,'tar1')]

            elif rules == [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]:
                special_point = np.concatenate(
                    ((h[(CHOICEATTEND_MOD1,'tar1')] + h[(CHOICE_INT,'tar1')])/2,
                    (h[(CHOICEATTEND_MOD2,'tar1')] + h[(CHOICE_INT,'tar1')])/2), axis=0)

            else:
                ValueError('Special points not provided')

        if plot_special_point:
            assert dim_reduction_type == 'PCA'
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
                           markersize=3, plot_label=False, dim_reduction_type='PCA',
                           plot_special_point=True, plot_arrow=True, **kwargs)

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

def temp_quantify_composition():
    save_addon = 'allrule_weaknoise_360'
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
    # aa = h2[(INHGO,'tar1')]-h2[(INHREMAP,'tar1')]
    # bb = h2[(DELAYGO,'tar1')]-h2[(DELAYREMAP,'tar1')]
    #
    # aa = h2[(DMCGO,'tar1')]-h2[(DMCNOGO,'tar1')]
    # bb = h2[(DMSGO,'tar1')]-h2[(DMSNOGO,'tar1')]
    #
    # aa = h2[(CHOICE_INT,'tar1')]-h2[(CHOICE_MOD2,'tar1')]
    # bb = h2[(CHOICE_INT,'tar1')]-h2[(CHOICEATTEND_MOD2,'tar1')]
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
        tsa.plot_taskspace(rules=rules, epochs=['tar1'], plot_text=True, figsize=(1.5,1.5),
                           markersize=3, plot_label=False, dim_reduction_type='PCA', get_lasttimepoint=True)

def plot_weight_rule_PCA():
    save_addon = 'allrule_weaknoise_480'
    with Run(save_addon, sigma_rec=0) as R:
        w_in  = R.w_in # for later sorting
        w_out = R.w_out
        config = R.config
    nx, nh, ny = config['shape']
    n_ring = config['N_RING']

    rules_list = [[INHGO, INHREMAP, DELAYGO, DELAYREMAP],
                      [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT],
                      [DMCGO, DMCNOGO, DMSGO, DMSNOGO]]

    for rules in rules_list:
        w_rules = w_in[:, 2*n_ring+1+np.array(rules)]

        from sklearn.decomposition import PCA
        model = PCA(n_components=5)

        # Transform data
        data_trans = model.fit_transform(w_rules.T)

        plt.figure(figsize=(3,3))
        for i, rule in enumerate(rules):
            plt.scatter(data_trans[i,0],data_trans[i,1])
            plt.text(data_trans[i,0],data_trans[i,1], rule_name[rule])
            plt.axis('equal')

save_addon = 'allrule_weaknoise_400'
# plot_taskspaces(save_addon, get_lasttimepoint=True)
# plot_taskspaces(save_addon, get_timeaverage=True)

# plot_dim()
# plot_dimpair()


# tsa.plot_taskspace(epochs=['tar1', 'delay1', 'go1'], **kwargs)

# tsa.plot_taskspace(epochs=['tar1'], plot_text=True, figsize=(3.5,3.5), **kwargs)

# for feature in features:
#     tsa.plot_taskspace(epochs=['tar1'], plot_text=False, color_by_feature=True,
#                        feature=feature, figsize=(1.0,1.0), markersize=2, plot_label=False, **kwargs)



#==============================================================================
# save_addon = 'allrule_weaknoise_400'
# import tensorflow as tf
# from network import get_perf
# 
# n_rep = 1
# n_tar_loc = 20 # increase repeat by increasing this
# batch_size = n_rep * n_tar_loc**2
# batch_shape = (n_rep, n_tar_loc,n_tar_loc)
# ind_rep, ind_tar_loc1, ind_tar_loc2 = np.unravel_index(range(batch_size),batch_shape)
# 
# # Looping target location
# tar1_locs = 2*np.pi*ind_tar_loc1/n_tar_loc
# tar2_locs = 2*np.pi*ind_tar_loc2/n_tar_loc
# 
# params = {'tar1_locs' : tar1_locs,
#           'tar2_locs' : tar2_locs}
# 
# rule = DMSGO
# # rule = DMCGO
# 
# rule_y = np.array([rule])
# 
# with Run(save_addon, fast_eval=True) as R:
#     config = R.config
#     nx, nh, ny = config['shape']
#     n_ring = config['N_RING']
# 
#     # rule_X = np.array([INHGO, DELAYGO, INHREMAP])
#     rule_X = np.array([DMSNOGO, DMSGO, DMCNOGO])
# 
#     w_rule_X = w_rec_[2*n_ring+1+rule_X, :]
#     w_rule_y = w_rec_[2*n_ring+1+rule_y, :]
#     beta = np.dot(w_rule_y, np.linalg.pinv(w_rule_X))
#     # beta = np.array([-1,1,1])
#     beta = np.array([0,1,0])
#     w_rec_[2*n_ring+1+rule, :] = np.dot(beta, w_rule_X)
#     
# 
#     change_w_rec = tf.trainable_variables()[3].assign(w_rec_)
#     R.run(change_w_rec)
# 
#     task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
#     y_sample = R.f_y_from_x(task.x)
#     perf = get_perf(y_sample, task.y_loc)
#     
# print(perf.mean())
#     
# if rule in [DMSGO, DMCGO]:
#     match_response = y_sample[-1, :, 0] < 0.5 # Last time point, fixation unit, match if go
# elif rule in [DMSNOGO, DMCNOGO]:
#     match_response = y_sample[-1, :, 0] > 0.5
# match_response = match_response.reshape(batch_shape)
# match_response = match_response.mean(axis=0)
# 
# kwargs = dict()
# fs = 6
# fig = plt.figure(figsize=(1.5,1.5))
# ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
# im = ax.imshow(match_response, cmap='BrBG', origin='lower',
#                aspect='auto', interpolation='nearest', vmin=0, vmax=1)
# ax.set_xlabel('Mod 2 loc.', fontsize=fs, labelpad=-3)
# plt.xticks([0, n_tar_loc-1], ['0', '360'],
#            rotation=0, va='center', fontsize=fs)
# if 'ylabel' in kwargs and kwargs['ylabel']==False:
#     plt.yticks([])
# else:
#     ax.set_ylabel('Mod 1 loc.', fontsize=fs, labelpad=-3)
#     plt.yticks([0, n_tar_loc-1], [0, 360],
#                rotation=0, va='center', fontsize=fs)
# ax.tick_params('both', length=0)
# for loc in ['bottom','top','left','right']:
#     ax.spines[loc].set_visible(False)
# 
# if 'colorbar' in kwargs and kwargs['colorbar']==False:
#     pass
# else:
#     ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
#     cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
#     cb.outline.set_linewidth(0.5)
#     cb.set_label('Prop. of match', fontsize=fs, labelpad=-3)
#     plt.tick_params(axis='both', which='major', labelsize=fs)
# 
# # plt.savefig('figure/'+rule_name[rule].replace(' ','')+
# #             '_perf2D_lesion'+str(lesion_group)+
# #             self.save_addon+'.pdf', transparent=True)
# plt.show()
#==============================================================================






def run_network_replacerule(save_addon, rule, rule_X, beta):
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
    from run import replacerule

    with Run(save_addon, fast_eval=True) as R:
        config = R.config

        beta = replacerule(R, rule, rule_X, beta)

        # Get performance
        batch_size_test = 2000
        n_rep = 20
        batch_size_test_rep = int(batch_size_test/n_rep)
        perf_rep = list()
        for i_rep in range(n_rep):
            task = generate_onebatch(rule, config, 'random', batch_size=batch_size_test_rep)
            h = R.f_h(task.x)
            y_hat = R.f_y(h)
            perf = get_perf(y_hat, task.y_loc)
            perf_rep.append(perf.mean())

    return np.mean(perf_rep)


save_addon = 'allrule_weaknoise_400'

rule = DELAYREMAP
rule_X = np.array([INHGO, DELAYGO, INHREMAP])
beta = np.array([-1,1,1])

# rule = DMCGO
# rule_X = np.array([DMSNOGO, DMSGO, DMCNOGO])
# beta = np.array([-1,1,1])
# beta = np.array([0,1,0])

perf = run_network_replacerule(save_addon, rule, rule_X, beta)
print(perf)