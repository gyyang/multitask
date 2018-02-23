"""
Task set analysis
Analyze the state-space of stimulus-averaged activity
"""

from __future__ import division

import os
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from task import rule_name
from task import generate_trials
from network import Model
from network import get_perf
import tools

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
    """Analyzing the representation of tasks."""

    def __init__(self, model_dir, rules=None):
        """Initialization.

        Args:
            model_dir: str, model directory
            rules: None or a list of rules
        """
        # Stimulus-averaged traces
        h_stimavg_byrule  = OrderedDict()
        h_stimavg_byepoch = OrderedDict()
        # Last time points of epochs
        h_lastt_byepoch   = OrderedDict()

        model = Model(model_dir)
        hparams = model.hparams

        if rules is None:
            # Default value
            rules = hparams['rules']
        n_rules = len(rules)

        with tf.Session() as sess:
            model.restore()

            for rule in rules:
                trial = generate_trials(rule=rule, hparams=hparams, mode='test')
                feed_dict = tools.gen_feed_dict(model, trial, hparams)
                h = sess.run(model.h, feed_dict=feed_dict)

                # Average across stimulus conditions
                h_stimavg = h.mean(axis=1)

                # dt_new = 50
                # every_t = int(dt_new/hparams['dt'])

                t_start = int(500/hparams['dt']) # Important: Ignore the initial transition
                # Average across stimulus conditions
                h_stimavg_byrule[rule] = h_stimavg[t_start:, :]

                for e_name, e_time in trial.epochs.items():
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
        self.model_dir = model_dir

    @staticmethod
    def filter(h, rules=None, epochs=None, non_rules=None, non_epochs=None,
               get_lasttimepoint=True, get_timeaverage=False, **kwargs):
        # h should be a dictionary
        # get a new dictionary containing keys from the list of rules and epochs
        # And avoid epochs from non_rules and non_epochs
        # h_new = OrderedDict([(key, val) for key, val in h.items() if key[1] in epochs])

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
            model = TSNE(n_components=2, init='pca',
             verbose=1, method='exact', learning_rate=100, perplexity=5)

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
        for key, val in h.items():
            i_end = i_start + val.shape[0]
            h_trans[key] = data_trans[i_start:i_end, :]
            i_start = i_end

        return h_trans

    def obsolete_compute_and_plot_taskspace(self,
               rules=None, epochs=None, **kwargs):

        h_trans = self.compute_taskspace(rules=rules, epochs=epochs, **kwargs)
        self.plot_taskspace(h_trans, **kwargs)

    def obsolete_plot_taskspace(self, h_trans, epochs=None, dim_reduction_type='MDS',
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
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

        # if plot_arrow:
        #     if rules == ['fdgo', 'fdanti', 'delaygo', 'delayanti']:
        #         arrow_starts = [h[('fdanti','stim1')], h[('fdgo','stim1')]]
        #         arrow_ends   = [h[('delayanti','stim1')],
        #                         -(h[('fdanti','stim1')] - h[('delayanti','stim1')]) + h[('fdgo','stim1')]]
        #     elif rules == ['dmcgo', 'dmcnogo', 'dmsgo', 'dmsnogo']:
        #         arrow_starts = [h[('dmsgo','stim1')], h[('dmcgo','stim1')]]
        #         arrow_ends   = [h[('dmsnogo','stim1')],
        #                         -(h[('dmsgo','stim1')] - h[('dmsnogo','stim1')]) + h[('dmcgo','stim1')]]
        #
        #     elif rules == ['contextdm1', 'contextdm2', 'dm1', 'dm2', 'multidm']:
        #         arrow_starts = [h[('multidm','stim1')], h[('multidm','stim1')]]
        #         arrow_ends   = [h[('contextdm1','stim1')], h[('contextdm2','stim1')]]
        #     else:
        #         ValueError('Arrows not provided')
        #
        #     for arrow_start, arrow_end in zip(arrow_starts, arrow_ends):
        #         arrow_start = model.transform(arrow_start)
        #         arrow_end   = model.transform(arrow_end)
        #
        #         ax.annotate("", xy=arrow_start[-1,:2], xytext=arrow_end[-1,:2],
        #             arrowprops=dict(arrowstyle="<-", ec='gray'))


        for key, val in h_trans.items():
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
            if dim_reduction_type == 'PCA':
                xlabel = 'PC {:d}'.format(dim0+1)
                ylabel = 'PC {:d}'.format(dim1+1)
            else:
                xlabel = dim_reduction_type + ' dim. {:d}'.format(dim0+1)
                ylabel = dim_reduction_type + ' dim. {:d}'.format(dim1+1)
            ax.set_xlabel(xlabel, fontsize=fs)
            ax.set_ylabel(ylabel, fontsize=fs)
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
        #     if rules == ['fdgo', 'fdanti', 'delaygo', 'delayanti']:
        #         special_point = -(h[('fdanti','stim1')] - h[('delayanti','stim1')]) + h[('fdgo','stim1')]
        #
        #     elif rules == ['dmcgo', 'dmcnogo', 'dmsgo', 'dmsnogo']:
        #         special_point = -(h[('dmsgo','stim1')] - h[('dmsnogo','stim1')]) + h[('dmcgo','stim1')]
        #
        #     elif rules == ['contextdm1', 'contextdm2', 'dm1', 'dm2', 'multidm']:
        #         special_point = np.concatenate(
        #             ((h[('contextdm1','stim1')] + h[('multidm','stim1')])/2,
        #             (h[('contextdm2','stim1')] + h[('multidm','stim1')])/2), axis=0)
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
        for key, val in self.h_lastt_byepoch.items():
            self.dim_lastt_byepoch[key] = get_dim(val, **kwargs)

    def compute_dim_pair(self, **kwargs):
        # Compute dimension of each pair of epochs, and the dimension ratio

        print('Computing dimensions of pairs of rule/epochs')

        self.dimpair_lastt_byepoch = OrderedDict()
        self.dimpairratio_lastt_byepoch = OrderedDict()

        for key1, val1 in self.h_lastt_byepoch.items():
            for key2, val2 in self.h_lastt_byepoch.items():

                #TODO: TEMP
                val1 = val1 - val1.mean(axis=0)
                val2 = val2 - val2.mean(axis=0)

                h_pair = np.concatenate((val1, val2), axis=0)

                dim_pair = get_dim(h_pair, **kwargs)
                dim1, dim2 = self.dim_lastt_byepoch[key1], self.dim_lastt_byepoch[key2]

                self.dimpair_lastt_byepoch[(key1, key2)] = dim_pair
                self.dimpairratio_lastt_byepoch[(key1, key2)] = dim_pair/(dim1 + dim2)


def obsolete_plot_dim():
    model_dir = 'allrule_weaknoise_400'
    tsa = TaskSetAnalysis(model_dir)
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


def obsolete_plot_dimpair():
    model_dir = 'allrule_weaknoise_400'
    tsa = TaskSetAnalysis(model_dir)
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


def compute_taskspace(model_dir, setup, restore=False, representation='rate'):
    if setup == 1:
        rules = ['fdgo', 'fdanti', 'delaygo', 'delayanti']
    elif setup == 2:
        rules = ['contextdelaydm1', 'contextdelaydm2', 'contextdm1', 'contextdm2']
    elif setup == 3:
        rules = ['dmsgo', 'dmcgo', 'dmsnogo', 'dmcnogo']
    elif setup == 4:
        rules = ['contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                 'contextdm1', 'contextdm2', 'multidm']
    elif setup == 5:
        rules = ['contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
                 'delaydm1', 'delaydm2', 'contextdm1', 'contextdm2',
                 'multidm', 'dm1', 'dm2',]
    elif setup == 6:
        rules = ['fdgo', 'delaygo', 'contextdm1', 'contextdelaydm1']

    if representation == 'rate':
        fname = 'taskset{:d}_space'.format(setup)+'.pkl'
        fname = os.path.join(model_dir, fname)

        if restore and os.path.isfile(fname):
            print('Reloading results from '+fname)
            with open(fname, 'rb') as f:
                h_trans = pickle.load(f)

        else:
            tsa = TaskSetAnalysis(model_dir, rules=rules)
            h_trans = tsa.compute_taskspace(rules=rules, epochs=['stim1'],
                                            dim_reduction_type='PCA', setup=setup)
            with open(fname, 'wb') as f:
                pickle.dump(h_trans, f)
            print('Results stored at : '+fname)

    elif representation == 'weight':
        from task import get_rule_index

        model = Model(model_dir)
        hparams = model.hparams
        n_hidden = hparams['n_rnn']
        n_output = hparams['n_output']
        with tf.Session() as sess:
            model.restore()
            w_in = sess.run(model.w_in).T

        rule_indices = [get_rule_index(r, hparams) for r in rules]
        w_rules = w_in[:, rule_indices]

        from sklearn.decomposition import PCA
        model = PCA(n_components=2)

        # Transform data
        data_trans = model.fit_transform(w_rules.T)
        
        # Turn into dictionary, and consistent with previous code
        h_trans = OrderedDict()
        for i, r in enumerate(rules):
            # shape will be (1,2), and the key is added an epoch value only for consistency
            h_trans[(r,'stim1')] = np.array([data_trans[i]])

    else:
        raise ValueError()

    return h_trans


def _plot_taskspace(h_trans, fig_name='temp', plot_example=False, lxy=None,
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

    for key, val in h_trans.items():
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
        arrow_starts = [h_trans[('fdgo','stim1')], h_trans[('fdanti','stim1')]]
        arrow_ends   = [h_trans[('delaygo','stim1')],
                        h_trans[('delayanti','stim1')]]
    elif kwargs['setup'] == 2:
        arrow_starts = [h_trans[('contextdm1','stim1')],
                        h_trans[('contextdelaydm1','stim1')]]
        arrow_ends   = [h_trans[('contextdm2','stim1')],
                        h_trans[('contextdelaydm2','stim1')]]
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
        plt.savefig(os.path.join('figure', fig_name+'.pdf'), transparent=True)
    plt.show()

    return (lx, ly)


def plot_taskspace(model_dir, setup=1, restore=True, representation='rate'):
    h_trans = compute_taskspace(
        model_dir, setup, restore=restore, representation=representation)
    save_name = 'taskset{:d}_space'.format(setup)
    _plot_taskspace(h_trans, save_name, setup=setup)


def plot_taskspace_group(model_dir, setup=1, restore=True, representation='rate', flip_sign=True):
    '''Plot task space for a group of networks.

    Args:
        model_dir : the root directory for all models to analyse
        setup: int, the combination of rules to use
        restore: bool, whether to restore results
        representation: 'rate' or 'weight'
        flip_sign: bool, whether to flip signs for consistency
    '''

    model_dirs = tools.valid_model_dirs(model_dir)
    print('Analyzing models : ')
    print(model_dirs)

    h_trans_all = OrderedDict()
    i = 0
    for model_dir in model_dirs:
        h_trans = compute_taskspace(
            model_dir, setup, restore=restore, representation=representation)

        if flip_sign:
            if setup != 1:
                # # The first data point should have all positive coordinate values
                signs = ((h_trans.values()[0]>0)*2.-1)
                for key, val in h_trans.items():
                    h_trans[key] = val*signs
            else:
                # When PC1 and PC2 capture similar variances, allow for a rotation
                # rotation_matrix, clock wise
                get_angle = lambda vec : np.arctan2(vec[1], vec[0])
                theta = get_angle(h_trans.values()[0][0])
                # theta = 0
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta),  np.cos(theta)]])

                for key, val in h_trans.items():
                    h_trans[key] = np.dot(val, rot_mat)

                if get_angle(h_trans.values()[1][0]) < 0:
                    for key, val in h_trans.items():
                        h_trans[key] = val*np.array([1, -1])

        if i == 0:
            for key, val in h_trans.items():
                h_trans_all[key] = val
        else:
            for key, val in h_trans.items():
                h_trans_all[key] = np.concatenate((h_trans_all[key], val), axis=0)
        i += 1

    fig_name = 'taskset{:d}_{:s}space'.format(setup, representation)

    lxy = _plot_taskspace(h_trans_all, fig_name, setup=setup)
    fig_name = fig_name + '_example'
    lxy = _plot_taskspace(h_trans_all, fig_name, setup=setup,
                          plot_example=True, lxy=lxy)


def run_network_replacerule(model_dir, rule, replace_rule, rule_strength):
    """Run the network but with replaced rule input weights.

    Args:
        model_dir: model directory
        rule: the rule to test on
        replace_rule: a list of rule input units to use
        rule_strength: the relative strength of each replace rule unit
    """
    model = Model(model_dir)
    hparams = model.hparams
    with tf.Session() as sess:
        model.restore()

        # Get performance
        batch_size_test = 1000
        n_rep = 20
        batch_size_test_rep = int(batch_size_test/n_rep)
        perf_rep = list()
        for i_rep in range(n_rep):
            trial = generate_trials(rule, hparams, 'random', batch_size=batch_size_test_rep,
                                     replace_rule=replace_rule, rule_strength=rule_strength)
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            y_hat_test = sess.run(model.y_hat, feed_dict=feed_dict)

            perf_rep.append(np.mean(get_perf(y_hat_test, trial.y_loc)))

    return np.mean(perf_rep), rule_strength


def replace_rule_name(replace_rule, rule_strength):
    """Helper function to replace rule name"""
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


def compute_replacerule_performance(model_dir, setup, restore=False):
    """Compute the performance of one task given a replaced rule input."""

    if setup == 1:
        rule = 'delayanti'
        replace_rule = np.array(['delayanti', 'fdanti', 'delaygo', 'fdgo'])

        rule_strengths = \
            [[1,0,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,1,1,-1]]

    elif setup == 2:
        rule = 'contextdelaydm1'
        replace_rule = np.array(['contextdelaydm1', 'contextdelaydm2',
                                 'contextdm1', 'contextdm2'])

        rule_strengths = \
            [[1,0,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,0,1,0],
             [0,1,1,-1]]

    elif setup == 3:
        rule = 'dmsgo'
        replace_rule = np.array(['dmsgo', 'dmcgo', 'dmsnogo', 'dmcnogo'])
        rule_strengths = \
            [[1,0,0,0],
             [0,1,0,0],
             [0,1,1,0],
             [0,1,1,-1]]

    else:
        raise ValueError('Unknown setup value')

    fname = 'taskset{:d}_perf'.format(setup)+'.pkl'
    fname = os.path.join(model_dir, fname)

    if restore and os.path.isfile(fname):
        print('Reloading results from '+fname)
        with open(fname, 'rb') as f:
            r = pickle.load(f)
        perfs, rule, names = r['perfs'], r['rule'], r['names']

    else:
        perfs = list()
        names = list()
        for rule_strength in rule_strengths:
            perf, _ = run_network_replacerule(model_dir, rule, replace_rule, rule_strength)
            perfs.append(perf)
            names.append(replace_rule_name(replace_rule, rule_strength))

        perfs = np.array(perfs)
        print(perfs)

        results = {'perfs':perfs, 'rule':rule, 'names':names}
        with open(fname, 'wb') as f:
            pickle.dump(results, f)
        print('Results stored at : '+fname)

    return perfs, rule, names


def _plot_replacerule_performance(perfs, rule, names, setup, perfs_all=None, fig_name=None):
    save = True

    fs = 7
    width = 0.3
    fig = plt.figure(figsize=(1.6,2.2))
    ax = fig.add_axes([0.55,0.05,0.35,0.7])
    b0 = ax.barh(np.arange(len(perfs)), perfs[::-1],
           height=width, color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
    if perfs_all is not None:
        n_net = perfs_all.shape[0]
        for i in range(len(perfs)):
            ax.plot(perfs_all[:,-i-1], [i]*n_net, 'o',
                    color='black', alpha=0.15, markersize=2)
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
    if fig_name is None:
        fig_name = 'taskset{:d}_perf'.format(setup)
    if save:
        plt.savefig(os.path.join('figure', fig_name+'.pdf'), transparent=True)
    plt.show()


def plot_replacerule_performance(model_dir, setup, perfs_all=None, fig_name=None, restore=True):
    perfs, rule, names = compute_replacerule_performance(model_dir, setup, restore)
    _plot_replacerule_performance(perfs, rule, names, setup, perfs_all, fig_name)


def plot_replacerule_performance_group(model_dir, setup=1, restore=True):
    model_dirs = tools.valid_model_dirs(model_dir)
    print('Analyzing models : ')
    print(model_dirs)

    perfs_plot = list()
    for model_dir in model_dirs:
        perfs, rule, names = compute_replacerule_performance(model_dir, setup, restore)
        perfs_plot.append(perfs)

    perfs_plot = np.array(perfs_plot)
    perfs_median = np.median(perfs_plot, axis=0)

    fig_name = 'taskset{:d}_perf'.format(setup)

    print(perfs_median)
    _plot_replacerule_performance(perfs_median, rule, names, setup, perfs_all=perfs_plot, fig_name=fig_name)


if __name__ == '__main__':
    pass