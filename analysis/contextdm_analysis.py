"""
Analysis of the choice att tasks
"""

from __future__ import division

import os
import pickle
import copy
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from task import generate_trials, rule_name, get_rule_index, get_dist
from network import Model
import tools
from analysis import performance

COLORS = ['xkcd:'+c for c in ['orange', 'green', 'pink', 'sky blue']]


class UnitAnalysis(object):
    def __init__(self, model_dir):
        """Analyze based on units."""
        data_type  = 'rule'
        fname = os.path.join(model_dir, 'variance_' + data_type + '.pkl')
        res = tools.load_pickle(fname)
        h_var_all = res['h_var_all']
        keys      = res['keys']

        rules = ['contextdm1', 'contextdm2']
        ind_rules = [keys.index(rule) for rule in rules]
        h_var_all = h_var_all[:, ind_rules]

        # First only get active units. Total variance across tasks larger than 1e-3
        ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
        # ind_active = np.where(h_var_all.sum(axis=1) > 1e-1)[0] # TEMPORARY
        h_var_all  = h_var_all[ind_active, :]

        # Normalize by the total variance across tasks
        h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

        group_ind = dict()

        group_ind['1'] = np.where(h_normvar_all[:,0] > 0.9)[0]
        group_ind['2'] = np.where(h_normvar_all[:, 0] < 0.1)[0]
        group_ind['12'] = np.where(np.logical_and(h_normvar_all[:,0] > 0.4,
                                                  h_normvar_all[:,0] < 0.6))[0]
        group_ind['1+2'] = np.concatenate((group_ind['1'], group_ind['2']))

        group_ind_orig = {key: ind_active[val] for key, val in group_ind.items()}

        self.model_dir = model_dir
        self.group_ind = group_ind
        self.group_ind_orig = group_ind_orig
        self.h_normvar_all = h_normvar_all
        self.rules = rules
        self.ind_active = ind_active
        colors = ['xkcd:'+c for c in ['orange', 'green', 'pink', 'sky blue']]
        self.colors = dict(zip([None, '1', '2', '12'], colors))
        self.lesion_group_names = {None : 'Intact',
                                   '1'  : 'Lesion group 1',
                                   '2'  : 'Lesion group 2',
                                   '12' : 'Lesion group 12',
                                   '1+2': 'Lesion group 1 & 2'}

    def prettyplot_hist_varprop(self):
        """Pretty version of variance.plot_hist_varprop."""
        rules = self.rules

        fs = 6
        fig = plt.figure(figsize=(2.0,1.2))
        ax = fig.add_axes([0.2,0.3,0.5,0.5])
        data_plot = self.h_normvar_all[:, 0]
        hist, bins_edge = np.histogram(data_plot, bins=30, range=(0,1))
        ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
               color='gray', edgecolor='none')
        bs = list()
        for i, group in enumerate(['1', '2', '12']):
            data_plot = self.h_normvar_all[self.group_ind[group], 0]
            hist, bins_edge = np.histogram(data_plot, bins=20, range=(0,1))
            b_tmp = ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
                   color=self.colors[group], edgecolor='none', label=group)
            bs.append(b_tmp)
        plt.locator_params(nbins=3)
        xlabel = 'FracVar({:s}, {:s})'.format(rule_name[rules[0]], rule_name[rules[1]])
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel('Units', fontsize=fs)
        ax.set_ylim(bottom=-0.02*hist.max())
        ax.set_xlim((-0.1, 1.1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
        lg = ax.legend(bs, ['1','2','12'], title='Group',
                       fontsize=fs, ncol=1, bbox_to_anchor=(1.5,1.0),
                       loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        plt.savefig('figure/prettyplot_hist_varprop'+
                rule_name[rules[0]].replace(' ','')+
                rule_name[rules[1]].replace(' ','')+
                '.pdf', transparent=True)

    def plot_inout_connections(self):
        self._plot_inout_connections('input')
        self._plot_inout_connections('output')

    def _plot_inout_connections(self, conn_type):
        """Plot connectivity while sorting by group.

        Args:
            conn_type: str, type of connectivity to plot.
        """

        # Sort data by labels and by input connectivity
        model = Model(self.model_dir)
        hp = model.hp
        with tf.Session() as sess:
            model.restore()
            w_in, w_out = sess.run([model.w_in, model.w_out])

        n_ring = hp['n_eachring']
        groups = ['1', '2', '12']

        # Plot input from stim or output to loc
        if conn_type == 'input':
            w_conn = w_in[1:n_ring+1, :].T
            xlabel = 'Preferred mod 1 input dir.'
            ylabel = 'Conn. weight\n from mod 1'
            lgtitle = 'To group'
        elif conn_type == 'output':
            w_conn = w_out[:, 1:]
            xlabel = 'Preferred output dir.'
            ylabel = 'Conn. weight to output'
            lgtitle = 'From group'
        else:
            raise ValueError('Unknown conn type')

        w_aves = dict()

        for group in groups:
            ind_group  = self.group_ind_orig[group]
            n_group    = len(ind_group)
            w_group = np.zeros((n_group, n_ring))

            for i, ind in enumerate(ind_group):
                tmp = w_conn[ind, :]
                ind_max = np.argmax(tmp)
                w_group[i, :] = np.roll(tmp, int(n_ring/2)-ind_max)

            w_aves[group] = w_group.mean(axis=0)

        fs = 6
        fig = plt.figure(figsize=(1.5, 1.0))
        ax = fig.add_axes([.35, .25, .55, .6])
        for group in groups:
            ax.plot(w_aves[group], color=self.colors[group], label=group, lw=1)
        ax.set_xticks([int(n_ring/2)])
        ax.set_xticklabels([xlabel])
        # ax.set_xlabel(xlabel, fontsize=fs, labelpad=3)
        ax.set_ylabel(ylabel, fontsize=fs)
        lg = ax.legend(title=lgtitle, fontsize=fs, bbox_to_anchor=(1.2,1.2),
                       labelspacing=0.2, loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(axis='y',nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/conn_'+conn_type+'_contextdm.pdf', transparent=True)

    def plot_rule_connections(self):
        """Plot connectivity while sorting by group.

        Args:
            conn_type: str, type of connectivity to plot.
        """

        # Sort data by labels and by input connectivity
        model = Model(self.model_dir)
        hp = model.hp
        with tf.Session() as sess:
            model.restore()
            w_in = sess.run(model.w_in)
        w_in = w_in.T

        groups = ['1', '2', '12']

        # Plot input rule connectivity
        rules = ['contextdm1', 'contextdm2', 'dm1', 'dm2', 'multidm']

        w_stores = OrderedDict()
        w_all_stores = OrderedDict()
        pos = list()
        width = 0.15
        colors = list()
        for i_group, group in enumerate(groups):
            w_store_tmp = list()
            ind = self.group_ind_orig[group]
            for i_rule, rule in enumerate(rules):
                ind_rule = get_rule_index(rule, hp)
                w_conn = w_in[ind, ind_rule].mean(axis=0)
                w_store_tmp.append(w_conn)
                w_all_stores[(group, rule)] = w_in[ind, ind_rule].flatten()
                pos.append(i_rule+(i_group-1.5)*width)
                colors.append(self.colors[group])
            w_stores[group] = w_store_tmp


        fs = 6
        fig = plt.figure(figsize=(2.5, 1.2))
        ax = fig.add_axes([0.17,0.45,0.8,0.4])
        # for i, group in enumerate(groups):
        #     x = np.arange(len(rules))+(i-1.5)*width
        #     b0 = ax.bar(x, w_stores[group],
        #                 width=width, color=self.colors[group], edgecolor='none')

        bp = ax.boxplot([w for w in w_all_stores.values()], notch=True, sym='',
                        bootstrap=10000,
                        showcaps=False, patch_artist=True, widths=width, positions=pos,
                        whiskerprops={'linewidth': 1.0})
        # for element in ['boxes', 'whiskers', 'fliers']:
        #     plt.setp(bp[element], color='xkcd:cerulean')

        for patch, c in zip(bp['boxes'], colors):
            plt.setp(patch, color=c)
        for i_whisker, patch in enumerate(bp['whiskers']):
            plt.setp(patch, color=colors[int(i_whisker/2)])
        for element in ['means', 'medians']:
            plt.setp(bp[element], color='white')

        ax.set_xticks(np.arange(len(rules)))
        ax.set_xticklabels([rule_name[r] for r in rules], rotation=25)
        ax.set_xlabel('Input from rule units', fontsize=fs, labelpad=3)
        ax.set_ylabel('Conn. weight', fontsize=fs)
        # lg = ax.legend(groups, fontsize=fs, ncol=3, bbox_to_anchor=(1,1.5),
        #                labelspacing=0.2, loc=1, frameon=False, title='To group')
        # plt.setp(lg.get_title(),fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(axis='y',nbins=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlim([-0.8, len(rules)-0.2])
        ax.plot([-0.5, len(rules)-0.5], [0, 0], color='gray', linewidth=0.5)
        plt.savefig('figure/conn_rule_contextdm.pdf', transparent=True)
        plt.show()

    def plot_rec_connections(self):
        """Plot connectivity while sorting by group.

        Args:
            conn_type: str, type of connectivity to plot.
        """

        # Sort data by labels and by input connectivity
        model = Model(self.model_dir)
        hp = model.hp
        with tf.Session() as sess:
            model.restore()
            w_in, w_rec = sess.run([model.w_in, model.w_rec])
        w_in, w_rec = w_in.T, w_rec.T

        n_ring = hp['n_eachring']
        groups = ['1', '2', '12']

        w_in_ = (w_in[:, 1:n_ring + 1] + w_in[:, 1+n_ring:2*n_ring+1]) / 2.

        # Plot recurrent connectivity
        # w_rec_group = np.zeros((len(groups), len(groups)))
        # for i1, group1 in enumerate(groups):
        #     for i2, group2 in enumerate(groups):
        #         ind1 = self.group_ind_orig[group1]
        #         ind2 = self.group_ind_orig[group2]
        #         w_rec_group[i2, i1] = w_rec[:, ind1][ind2, :].mean()

        i_pairs = list()
        for i1 in range(len(groups)):
            for i2 in range(len(groups)):
                i_pairs.append((i1, i2))

        pref_diffs_list = list()
        w_recs_list = list()

        w_rec_bygroup = np.zeros((len(groups), len(groups)))

        inds = [self.group_ind_orig[g] for g in groups]
        for i_pair in i_pairs:
            ind1, ind2 = inds[i_pair[0]], inds[i_pair[1]]
            # For each neuron get the preference based on input weight
            # sort by weights
            w_sortby = w_in_
            # w_sortby = w_out_
            prefs1 = np.argmax(w_sortby[ind1, :], axis=1)*2.*np.pi/n_ring
            prefs2 = np.argmax(w_sortby[ind2, :], axis=1)*2.*np.pi/n_ring

            # Compute the pairwise distance based on preference
            # Then get the connection weight between pairs
            pref_diffs = list()
            w_recs = list()
            for i, ind_i in enumerate(ind1):
                for j, ind_j in enumerate(ind2):
                    if ind_j == ind_i:
                        # Excluding self connections, which tend to be positive
                        continue
                    pref_diffs.append(get_dist(prefs1[i]-prefs2[j]))
                    # pref_diffs.append(prefs1[i]-prefs2[j])
                    w_recs.append(w_rec[ind_j, ind_i])
            pref_diffs, w_recs = np.array(pref_diffs), np.array(w_recs)
            pref_diffs_list.append(pref_diffs)
            w_recs_list.append(w_recs)

            w_rec_bygroup[i_pair[1], i_pair[0]] = np.mean(w_recs[pref_diffs<np.pi/6.])

        fs = 6
        vmax = np.ceil(np.max(w_rec_bygroup) * 100) / 100.
        vmin = np.floor(np.min(w_rec_bygroup) * 100) / 100.
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes([0.2, 0.1, 0.5, 0.5])
        im = ax.imshow(w_rec_bygroup, interpolation='nearest',
                       cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        # ax.axis('off')
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        plt.xticks([0, 1, 2], groups, fontsize=6)
        plt.yticks([0, 1, 2], groups, fontsize=6)
        ax.tick_params('both', length=0)
        ax.set_xlabel('From', fontsize=fs, labelpad=2)
        ax.set_ylabel('To', fontsize=fs, labelpad=2)
        for s in ['right', 'left', 'top', 'bottom']:
            ax.spines[s].set_visible(False)

        ax = fig.add_axes([0.72, 0.1, 0.03, 0.5])
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        cb.set_label(r'Rec. weight', fontsize=fs, labelpad=-7)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(nbins=3)

        plt.savefig('figure/conn_rec_contextdm.pdf', transparent=True)


def _gen_taskparams(stim1_loc, n_rep=1):
    """Generate parameters for task."""
    # Do not loop over stim1_loc
    # Generate task parameterse for state-space analysis
    n_stim = 6
    batch_size = n_rep * n_stim**2
    batch_shape = (n_stim, n_stim, n_rep)
    ind_stim_mod1, ind_stim_mod2, ind_rep = np.unravel_index(range(batch_size), batch_shape)

    stim1_locs = np.ones(batch_size)*stim1_loc

    stim2_locs = (stim1_locs+np.pi) % (2*np.pi)

    stim_cohs = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])*0.5
    stim_mod1_cohs = np.array([stim_cohs[i] for i in ind_stim_mod1])
    stim_mod2_cohs = np.array([stim_cohs[i] for i in ind_stim_mod2])

    params = {'stim1_locs' : stim1_locs,
              'stim2_locs' : stim2_locs,
              'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
              'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
              'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
              'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
              'stim_time'    : 800}

    return params, batch_size


def _sort_ind_byvariance(**kwargs):
    """Sort ind into group 1, 2, 12, and others according task variance

    Returns:
        ind_group: indices for the current matrix, not original
        ind_active_group: indices for original matrix
    """
    ua = UnitAnalysis(kwargs['model_dir'])
    
    ind_active = ua.ind_active  # TODO(gryang): check if correct

    ind_group = dict()
    ind_active_group = dict()

    for group in ['1', '2', '12', None]:
        if group is not None:
            # Find all units here that belong to group 1, 2, or 12 as defined in UnitAnalysis
            ind_group[group] = [k for k, ind in enumerate(ind_active)
                                if ind in ua.group_ind_orig[group]]
        else:
            ind_othergroup = np.concatenate(list(ua.group_ind_orig.values()))
            ind_group[group] = [k for k, ind in enumerate(ind_active)
                                if ind not in ind_othergroup]

        # Transform to original matrix indices
        ind_active_group[group] = [ind_active[k] for k in ind_group[group]]

    return ind_group, ind_active_group


def sort_ind_bygroup(grouping, **kwargs):
    """Sort indices by group."""
    if grouping == 'var':
        return _sort_ind_byvariance(**kwargs)
    else:
        raise ValueError()


def _plot_performance_choicetasks(model_dir, lesion_units_list, rules_perf=None,
                                  legends=None, save_name=None):
    """Plot performance across tasks for different lesioning.

    Args:
        model_dir: str, model directory
        lesion_units_list: list of lists of units to lesion
        rules_perf: list of rules to perform
        legends: list of str
        save_name: None or str, add to the figure name
    """
    if rules_perf is None:
        rules_perf = ['contextdm1', 'contextdm2', 'dm1', 'dm2', 'multidm']

    perf_stores = list()
    for lesion_units in lesion_units_list:
        perf_stores_tmp = list()
        for rule in rules_perf:
            perf, prop1s, cohs = performance.psychometric_choicefamily_2D(
                model_dir, rule, lesion_units=lesion_units,
                n_coh=4, n_stim_loc=10)
            perf_stores_tmp.append(perf)

        perf_stores_tmp = np.array(perf_stores_tmp)
        perf_stores.append(perf_stores_tmp)

    fs = 6
    width = 0.15
    fig = plt.figure(figsize=(2.5, 1.2))
    ax = fig.add_axes([0.17, 0.35, 0.8, 0.4])
    for i, perf in enumerate(perf_stores):
        b0 = ax.bar(np.arange(len(rules_perf))+(i-2)*width, perf,
                    width=width, color=COLORS[i], edgecolor='none')
    ax.set_xticks(np.arange(len(rules_perf)))
    ax.set_xticklabels([rule_name[r] for r in rules_perf], rotation=25)
    ax.set_xlabel('Tasks', fontsize=fs, labelpad=-2)
    ax.set_ylabel('Performance', fontsize=fs)
    lg = ax.legend(legends,
                   fontsize=fs, ncol=2, bbox_to_anchor=(1,1.6),
                   labelspacing=0.2, loc=1, frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.locator_params(axis='y',nbins=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim([-0.8, len(rules_perf)-0.2])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])
    fig_name = 'figure/perf_contextdm_lesion'
    if save_name is not None:
        fig_name += save_name
    plt.savefig(fig_name + '.pdf', transparent=True)
    plt.show()


def plot_performance_choicetasks(model_dir, grouping):
    """Plot performance across tasks for different lesioning.

    Args:
        model_dir: str, model directory
        grouping: str, how to group different populations
    """
    ind_group, ind_group_orig = sort_ind_bygroup(grouping=grouping,
                                                 model_dir=model_dir)
    groups = ['1', '2', '12']
    lesion_units_list = [None] + [ind_group_orig[g] for g in groups]
    legends = ['Intact']+['Lesion group {:s}'.format(l) for l in groups]
    save_name = '_bygrouping_' + grouping
    print(lesion_units_list)
    _plot_performance_choicetasks(model_dir, lesion_units_list,
                                  legends=legends, save_name=save_name)


def plot_performance_2D(model_dir, rule, lesion_units=None,
                        title=None, save_name=None, **kwargs):
    """Plot performance as function of both modality coherence."""
    perf, prop1s, cohs = performance.psychometric_choicefamily_2D(
        model_dir, rule, lesion_units=lesion_units,
        n_coh=8, n_stim_loc=20)

    performance._plot_psychometric_choicefamily_2D(
        prop1s, cohs, rule, title=title, save_name=save_name, **kwargs)


def plot_performance_2D_all(model_dir, rule):
    """Plot performance of both modality coherence for all groups."""
    ua = UnitAnalysis(model_dir)
    title = 'Intact'
    save_name = rule_name[rule].replace(' ', '') + '_perf2D_intact.pdf'
    plot_performance_2D(model_dir, rule, title=title, save_name=save_name)
    for lesion_group in ['1', '2', '12', '1+2']:
        lesion_units = ua.group_ind_orig[lesion_group]
        title = ua.lesion_group_names[lesion_group]
        save_name = rule_name[rule].replace(' ', '') + \
                    '_perf2D_lesion' + str(lesion_group) + '.pdf'
        plot_performance_2D(
            model_dir, rule, lesion_units=lesion_units,
            title=title, save_name=save_name, ylabel=False, colorbar=False)


def quick_statespace(model_dir):
    """Quick state space analysis using simply PCA."""
    rules = ['contextdm1', 'contextdm2']
    h_lastts = dict()
    model = Model(model_dir)
    hp = model.hp
    with tf.Session() as sess:
        model.restore()
        for rule in rules:
            # Generate a batch of trial from the test mode
            trial = generate_trials(rule, hp, mode='test')
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)
            lastt = trial.epochs['stim1'][-1]
            h_lastts[rule] = h[lastt,:,:]

    from sklearn.decomposition import PCA
    model = PCA(n_components=5)
    model.fit(np.concatenate(h_lastts.values(), axis=0))
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([.3, .3, .6, .6])
    for rule, color in zip(rules, ['red', 'blue']):
        data_trans = model.transform(h_lastts[rule])
        ax.scatter(data_trans[:, 0], data_trans[:, 1], s=1,
                   label=rule_name[rule], color=color)
    plt.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlabel('PC 1', fontsize=7)
    ax.set_ylabel('PC 2', fontsize=7)
    lg = ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1,0.3),
                   loc=1, frameon=False)
    if save:
        plt.savefig('figure/choiceatt_quickstatespace.pdf',transparent=True)


def load_data(model_dir=None, sigma_rec=0, lesion_units=None, n_rep=1):
    """Generate model data into standard format.

    Returns:
        data: standard format, list of dict of arrays/dict
            list is over neurons
            dict is for response array and task variable dict
            response array has shape (n_trial, n_time)
    """
    if model_dir is None:
        model_dir = './mantetemp'  # TEMPORARY SETTING

    # Get rules and regressors
    rules = ['contextdm1', 'contextdm2']

    n_rule = len(rules)

    data = list()

    model = Model(model_dir, sigma_rec=sigma_rec)
    hp = model.hp
    with tf.Session() as sess:
        model.restore()
        if lesion_units is not None:
            model.lesion_units(sess, lesion_units)

        # Generate task parameters used
        # Target location
        stim1_loc_list = np.arange(0, 2*np.pi, 2*np.pi/12)
        for stim1_loc in stim1_loc_list:
            params, batch_size = _gen_taskparams(stim1_loc=stim1_loc, n_rep=n_rep)
            stim1_locs_tmp = np.tile(params['stim1_locs'], n_rule)

            x = list() # Network input
            y_loc = list() # Network target output location

            # Start computing the neural activity
            for i, rule in enumerate(rules):
                # Generating task information
                trial = generate_trials(rule, hp, 'psychometric',
                                        params=params, noise_on=True)
                x.append(trial.x)
                y_loc.append(trial.y_loc)

            x = np.concatenate(x, axis=1)
            y_loc = np.concatenate(y_loc, axis=1)

            # Coherences
            stim_mod1_cohs = params['stim1_mod1_strengths'] - params[
                'stim2_mod1_strengths']
            stim_mod2_cohs = params['stim1_mod2_strengths'] - params[
                'stim2_mod2_strengths']
            stim_mod1_cohs /= stim_mod1_cohs.max()
            stim_mod2_cohs /= stim_mod2_cohs.max()

            # Get neural activity
            fetches = [model.h, model.y_hat, model.y_hat_loc]
            H, y_sample, y_sample_loc = sess.run(
                fetches, feed_dict={model.x: x})

            # Downsample in time
            dt_new = 50
            every_t = int(dt_new / hp['dt'])
            # Only analyze the target epoch
            epoch = trial.epochs['stim1']
            H = H[epoch[0]:epoch[1], ...][int(every_t / 2)::every_t, ...]

            # Get performance and choices
            # perfs = get_perf(y_sample, y_loc)
            # y_choice is 1 for choosing stim1_loc, otherwise -1
            y_actual_choice = 2*(get_dist(y_sample_loc[-1]-stim1_loc)<np.pi/2)-1
            y_target_choice = 2*(get_dist(y_loc[-1]-stim1_loc)<np.pi/2)-1

            # Get task variables
            task_var = dict()
            task_var['targ_dir'] = y_actual_choice
            task_var['stim_dir'] = np.tile(stim_mod1_cohs, n_rule)
            task_var['stim_col2dir'] = np.tile(stim_mod2_cohs, n_rule)
            task_var['context'] = np.repeat([1, -1], batch_size)
            task_var['correct'] = (y_actual_choice == y_target_choice).astype(int)
            task_var['stim_dir_sign'] = (task_var['stim_dir']>0).astype(int)*2-1
            task_var['stim_col2dir_sign'] = (task_var['stim_col2dir']>0).astype(int)*2-1


            n_unit = H.shape[-1]
            for i_unit in range(n_unit):
                unit_dict = {
                    'rate': H[:, :, i_unit].T,  # standard format (n_trial, n_time)
                    'task_var': copy.deepcopy(task_var)
                }
                data.append(unit_dict)
    return data


if __name__ == '__main__':
    root_dir = './data/vary_pweighttrain_mante'
    hp_target = {'activation': 'softplus',
                 'rnn_type': 'LeakyRNN',
                 'w_rec_init': 'randortho',
                 'p_weight_train': 0.1}
    
    model_dir = tools.find_model(root_dir, hp_target, perf_min=0.8)

    root_dir = './data/train_all'
    model_dir = root_dir + '/0'

    ######################### Connectivity and Lesioning ######################
    # ua = UnitAnalysis(model_dir)
    # ua.plot_inout_connections()
    # ua.plot_rec_connections()
    # ua.plot_rule_connections()
    # ua.prettyplot_hist_varprop()

    # plot_performance_choicetasks(model_dir, grouping='var')
    # plot_performance_choicetasks(model_dir, grouping='beta')

    plot_performance_2D_all(model_dir, 'contextdm1')

    # plot_fullconnectivity(model_dir)
    # plot_groupsize('allrule_weaknoise')  # disabled now
    
    # plot_performance_lesionbyactivity(root_dir, 'tanh', n_lesion=50)
    # plot_performance_lesionbyactivity(root_dir, 'softplus', n_lesion=50)