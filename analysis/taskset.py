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

import tensorflow as tf

from task import rule_name
from task import generate_trials
from network import Model
from network import get_perf
import tools


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
        hp = model.hp

        if rules is None:
            # Default value
            rules = hp['rules']
        n_rules = len(rules)

        with tf.Session() as sess:
            model.restore()

            for rule in rules:
                trial = generate_trials(rule=rule, hp=hp, mode='test')
                feed_dict = tools.gen_feed_dict(model, trial, hp)
                h = sess.run(model.h, feed_dict=feed_dict)

                # Average across stimulus conditions
                h_stimavg = h.mean(axis=1)

                # dt_new = 50
                # every_t = int(dt_new/hp['dt'])

                t_start = int(500/hp['dt']) # Important: Ignore the initial transition
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

    def compute_and_plot_taskspace(self, rules=None, epochs=None, **kwargs):
        h_trans = self.compute_taskspace(rules=rules, epochs=epochs, **kwargs)
        self.plot_taskspace(h_trans, **kwargs)

    def compute_taskspace(self, rules=None, epochs=None, dim_reduction_type='MDS', **kwargs):
        # Only get last time points for each epoch
        h = self.filter(self.h_stimavg_byepoch, epochs=epochs, rules=rules, **kwargs)

        # Concatenate across rules to create dataset
        data = np.concatenate(list(h.values()), axis=0)
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

    def plot_taskspace(self, h_trans, epochs=None, dim_reduction_type='MDS',
                       plot_text=True, figsize=(4,4), markersize=5, plot_label=True):
        # Plot tasks in space
        shape_mapping = {'stim1' : 'o',
                         'stim2' : 'o',
                         'delay1' : 'v',
                         'delay2' : 'd',
                         'go1'  : 's',
                         'fix1' : 'p'}

        from analysis.performance import rule_color

        fs = 6 # fontsize
        dim0, dim1 = (0, 1) # plot dimensions

        texts = list()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

        for key, val in h_trans.items():
            rule, epoch = key

            # Default coloring by rule_color
            color = rule_color[rule]

            ax.plot(val[-1, dim0], val[-1, dim1], shape_mapping[epoch],
                    color=color, mec=color, mew=1.0, ms=markersize)

            if plot_text:
                texts.append(ax.text(val[-1, dim0]+0.03, val[-1, dim1]+0.03, rule_name[rule],
                                     fontsize=6, color=color))

            if 'fix' not in epoch:
                ax.plot(val[:, dim0], val[:, dim1], color=color, alpha=0.5)

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

        save_name = 'taskspace'+dim_reduction_type

        if epochs is not None:
            save_name = save_name + ''.join(epochs)

        plt.savefig(os.path.join('figure', save_name+'.pdf'), transparent=True)
        plt.show()


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
            h_trans = tools.load_pickle(fname)
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
        hp = model.hp
        n_hidden = hp['n_rnn']
        n_output = hp['n_output']
        with tf.Session() as sess:
            model.restore()
            w_in = sess.run(model.w_in).T

        rule_indices = [get_rule_index(r, hp) for r in rules]
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
    from analysis.performance import rule_color
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
        color = rule_color[rule]

        if plot_example:
            xplot, yplot = val[i_example,dim0], val[i_example,dim1]
        else:
            xplot, yplot = val[:,dim0], val[:,dim1]

        ax.plot(xplot, yplot, 'o', color=color, mec=color, mew=1.0, ms=2)


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
    elif kwargs['setup'] == 3:
        arrow_starts = [h_trans[('dmsgo','stim1')],
                        h_trans[('dmsnogo','stim1')]]
        arrow_ends   = [h_trans[('dmcgo','stim1')],
                        h_trans[('dmcnogo','stim1')]]
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
    pc_name = 'rPC'
    ax.set_xlabel(pc_name+' {:d}'.format(dim0+1), fontsize=fs, labelpad=-5)
    ax.set_ylabel(pc_name+' {:d}'.format(dim1+1), fontsize=fs, labelpad=-5)

    plt.savefig(os.path.join('figure', fig_name+'.pdf'), transparent=True)
    plt.show()

    return (lx, ly)


def plot_taskspace(model_dir, setup=1, restore=True, representation='rate'):
    h_trans = compute_taskspace(
        model_dir, setup, restore=restore, representation=representation)
    save_name = 'taskset{:d}_space'.format(setup)
    _plot_taskspace(h_trans, save_name, setup=setup)


def plot_taskspace_group(root_dir, setup=1, restore=True,
                         representation='rate', fig_name_addon=None):
    """Plot task space for a group of networks.

    Args:
        root_dir : the root directory for all models to analyse
        setup: int, the combination of rules to use
        restore: bool, whether to restore results
        representation: 'rate' or 'weight'
    """

    model_dirs = tools.valid_model_dirs(root_dir)
    print('Analyzing models : ')
    print(model_dirs)

    h_trans_all = OrderedDict()
    i = 0
    for model_dir in model_dirs:
        try:
            h_trans = compute_taskspace(model_dir, setup,
                                        restore=restore,
                                        representation=representation)
        except ValueError:
            print('Skipping model at ' + model_dir)
            continue

        h_trans_values = list(h_trans.values())

        # When PC1 and PC2 capture similar variances, allow for a rotation
        # rotation_matrix, clock wise
        get_angle = lambda vec : np.arctan2(vec[1], vec[0])
        theta = get_angle(h_trans_values[0][0])
        # theta = 0
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])

        for key, val in h_trans.items():
            h_trans[key] = np.dot(val, rot_mat)

        h_trans_values = list(h_trans.values())
        if h_trans_values[1][0][1] < 0:
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
    if fig_name_addon is not None:
        fig_name = fig_name + fig_name_addon

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
    hp = model.hp
    with tf.Session() as sess:
        model.restore()

        # Get performance
        batch_size_test = 1000
        n_rep = 20
        batch_size_test_rep = int(batch_size_test/n_rep)
        perf_rep = list()
        for i_rep in range(n_rep):
            trial = generate_trials(rule, hp, 'random', batch_size=batch_size_test_rep,
                                     replace_rule=replace_rule, rule_strength=rule_strength)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
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
        r = tools.load_pickle(fname)
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


def _plot_replacerule_performance(perfs_all, rule, names, setup, fig_name=None):
    perfs_all = perfs_all.T  # make it (4, n_nets)
    from scipy.stats import mannwhitneyu
    print(mannwhitneyu(perfs_all[-1], perfs_all[-2]))
    print(mannwhitneyu(perfs_all[-1], perfs_all[-3]))
    
    n_condition, n_net = perfs_all.shape
    fs = 7
    fig = plt.figure(figsize=(1.6,2.2))
    ax = fig.add_axes([0.55,0.05,0.35,0.7])

    bp = ax.boxplot(list(perfs_all[::-1]), notch=True, vert=False, bootstrap=10000,
               showcaps=False, patch_artist=True, widths=0.4,
               flierprops={'markersize': 2}, whiskerprops={'linewidth': 1.5})
    for element in ['boxes', 'whiskers', 'fliers']:
        plt.setp(bp[element], color='xkcd:cerulean')
    for patch in bp['boxes']:
        patch.set_facecolor('xkcd:cerulean')
    for element in ['means', 'medians']:
        plt.setp(bp[element], color='white')

    ax.set_yticks(np.arange(1, 1+n_condition))
    ax.set_yticklabels(names[::-1], rotation=0, horizontalalignment='right')
    ax.set_ylabel('Rule input', fontsize=fs, labelpad=3)
    # ax.set_ylabel('performance', fontsize=fs)
    title = 'Performance on\n'+rule_name[rule]
    if perfs_all is not None:
        n_net = perfs_all.shape[1]
        title = title + ' (n={:d})'.format(n_net)
    ax.set_title(title, fontsize=fs, y=1.13)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.grid(True)
    ax.set_xticks([0,0.5,1.0])
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0.5, n_condition+0.5])
    if fig_name is None:
        fig_name = 'taskset{:d}_perf'.format(setup)
    plt.savefig(os.path.join('figure', fig_name+'.pdf'), transparent=True)
    plt.show()


def plot_replacerule_performance(
        model_dir,setup, perfs_all=None, fig_name=None, restore=True):
    perfs, rule, names = compute_replacerule_performance(
            model_dir, setup, restore)
    _plot_replacerule_performance(
        perfs_all, rule, names, setup, fig_name)


def plot_replacerule_performance_group(model_dir, setup=1, restore=True, fig_name_addon=None):
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
    if fig_name_addon is not None:
        fig_name = fig_name + fig_name_addon

    print(perfs_median)
    _plot_replacerule_performance(perfs_plot, rule, names, setup, fig_name=fig_name)


if __name__ == '__main__':
    root_dir = './data/train_all'
    model_dir = root_dir + '/0'
    setups = [3]
    for setup in setups:
        pass
        plot_taskspace_group(root_dir, setup=setup,
                                     restore=True, representation='rate')
        plot_taskspace_group(root_dir, setup=setup,
                                     restore=True, representation='weight')
        plot_replacerule_performance_group(
                root_dir, setup=setup, restore=True)