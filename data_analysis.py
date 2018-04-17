"""Data analysis."""

from __future__ import division

import os
import sys
import numpy as np
import pickle
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.io import loadmat, whosmat

import mante_dataset_preprocess
import siegel_dataset_preprocess
import contextdm_analysis
import tools

DATAPATH = './datasets/mante_dataset'
# sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])
COLORS = [(0.9764705882352941, 0.45098039215686275, 0.023529411764705882),
 (0.08235294117647059, 0.6901960784313725, 0.10196078431372549),
 (1.0, 0.5058823529411764, 0.7529411764705882),
 (0.4588235294117647, 0.7333333333333333, 0.9921568627450981)]


def load_data(dataset='mante', smooth=True, analyze_single_units=True,
              model_dir=None):
    """Analyzing a dataset.

    Args:
        dataset: str, 'mante', 'siegel', or 'model'
        smooth: bool, whether to use smoothed data
        analyze_single_units: bool, whether to analyze single units
    """
    if smooth:
        print('Loading smooth data')
    else:
        print('Loading original data')
    if dataset == 'mante':
        return mante_dataset_preprocess.load_data(
            smooth=smooth, single_units=analyze_single_units)
    elif dataset == 'siegel':
        return siegel_dataset_preprocess.load_data(single_file=False)
    elif dataset == 'model':
        if model_dir is None:
            raise ValueError(
                'model_dir need to be provided for dataset==model')
        return contextdm_analysis.load_data(model_dir)
    else:
        raise ValueError('Wrong dataset')

    # TODO(gryang): make the following compatible
    # self.var_key_list = ['targ_dir', 'stim_dir_sign', 'stim_col2dir_sign']
    # self.data_condavg_list = [] # for two contexts
    # for context in [1, -1]:
    #     tmp = dict()
    #     for var_key in self.var_key_list:
    #         tmp[var_key] = get_trial_avg(data=self.data, split_traintest=False,
    #                                    var_keys=[var_key], context=context)
    #     self.data_condavg_list.append(tmp)


def get_trial_avg(data,
                  var_keys=None,
                  context=None,
                  split_traintest=False,
                  random_shuffle=False,
                  ):
    """Get trial-averaged rate for each condition.

    Args:
        data: Standard format data.
        var_keys: the keys for which condition would be looped over
        context:
        split_traintest: bool, whether to split training and testing
        random_shuffle: bool, if True, randomly shuffle the indices

    Returns:
        data_train: numpy array, (n_time, n_cond, n_unit)
        data_test: (optional) numpy array, (n_time, n_cond, n_unit)
    """

    if var_keys is None:
        # var_keys = ['stim_dir', 'stim_col2dir', 'context']
        # var_keys = ['targ_dir', 'stim_dir', 'stim_col2dir', 'context']
        var_keys = ['stim_dir_sign', 'stim_col2dir_sign', 'context']
        # var_keys = ['targ_dir', 'stim_dir_sign', 'stim_col2dir_sign', 'context']

    if context is not None:
        # If specifying context, then context should not be in var_keys
        assert 'context' not in var_keys

    # number of variables
    n_var = len(var_keys)

    # initialize
    task_var = data[0]['task_var']  # turn into dictionary
    n_time = data[0]['rate'].shape[1]
    n_unit = len(data)

    n_cond = np.prod([len(np.unique(task_var[k])) for k in var_keys])

    data_shape = (n_time, n_cond, n_unit)
    if split_traintest:
        p_train = 0.7 # proportion of training data
        data_train = np.zeros(data_shape)
        data_test  = np.zeros(data_shape)
    else:
        data_train = np.zeros(data_shape)

    for i_unit in range(n_unit):
        task_var = data[i_unit]['task_var']

        # number of trials
        n_trial = len(task_var[var_keys[0]])

        # dictionary of unique task variable values
        var_unique = [np.unique(task_var[k]) for k in var_keys]

        # list of number of unique task variable values
        n_var_unique = [len(v) for v in var_unique]

        # number of condition
        n_cond = np.prod(n_var_unique)

        # List of indices for each task variable
        ind_var_conds = np.unravel_index(range(n_cond), n_var_unique)

        # Responses of this unit
        response = data[i_unit]['rate']  # (trial, time)

        for i_cond in range(n_cond):
            if context is None:
                ind_cond = np.ones(n_trial, dtype=bool)
            else:
                # only look at this context
                ind_cond = task_var['context']==context

            for i_var in range(n_var):
                ind_var_cond = ind_var_conds[i_var][i_cond]
                ind_cond_tmp = (task_var[var_keys[i_var]] ==
                                var_unique[i_var][ind_var_cond])

                ind_cond *= ind_cond_tmp

            if random_shuffle:
                np.random.shuffle(ind_cond)

            if split_traintest:
                # Turn into actual indices
                ind_cond = np.where(ind_cond)[0]
                # randomly shuffle
                np.random.shuffle(ind_cond)
                # split into training and testing ones
                n_trial_cond = len(ind_cond)
                n_trial_cond_train = int(n_trial_cond*p_train)

                ind_cond_train = ind_cond[:n_trial_cond_train]
                ind_cond_test  = ind_cond[n_trial_cond_train:]

                # if n_trial_cond_train == 0:
                #     print(i_unit, i_cond, n_trial_cond)

                data_train[:, i_cond, i_unit] = np.mean(
                    response[ind_cond_train, :], axis=0)
                data_test[:, i_cond, i_unit] = np.mean(
                    response[ind_cond_test, :], axis=0)
            else:
                data_train[:, i_cond, i_unit] = np.mean(
                    response[ind_cond, :], axis=0)

    if split_traintest:
        return data_train, data_test
    else:
        return data_train


def get_trial_avg_var(data, var_method, rotation_matrix=None,
                      random_shuffle=False):
    """Compute the trial averaged variance.

    For efficiency, fuse the rotation here

    Args:
        data: standard format data
        var_method: str, method used to compute task variance
            Can be 'time_avg_late', 'time_avg_none', 'time_avg_early'
        rotation_matrix: None or np 2D array, the matrix to rotate the data
        random_shuffle: bool, whether to randomly shuffle the data
    """
    var_keys = ['stim_dir_sign', 'stim_col2dir_sign']
    # var_keys = ['stim_dir', 'stim_col2dir']

    vars = list()
    for context in [1, -1]:
        # r has shape (n_time, n_cond, n_unit)
        r = get_trial_avg(data, var_keys=var_keys, context=context,
                          random_shuffle=random_shuffle)
        # transpose to (n_cond, n_time, n_unit)
        r = np.swapaxes(r, 0, 1)

        if rotation_matrix is not None:
            r = np.dot(r, rotation_matrix)

        # input should be (n_cond, n_time, n_unit)
        v = compute_var(r, var_method)
        vars.append(v)

    return vars


def get_shuffle_var(data, var_method, rotation_matrix=None, n_rep=10):
    """Get task variance when data is shuffled."""
    start = time.time()

    var1s_shuffle, var2s_shuffle = 0, 0
    for i_rep in range(n_rep):
        v1, v2 = get_trial_avg_var(
            data, var_method, rotation_matrix, random_shuffle=True)
        var1s_shuffle += v1
        var2s_shuffle += v2

    var1s_shuffle /= n_rep
    var2s_shuffle /= n_rep

    print('Time taken {:0.2f}s'.format(time.time()-start))

    return var1s_shuffle, var2s_shuffle


def smoothing(response, response_time):
    '''
    Smoothing response
    :param response: numpy array (n_trial, n_time)
    :param response_time: numpy array (n_time,)
    :return:
    '''
    dt = response_time[1] - response_time[0]

    filter_type = 'gauss'
    filter_width = 0.04

    # Make filter
    if filter_type == 'gauss':

        # Filter parameters
        fw = filter_width

        # Time axis
        nt = round(fw*3.75/dt)
        tf = np.arange(-nt, nt+1)*dt

        # Make filter
        rf = np.exp(-tf**2/(2.*fw**2))

    elif filter_type == 'box':

        # Filter parameters
        fw = filter_width

        # Box width
        nt = round(fw/2/dt)

        # Make filter
        rf = np.ones(2*nt+1)

    rf /= sum(rf)

    # Initialize
    response_smth = response

    # if len(rf) < 2:
    #     return response_smth

    # Temporal smoothing
    # Dimensions
    n_trial, n_time = response.shape

    from scipy.signal import convolve

    # Loop over trials
    for i_trial in range(n_trial):
        # Pad extremes
        rpad = np.concatenate((np.repeat(response[i_trial,0], len(rf)),
                               response[i_trial,:],
                               np.repeat(response[i_trial,-1], len(rf))))

        filtered = convolve(rpad, rf, mode='same')

        # Shift
        rfil = filtered[len(rf):len(rf)+n_time]

        response_smth[i_trial] = rfil

    return response_smth


def save_smoothed_data():
    f_loadname = os.path.join(DATAPATH, 'dataT.mat')

    mat_dict = loadmat(f_loadname, squeeze_me=True, struct_as_record=False)
    data = mat_dict['dataT'].unit

    # Smooth
    for i in range(len(data)):
        data[i].response = smoothing(data[i].response, mat_dict['dataT'].time)

    from scipy.io import savemat
    f_savename = os.path.join(DATAPATH, 'dataT_smooth.mat')
    savemat(f_savename, mat_dict, do_compression=True)


def get_vars(random_shuffle=False):
    var1s = list()
    var2s = list()
    # Condition average activity
    for i_unit in range(n_unit):

        response = data[i_unit].response # (trial, time)
        task_var = data[i_unit].task_variable.__dict__ # turn into dictionary

        ind_unit1 = (task_var['context']== 1)
        ind_unit2 = (task_var['context']==-1)

        if random_shuffle:
            np.random.shuffle(ind_unit1)
            np.random.shuffle(ind_unit2)

        rate1_unit = response[ind_unit1, :]
        rate2_unit = response[ind_unit2, :]

        var1s.append(rate1_unit.mean(axis=1).var())
        var2s.append(rate2_unit.mean(axis=1).var())

    var1s = np.array(var1s)
    var2s = np.array(var2s)

    return var1s, var2s


def plot_vars():
    var1s, var2s = get_vars()

    start = time.time()
    n_rep = 50
    var1s_shuffle, var2s_shuffle = 0, 0
    for i_rep in range(n_rep):
        v1, v2 = get_vars(random_shuffle=True)
        var1s_shuffle += v1
        var2s_shuffle += v2

    var1s_shuffle /= n_rep
    var2s_shuffle /= n_rep

    print(time.time()-start)

    plt.figure()
    plt.scatter(var1s_shuffle, var2s_shuffle)

    plt.figure()
    plt.scatter(var1s, var2s)

    plt.figure()
    plt.scatter(var1s, var1s_shuffle)
    plt.plot([0, 1000], [0, 1000])


def compute_var(rates, var_method):
    """Compute task variance.

    Args:
        rates: numpy array (n_cond, n_time, n_unit),
            trial-averaged activities for context 1
        var_method: str, method to compute variance
            Can be 'time_avg_late', 'time_avg_none', 'time_avg_early'

    Return:
        vars: numpy array (n_unit,), task variance
    """
    if var_method == 'time_avg_late':
        # variance across conditions, then average across time
        vars = rates.var(axis=0).mean(axis=0)
    elif var_method == 'time_avg_none':
        # variance across conditions and time
        # The noise is much stronger in this case
        # Even after shuffling we see trimodal distribution in this case.
        # So we shouldn't use this method
        vars = rates.reshape((-1, rates.shape[-1])).var(axis=0)
    elif var_method == 'time_avg_early':
        # first average across time, then variance across conditions
        # The noise is much weaker in this case
        vars = rates.mean(axis=1).var(axis=0)
    else:
        raise ValueError('Variance method var_method unrecognized')

    return vars


# def get_shuffle_var(data, **kwargs):
#     start = time.time()
#     n_rep = 200
#     var1s_shuffle, var2s_shuffle = 0, 0
#     for i_rep in range(n_rep):
#         v1, v2 = get_trial_avg_var(data, random_shuffle=True, **kwargs)
#         var1s_shuffle += v1
#         var2s_shuffle += v2
#
#     var1s_shuffle /= n_rep
#     var2s_shuffle /= n_rep
#
#     print(time.time()-start)
#
#     return var1s_shuffle, var2s_shuffle
#
#     # Threshold should be 3.65
#     # return np.percentile(var1s_shuffle, [95]) + np.percentile(var2s_shuffle, [95])


def plot_fracVar(fracVar, save_name=None, fancy_color=False):
    """Plot distribution of fractional variance."""
    ind_group = dict()
    ind_group['1'] = np.where(fracVar > 0.8)[0]
    ind_group['2'] = np.where(fracVar < -0.8)[0]
    # ind_group['12']= np.where(np.logical_and(-0.8<fracVar, fracVar<0.8))[0]
    ind_group['12'] = np.where(np.logical_and(-0.3 < fracVar,
                                              fracVar < 0.3))[0]

    colors = dict(zip([None, '1', '2', '12'], COLORS))
    fs = 6
    fig = plt.figure(figsize=(2.0, 1.2))
    ax = fig.add_axes([0.2, 0.3, 0.5, 0.5])
    data_plot = fracVar
    hist, bins_edge = np.histogram(data_plot, bins=20, range=(-1, 1))
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1] - bins_edge[0],
           color='gray', edgecolor='none')
    if fancy_color:
        bs = list()
        for i, group in enumerate(['1', '2', '12']):
            data_plot = fracVar[ind_group[group]]
            hist, bins_edge = np.histogram(data_plot, bins=20, range=(-1, 1))
            b_tmp = ax.bar(bins_edge[:-1], hist,
                           width=bins_edge[1] - bins_edge[0],
                           color=colors[group], edgecolor='none', label=group)
            bs.append(b_tmp)
    plt.locator_params(nbins=3)
    xlabel = 'FracVar'
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel('Units', fontsize=fs)
    ax.set_ylim(bottom=-0.02 * hist.max())
    ax.set_xlim([-1.1, 1.1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
    if fancy_color:
        lg = ax.legend(bs, ['1', '2', '12'], title='Group',
                       fontsize=fs, ncol=1, bbox_to_anchor=(1.5, 1.0),
                       loc=1, frameon=False)
        plt.setp(lg.get_title(), fontsize=fs)

    fig_name = 'figure/fracVar_data'
    if save_name is not None:
        fig_name += save_name
    plt.savefig(fig_name + '.pdf', transparent=True)


def clustering_tmp():
    # Perform clustering
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=0.1, min_samples=20).fit(coefs)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)


def study_betaweights(analyze_single_units=True, ind_group=None):
    fname = os.path.join(DATAPATH, 'vBeta.mat')

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)

    vBeta = mat_dict['vBeta'].__dict__ # as dictionary

    coefs = vBeta['response']

    if analyze_single_units:
        print('Analyzing single units')
        single_units = get_single_units()
        coefs_show = coefs[single_units, :]
    else:
        print('Analyzing all units')
        coefs_show = coefs

    colors = dict(zip([None, '1', '2', '12'], COLORS))

    # fig, axarr = plt.subplots(3, 2, figsize=(4,5))
    # pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    fig, axarr = plt.subplots(1, 2, figsize=(4,2))
    pairs = [(0, 3), (1, 2)]

    regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']

    fs = 6
    lim = 0.5
    for i_plot in range(len(pairs)):
        i, j = pairs[i_plot]
        # ax = axarr[i_plot//2, i_plot%2]
        ax = axarr[i_plot]
        color = 'gray'
        ax.plot(coefs_show[:,i], coefs_show[:,j], 'o', color=color, ms=3, mec='white', mew=0.1, alpha=0.3)

        ax.plot([-lim, lim], [0, 0], color='gray')
        ax.plot([0, 0], [-lim, lim], color='gray')
        ax.set_xlabel(regr_names[i], fontsize=fs)
        ax.set_ylabel(regr_names[j], fontsize=fs)
        if i == 0:
            ax.set_xlim([-0.1, 0.4])
            ax.set_xticks([-0.1,0,0.4])
        else:
            ax.set_xlim([-0.25, 0.25])
            ax.set_xticks([-0.25,0,0.25])
        if j == 0:
            ax.set_ylim([-0.1, 0.4])
            ax.set_yticks([-0.1,0,0.4])
        else:
            ax.set_ylim([-0.25, 0.25])
            ax.set_yticks([-0.25,0,0.25])

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.locator_params(nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if ind_group is None:
            continue

        for group in ['12', '1', '2']:
            if group is not None:
                color = colors[group]
            else:
                color = 'gray'

            ax.plot(coefs[ind_group[group],i], coefs[ind_group[group],j],
                    'o', color=color, ms=3, mec='white', mew=0.1)

    plt.tight_layout()

    save_name = 'betaweightscolored_data'
    if analyze_single_units:
        save_name = save_name + '_SU'
    else:
        save_name = save_name + '_AU'

    plt.savefig(os.path.join('figure',save_name+'.pdf'), transparent=True)


    for group in ['12', '1', '2']:
        print('Group '+group)
        print(np.median(coefs[ind_group[group]], axis=0))

    # for i_coef in [1, 2]:
    #     fig = plt.figure(figsize=(4.0,3))
    #     ax = fig.add_axes([0.2,0.3,0.5,0.5])
    #     for i_group, group in enumerate(['1', '2']):
    #         hist, bins_edge = np.histogram(coefs[ind_group[group],i_coef], bins=20, range=(-0.2,0.3))
    #         hist = hist/hist.sum()
    #         ax.plot((bins_edge[:-1]+bins_edge[1:])/2, hist, '-', color=colors[group], linewidth=2)
    #     ax.set_xlabel(regr_names[i_coef])
    #
    # # Statistics
    # def get_mean_confidenceinterval(data):
    #     n_data = len(data)
    #     n_rep = 10000
    #     mean_list = np.zeros(n_rep)
    #     for i_rep in range(n_rep):
    #         ind = np.random.choice(range(n_data), size=(n_data,), replace=True)
    #         mean_list[i_rep] = data[ind].mean()
    #     return np.percentile(mean_list, [2.5, 97.5])
    #
    # def compare_twogroups(data0, data1):
    #     n_data0 = len(data0)
    #     n_data1 = len(data1)
    #     n_rep = 10000
    #     mean_list0 = np.zeros(n_rep)
    #     mean_list1 = np.zeros(n_rep)
    #     for i_rep in range(n_rep):
    #         ind0 = np.random.choice(range(n_data0), size=(n_data0,), replace=True)
    #         ind1 = np.random.choice(range(n_data1), size=(n_data1,), replace=True)
    #         mean_list0[i_rep] = data0[ind0].mean()
    #         mean_list1[i_rep] = data1[ind1].mean()
    #     p0 = 1 - np.mean(mean_list0 > mean_list1)
    #     p1 = 1 - np.mean(mean_list0 < mean_list1)
    #     if p0 < p1:
    #         stronger = 0
    #         p = p0
    #     else:
    #         stronger = 1
    #         p = p1
    #     return stronger, p
    #
    # for i_coef in [1, 2]:
    #     group1, group2 = '1', '2'
    #     stronger, p = compare_twogroups(coefs[ind_group[group1],i_coef],
    #                                     coefs[ind_group[group2],i_coef])
    #
    #     print(regr_names[i_coef], group1, group2)
    #     print('Stronger group: ' + [group1, group2][stronger])
    #     print('p value: {:0.7f}'.format(p))


def compute_var_all(data, var_method='time_avg_early'):
    """Compute task variance for data and shuffled data.

    Args:
        var_method: str, method to compute the variance
            Can be 'time_avg_late', 'time_avg_none', 'time_avg_early'
    """

    # Generate a random orthogonal matrix for later use
    # rotation_matrix = gen_ortho_matrix(self.n_unit) # has to be the same matrix

    var1s, var2s = get_trial_avg_var(data, var_method)
    var1s_shuffle, var2s_shuffle = get_shuffle_var(data, var_method)

    var_dict = {'var1s': var1s, 'var2s': var2s,
                'var1s_shuffle': var1s_shuffle, 'var2s_shuffle': var2s_shuffle}
    return var_dict


def compute_frac_var(var_dict, var_thr=0.0, save_name=None,
                        thr_type='sum'):

    var1s = var_dict['var1s']
    var2s = var_dict['var2s']
    var1s_shuffle = var_dict['var1s_shuffle']
    var2s_shuffle = var_dict['var2s_shuffle']

    # Plot vars
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.25, 0.25, 0.65, 0.65])
    ax.scatter(np.concatenate((var1s, var2s)),
                np.concatenate((var1s_shuffle, var2s_shuffle)), s=4)
    ax.plot([0,500], [0,500])
    lim = np.max(np.concatenate(
        (var1s, var2s, var1s_shuffle, var2s_shuffle)))
    plt.xlim([0,lim*1.1])
    plt.ylim([0,lim*1.1])
    plt.xlabel('Stimulus variance', fontsize=7)
    plt.ylabel('Stimulus variance (shuffled data)', fontsize=7)
    # title = self.var_method + ' rand. rot. ' + str(random_rotation)
    # plt.title(title, fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.locator_params(nbins=3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    fig_name = 'figure/stimvarvsshuffle'
    if save_name is not None:
        fig_name += save_name
    plt.savefig(fig_name + '.pdf', transparent=True)
    #
    # plt.figure()
    # plt.scatter(var1s_shuffle, var2s_shuffle)
    # plt.plot([0,500], [0,500])
    # plt.xlim([0,100])
    # plt.ylim([0,100])
    # plt.xlabel('Shuffled stim. var. contex 1')
    # plt.ylabel('Shuffled stim. var. contex 2')
    # plt.title(self.var_method + ' rand. rot. '+str(random_rotation))

    # Denoise
    # if denoise:
    #     relu = lambda x : x*(x>0.)
    #     var1s = relu(var1s - var1s_shuffle)
    #     var2s = relu(var2s - var2s_shuffle)

    if thr_type == 'sum':
        ind = (var1s+var2s)>var_thr
    elif thr_type == 'and':
        ind = np.logical_and(var1s>var_thr, var2s>var_thr)
    elif thr_type == 'or':
        ind = np.logical_or(var1s>var_thr, var2s>var_thr)
    else:
        raise ValueError('Unknown threshold type')

    var1s = var1s[ind]
    var2s = var2s[ind]
    # ind_original = self.ind_original[ind]

    fracVar = (var1s-var2s)/(var1s+var2s)

    # Plot fracVar distribution
    # fig = plt.figure(figsize=(4.0,3))
    # ax = fig.add_axes([0.2,0.3,0.5,0.5])
    # hist, bins_edge = np.histogram(fracVar, bins=20, range=(-1,1))
    # ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
    #        color='blue', edgecolor='none')
    # ax.set_title(self.var_method + ' rand. rot. '+str(random_rotation))
    #
    # save_name = 'fracVar_data'
    # if random_rotation:
    #     save_name = save_name + '_rndrot'
    # plt.savefig(os.path.join('figure',save_name+'.pdf'), transparent=True)

    return fracVar

# def plot_single_unit(self, i_unit, save=False):
#     context_names = ['motion', 'color']
#     colors = ['red', 'black', 'blue']
#     fig, axarr = plt.subplots(3, 2, figsize=(4,4.0), sharey=True)
#     for i_ax in [0,1]:
#         for i_var_key, var_key in enumerate(self.var_key_list):
#             ax = axarr[i_var_key, i_ax]
#
#             data_ = self.data_condavg_list[i_ax][var_key][:,:,i_unit]
#             color=colors[i_var_key]
#             ax.plot(data_[:,0], '--', color=color)
#             ax.plot(data_[:,1], color=color)
#
#             if i_ax == 0:
#                 ax.set_ylabel(var_key)
#
#             if i_var_key == 0:
#                 ax.set_title('context '+context_names[i_ax])
#
#     if save:
#         save_name = 'mante_unit{:d}'.format(i_unit)
#         plt.savefig(os.path.join('figure',save_name+'.pdf'), transparent=True)


if __name__ == '__main__':
    pass
    analyze_single_units = False
    denoise = False

    root_dir = './data/varyhparams'
    hp_target = {'activation': 'retanh',
                 'rnn_type': 'LeakyGRU',
                 'w_rec_init': 'randortho',
                 'l1_h': 0,
                 'l1_weight': 1e-5}
    model_dir, _ = tools.find_model(root_dir, hp_target)
    dataset = 'model'

    dataset, model_dir = 'mante', None
    data = load_data(dataset=dataset,
                      analyze_single_units=analyze_single_units,
                      model_dir=model_dir)
    var_dict = compute_var_all(data, var_method='time_avg_late')
    var_thr, thr_type = 0.05, 'or'
    fracVar = compute_frac_var(var_dict, var_thr=var_thr, thr_type=thr_type)
    plot_fracVar(fracVar, save_name=dataset)
