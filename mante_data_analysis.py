"""
Data analysis
Delay-match-to-category task: The boundary is at 45 and 225 for both monkeys
"""

from __future__ import division

import os
import sys
import numpy as np
import pickle
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.io import loadmat, whosmat
import seaborn as sns

from network import gen_ortho_matrix


DATAPATH = './datasets/mante_dataset'


def load_mante_data(smooth=True):
    if smooth:
        fname = 'dataT_smooth.mat'
    else:
        fname = 'dataT.mat'

    fname = os.path.join(DATAPATH, fname)

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)

    dataT = mat_dict['dataT'].__dict__ # as dictionary

    data = dataT['unit']
    # time = dataT['time']

    # Number of units
    n_unit = len(data)
    n_cond = 72 # condition
    n_time = data[0].response.shape[1]
    return data


def generate_data_in_standard_format():
    '''Generate data dictionary in standardize form'''

    datas = load_mante_data(smooth=True)

    # List of data, each item corresponds to one unit
    Data = list()
    for data in datas:
        task_var = data.task_variable

        # For computing the coherence ID (integer)
        unique_mod1coh = np.unique(task_var.stim_dir) # return sorted unique items
        unique_mod2coh = np.unique(task_var.stim_col2dir)
        assert len(unique_mod1coh)==len(unique_mod2coh)==6

        # map coherence to their ID
        coh_ids = [-3, -2, -1, +1, +2, +3]
        mod1coh_map = dict([(c, i) for c, i in zip(unique_mod1coh, coh_ids)])
        mod2coh_map = dict([(c, i) for c, i in zip(unique_mod2coh, coh_ids)])

        Data.append({
            'Activity'          : data.response, # (n_trial, n_timepoint)
            'Context'           : task_var.context,
            # Redefine choice, (+1 for the preferred choice)
            'ActualChoiceRedef' : task_var.targ_dir,
            'Mod1Coh'           : (task_var.stim_dir/np.max(task_var.stim_dir)).astype(np.float16),
            'Mod2Coh'           : (task_var.stim_col2dir/np.max(task_var.stim_col2dir)).astype(np.float16),
            'Mod1CohID'         : np.array([mod1coh_map[c] for c in task_var.stim_dir], dtype=np.int16),
            'Mod2CohID'         : np.array([mod2coh_map[c] for c in task_var.stim_col2dir], dtype=np.int16),
            'Correct'           : task_var.correct,
                })

    import pickle, os
    with open(os.path.join('data', 'ManteData.pkl'), 'wb') as f:
        pickle.dump(Data, f)


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


def get_single_units():
    # get single units
    fname = os.path.join('mante_dataset', 'metadata.mat')
    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)
    metadata = mat_dict['metadata'].unit # as dictionary
    discriminable = np.array([(m.unitInfo.discriminability in [3,4]) for m in metadata])
    single_units = np.array([m.unitInfo.type=='s' for m in metadata])
    single_units = np.logical_and(single_units, discriminable)
    assert np.sum(single_units)==181
    return single_units


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


def get_trial_avg_rate(data_unit, context=1, random_shuffle=False,
                       return_avg=True, only_correct=False):
    response = data_unit.response # (trial, time)
    task_var = data_unit.task_variable.__dict__ # turn into dictionary

    ctx = task_var['context'] == context

    tmp = list()
    for m in [1, -1]:
        for c in [1, -1]:
            # for choice in [1, -1]:
            if m == 1:
                m_ind = task_var['stim_dir'] > 0 # motion positive
            else:
                m_ind = task_var['stim_dir'] < 0 # motion positive

            if c == 1:
                c_ind = task_var['stim_col2dir'] > 0 # motion positive
            else:
                c_ind = task_var['stim_col2dir'] < 0 # motion positive

            # if choice == 1:
            #     choice_ind = task_var['targ_dir'] > 0
            # else:
            #     choice_ind = task_var['targ_dir'] < 0

            # ind = m_ind*c_ind*ctx*choice_ind
            ind = m_ind*c_ind*ctx

            if only_correct:
                ind = ind*task_var['correct']

            if random_shuffle:
                np.random.shuffle(ind)

            ind = ind.astype(bool)

            if return_avg:
                # tmp.append(response[ind,:].mean())
                # TODO: TEMPORARY
                tmp.append(response[ind,:].mean(axis=0)) # average across trials
            else:
                tmp.append(response[ind,:])

    if return_avg:
        return np.array(tmp) # (n_condition, n_time)
    else:
        return tmp


def get_shuffle_var(data, **kwargs):
    start = time.time()
    n_rep = 200
    var1s_shuffle, var2s_shuffle = 0, 0
    for i_rep in range(n_rep):
        v1, v2 = get_trial_avg_var(data, random_shuffle=True, **kwargs)
        var1s_shuffle += v1
        var2s_shuffle += v2

    var1s_shuffle /= n_rep
    var2s_shuffle /= n_rep

    print(time.time()-start)

    return var1s_shuffle, var2s_shuffle

    # Threshold should be 3.65
    # return np.percentile(var1s_shuffle, [95]) + np.percentile(var2s_shuffle, [95])


def plot_fracVar(data, analyze_units='single', plot_single_units=False):
    print('Analyzing' + analyze_units + 'units')

    single_units = get_single_units()
    multi_units = np.logical_not(single_units)

    _var1s, _var2s = get_trial_avg_var(data, random_rotation=random_rotation)
    _var1s_shuffle, _var2s_shuffle = get_shuffle_var(data, random_rotation=random_rotation)

    var1s, var2s = get_trial_avg_var(data, random_shuffle=False)

    n_unit = len(data)

    ind_orig = np.arange(n_unit)

    # Get units with variance higher than threshold
    # ind_used = (var1s+var2s)>3.65
    var_thr = 0.00
    ind_used = (var1s+var2s)>var_thr

    if analyze_units == 'single':
        ind_used = np.logical_and(ind_used, single_units)
    elif analyze_units == 'multi':
        ind_used = np.logical_and(ind_used, multi_units)
        # ind_used = single_units

    var1s = var1s[ind_used]
    var2s = var2s[ind_used]
    ind_used_orig = ind_orig[ind_used]

    fracVar = (var1s-var2s)/(var1s+var2s)
    plt.figure()
    _ = plt.hist(fracVar, bins=10, range=(-1,1))

    if plot_single_units:
        # ind_sort = np.argsort(fracVar)
        i_units = np.argsort(fracVar)[:5]
        i_units = np.concatenate((i_units, np.argsort(fracVar)[-5:]))
        fracVar_tmp = fracVar[i_units]
        # print(fracVar_tmp)

        i_units = np.argsort(fracVar)

        i_units = ind_used_orig[i_units]
        for i_unit in i_units:
            plot_singleunit(data, i_unit)


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

    import seaborn.apionly as sns # If you don't have this, then some colormaps won't work
    colors = dict(zip([None, '1', '2', '12'],
                       sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))

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


class AnalyzeManteData(object):

    def __init__(self, smooth=True, analyze_single_units=True):
        """Analyzing the Mante dataset.

        Args:
            smooth: bool, whether to use smoothed data
            analyze_single_units: bool, whether to analyze single units
        """
        if smooth:
            print('Loading smooth data')
        else:
            print('Loading original data')
        data = load_mante_data(smooth=smooth)

        ind_original = np.arange(len(data))

        if analyze_single_units:
            single_units = get_single_units()
            ind_single_units = np.where(single_units)[0]
            self.data = [data[i] for i in ind_single_units]
            self.ind_original = ind_original[single_units]
        else:
            self.data = data
            self.ind_original = ind_original

        self.n_unit = len(self.data)

        from dimensionality import get_trial_avg
        self.var_key_list = ['targ_dir', 'stim_dir_sign', 'stim_col2dir_sign']
        self.data_condavg_list = [] # for two contexts
        for context in [1, -1]:
            tmp = dict()
            for var_key in self.var_key_list:
                # TEMPORARY TODO
                # if var_key == 'targ_dir':
                #     context_ = None
                # else:
                #     context_ = context

                tmp[var_key] = get_trial_avg(analyze_data=True, split_traintest=False,
                                           var_keys=[var_key], context=context, data=data)
            self.data_condavg_list.append(tmp)


    def compute_var_all(self, var_method='time_avg_early'):

        self.var_method = var_method

        # Generate a random orthogonal matrix for later use
        self.rotation_matrix = gen_ortho_matrix(self.n_unit) # has to be the same matrix

        self._var1s, self._var2s, self._var1s_rot, self._var2s_rot = self.get_trial_avg_var()
        self._var1s_shuffle, self._var2s_shuffle, \
        self._var1s_rot_shuffle, self._var2s_rot_shuffle = self.get_shuffle_var()


    def get_trial_avg_var(self, random_shuffle=False):

        data = self.data
        n_unit = len(data)

        rate1s = list()
        rate2s = list()
        # Condition average activity
        for i_unit in range(n_unit):
            # Get trial-averaged condition-based responses (n_condition, n_time)
            rate1s.append(get_trial_avg_rate(data[i_unit], context=1, random_shuffle=random_shuffle))
            rate2s.append(get_trial_avg_rate(data[i_unit], context=-1, random_shuffle=random_shuffle))

        # (n_unit, n_condition, n_time)
        rate1s, rate2s = np.array(rate1s), np.array(rate2s)

        # Rotate with random orthogonal matrix
        rate1s_rot = np.swapaxes(rate1s, 0, 1)
        rate2s_rot = np.swapaxes(rate2s, 0, 1)
        rate1s_rot = np.dot(self.rotation_matrix, rate1s_rot)
        rate2s_rot = np.dot(self.rotation_matrix, rate2s_rot)

        var1s, var2s = self.compute_var(rate1s, rate2s)
        var1s_rot, var2s_rot = self.compute_var(rate1s_rot, rate2s_rot)

        return var1s, var2s, var1s_rot, var2s_rot

    def compute_var(self, rate1s, rate2s):
        # Rate1s and rate2s are arrays (n_unit, n_condition, n_time)
        # They are trial-averaged activities for context 1 and context 2 respectively

        if self.var_method == 'time_avg_late':
            # variance across conditions, then average across time
            var1s = rate1s.var(axis=1).mean(axis=1)
            var2s = rate2s.var(axis=1).mean(axis=1)
        elif self.var_method == 'time_avg_none':
            # variance across conditions and time
            # The noise is much stronger in this case
            # Even after shuffling we see trimodal distribution in this case.
            # So we shouldn't use this method
            var1s = rate1s.reshape((rate1s.shape[0], -1)).var(axis=1)
            var2s = rate2s.reshape((rate2s.shape[0], -1)).var(axis=1)
        elif self.var_method == 'time_avg_early':
            # first average across time, then variance across conditions
            # The noise is much weaker in this case
            var1s = rate1s.mean(axis=2).var(axis=1)
            var2s = rate2s.mean(axis=2).var(axis=1)
        else:
            raise ValueError('Variance method var_method unrecognized')

        return var1s, var2s

    def get_shuffle_var(self, n_rep=10, **kwargs):
        start = time.time()

        var1s_shuffle, var2s_shuffle = 0, 0
        var1s_rot_shuffle, var2s_rot_shuffle = 0, 0
        for i_rep in range(n_rep):
            v1, v2, v1_rot, v2_rot = self.get_trial_avg_var(random_shuffle=True, **kwargs)
            var1s_shuffle += v1
            var2s_shuffle += v2
            var1s_rot_shuffle += v1_rot
            var2s_rot_shuffle += v2_rot

        var1s_shuffle /= n_rep
        var2s_shuffle /= n_rep
        var1s_rot_shuffle /= n_rep
        var2s_rot_shuffle /= n_rep

        print('Time taken {:0.2f}s'.format(time.time()-start))

        return var1s_shuffle, var2s_shuffle, var1s_rot_shuffle, var2s_rot_shuffle

    def compute_denoise_var(self, var_thr=0.0, random_rotation=False,
                            thr_type='sum', denoise=True):

        if random_rotation:
            var1s = self._var1s_rot
            var2s = self._var2s_rot
            var1s_shuffle = self._var1s_rot_shuffle
            var2s_shuffle = self._var2s_rot_shuffle
        else:
            var1s = self._var1s
            var2s = self._var2s
            var1s_shuffle = self._var1s_shuffle
            var2s_shuffle = self._var2s_shuffle

        # Plot vars
        plt.figure()
        plt.scatter(var1s, var1s_shuffle)
        plt.plot([0,500], [0,500])
        plt.xlim([0,100])
        plt.ylim([0,100])
        plt.xlabel('Stimulus variance')
        plt.ylabel('Stimulus variance (shuffled data)')
        plt.title(self.var_method + ' rand. rot. '+str(random_rotation))
        plt.savefig('figure/stimvarvsshuffle.pdf')

        plt.figure()
        plt.scatter(var1s_shuffle, var2s_shuffle)
        plt.plot([0,500], [0,500])
        plt.xlim([0,100])
        plt.ylim([0,100])
        plt.xlabel('Shuffled stim. var. contex 1')
        plt.ylabel('Shuffled stim. var. contex 2')
        plt.title(self.var_method + ' rand. rot. '+str(random_rotation))

        # Denoise
        if denoise:
            relu = lambda x : x*(x>0.)
            var1s = relu(var1s - var1s_shuffle)
            var2s = relu(var2s - var2s_shuffle)

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
        ind_original = self.ind_original[ind]

        fracVar = (var1s-var2s)/(var1s+var2s)


        # Plot fracVar distribution
        fig = plt.figure(figsize=(4.0,3))
        ax = fig.add_axes([0.2,0.3,0.5,0.5])
        hist, bins_edge = np.histogram(fracVar, bins=20, range=(-1,1))
        ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
               color='blue', edgecolor='none')
        ax.set_title(self.var_method + ' rand. rot. '+str(random_rotation))

        save_name = 'fracVar_data'
        if random_rotation:
            save_name = save_name + '_rndrot'
        # plt.savefig(os.path.join('figure',save_name+'.pdf'), transparent=True)

        return fracVar, ind_original

    def plot_single_unit(self, i_unit, save=False):
        context_names = ['motion', 'color']
        colors = ['red', 'black', 'blue']
        fig, axarr = plt.subplots(3, 2, figsize=(4,4.0), sharey=True)
        for i_ax in [0,1]:
            for i_var_key, var_key in enumerate(self.var_key_list):
                ax = axarr[i_var_key, i_ax]

                data_ = self.data_condavg_list[i_ax][var_key][:,:,i_unit]
                color=colors[i_var_key]
                ax.plot(data_[:,0], '--', color=color)
                ax.plot(data_[:,1], color=color)

                if i_ax == 0:
                    ax.set_ylabel(var_key)

                if i_var_key == 0:
                    ax.set_title('context '+context_names[i_ax])

        if save:
            save_name = 'mante_unit{:d}'.format(i_unit)
            plt.savefig(os.path.join('figure',save_name+'.pdf'), transparent=True)


if __name__ == '__main__':
    pass

    # for var_method in ['time_avg_early', 'time_avg_late', 'time_avg_none']:
    #     amd = AnalyzeManteData(random_rotation=False, var_method=var_method)
    #     amd.compute_denoise_var()
    #
    #     amd = AnalyzeManteData(random_rotation=True, var_method=var_method)
    #     amd.compute_denoise_var()

    analyze_single_units = False
    denoise = False

    amd = AnalyzeManteData(analyze_single_units=analyze_single_units)
    amd.compute_var_all(var_method='time_avg_late')
    var_thr, thr_type = 0.0, 'or'
    fracVar, ind_original = amd.compute_denoise_var(
        var_thr=var_thr, thr_type=thr_type, denoise=denoise)
#==============================================================================
#     _ = amd.compute_denoise_var(random_rotation=True, var_thr=var_thr,
#                                 thr_type=thr_type)
#==============================================================================

    ind_group = dict()
    ind_group['1'] = np.where(fracVar>0.8)[0]
    ind_group['2'] = np.where(fracVar<-0.8)[0]
    # ind_group['12']= np.where(np.logical_and(-0.8<fracVar, fracVar<0.8))[0]
    ind_group['12']= np.where(np.logical_and(-0.3<fracVar, fracVar<0.3))[0]

    colors = dict(zip([None, '1', '2', '12'],
                       sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))
    fs = 6
    fig = plt.figure(figsize=(2.0,1.2))
    ax = fig.add_axes([0.2,0.3,0.5,0.5])
    data_plot = fracVar
    hist, bins_edge = np.histogram(data_plot, bins=20, range=(-1,1))
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
           color='gray', edgecolor='none')
    bs = list()
    for i, group in enumerate(['1', '2', '12']):
        data_plot = fracVar[ind_group[group]]
        hist, bins_edge = np.histogram(data_plot, bins=20, range=(-1,1))
        b_tmp = ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
               color=colors[group], edgecolor='none', label=group)
        bs.append(b_tmp)
    plt.locator_params(nbins=3)
    xlabel = 'FracVar'
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel('Units', fontsize=fs)
    ax.set_ylim(bottom=-0.02*hist.max())
    ax.set_xlim([-1.1,1.1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
    lg = ax.legend(bs, ['1','2','12'], title='Group',
                   fontsize=fs, ncol=1, bbox_to_anchor=(1.5,1.0),
                   loc=1, frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)

    save_name = 'fracVarcolored_data'
    if analyze_single_units:
        save_name = save_name + '_SU'
    else:
        save_name = save_name + '_AU'
    if not denoise:
        save_name = save_name + '_nodenoise'
    # plt.savefig('figure/'+save_name+'.pdf', transparent=True)

    ind_group_original = dict()
    for group in ['1', '2', '12']:
        ind_group_original[group] = ind_original[ind_group[group]]

    # TEMPORARY
    fname = os.path.join(DATAPATH, 'vBeta.mat')

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)

    vBeta = mat_dict['vBeta'].__dict__ # as dictionary

    coefs = vBeta['response']

    # from contextdm_analysis import plot_coefvsfracvar
    #
    # plot_coefvsfracvar(coefs[ind_original,:], fracVar)


#==============================================================================
#     study_betaweights(analyze_single_units, ind_group=ind_group_original)
# 
#     data = load_mante_data(smooth=True)
#     for group in ['1']:
#         for i_unit in ind_group_original[group]:
#             amd.plot_single_unit(i_unit)
#==============================================================================
