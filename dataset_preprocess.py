"""Dataset preprocessing.

Standardize several datasets into the same format for task variance analysis.
Run siegel_preprocess.m before for correct results from the Siegel dataset.
"""

from __future__ import division

import os
import csv
from collections import defaultdict
import math
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

DATASETPATH = './datasets'


def load_mante_data(smooth=True):
    """Load Mante data into raw format.

    Args:
        smooth: bool, whether to load smoothed data

    Returns:
        data: a list of mat_struct, storing info for each unit
    """
    datasetpath = os.path.join(DATASETPATH, 'mante_dataset')
    if smooth:
        fname = 'dataT_smooth.mat'
    else:
        fname = 'dataT.mat'

    fname = os.path.join(datasetpath, fname)

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)

    dataT = mat_dict['dataT'].__dict__ # as dictionary

    data = dataT['unit']
    # time = dataT['time']

    # Number of units
    # n_unit = len(data)
    # n_cond = 72  # condition
    # n_time = data[0].response.shape[1]
    return data


def get_trial_avg_rate_mante(data_unit, context=1, random_shuffle=False,
                       return_avg=True, only_correct=False):
    """Get trial-averaged rate activity for Mante dataset.

    Args:
        data_unit: mat_struct, loaded from load_mante_data
        context: +1 or -1, the context to study
        random_shuffle: bool, whether to random shuffle trials
        return_avg: bool, whether to return the average across trials
        only_correct: bool, if True, only analyze correct trials

    Returns:
        data_avg: numpy array (n_condition, n_time),
            the trial-averaged activity
    """
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


def get_mante_data():
    """Get Mante data in standard format.

    Returns:
        rate1s: numpy array (n_unit, n_condition, n_time) in context 1
        rate2s: numpy array (n_unit, n_condition, n_time) in context 2
    """
    data = load_mante_data()

    n_unit = len(data)

    rate1s = list()
    rate2s = list()
    random_shuffle = True
    for i_unit in range(n_unit):
        # Get trial-averaged condition-based responses (n_condition, n_time)
        rate1s.append(get_trial_avg_rate_mante(data[i_unit], context=1,
                                         random_shuffle=random_shuffle))
        rate2s.append(get_trial_avg_rate_mante(data[i_unit], context=-1,
                                         random_shuffle=random_shuffle))
    # (n_unit, n_condition, n_time)
    rate1s, rate2s = np.array(rate1s), np.array(rate2s)
    return rate1s, rate2s


def _load_siegel_spikes(fname):
    """Load spiking data from Siegel dataset."""
    fname_full = os.path.join(
        DATASETPATH, 'siegel_dataset', 'sorted', fname)

    # Analyze a single file
    mat_dict = loadmat(fname_full, squeeze_me=True, struct_as_record=False)

    # Get spike times
    # spikes is an array (trials, neurons) of 1-D array
    spikes = mat_dict['spikeTimes']
    return spikes


def _load_siegel_tables(fname, table_name):
    """Load tables from Siegel dataset and convert to dictionary.

    Args:
        fname: str, file name
        table_name: str, can be 'trialinfo', 'unitinfo', and 'electrodeinfo'

    Returns:
        table_dict: dictionary, for each (key, val) pair, val is an array
    """
    fname_full = os.path.join(DATASETPATH, 'siegel_dataset',
                              table_name, fname[:6] + '.csv')

    table_dict = defaultdict(list)
    with open(fname_full) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        keys = csvReader.next()
        for row in csvReader:
            for k, r in zip(keys, row):
                # r is a string, convert to int, float, or string
                try:
                    r2 = int(r)
                except ValueError:
                    try:
                        r2 = float(r)
                    except ValueError:
                        r2 = r

                table_dict[k].append(r2)
    return table_dict





if __name__ == '__main__':
    # rate1s, rate2s = get_mante_data()
    
    datasetpath = os.path.join(DATASETPATH, 'siegel_dataset', 'sorted')
    
    files = os.listdir(datasetpath)
    files = [f for f in files if '1' in f]

    f = files[0]

    trial_infos = _load_siegel_tables(f, 'trialinfo')
    unit_infos = _load_siegel_tables(f, 'unitinfo')
    electrode_infos = _load_siegel_tables(f, 'electrodeinfo')

    spikes = _load_siegel_spikes(f)
    n_trial, n_unit = spikes.shape

    assert len(trial_infos.values()[0]) == n_trial
    assert len(unit_infos.values()[0]) == n_unit

    # Get valid trials
    new_trial_infos = defaultdict(list)
    keys = trial_infos.keys()
    valid_trials = list()
    for i_trial in range(n_trial):
        if (trial_infos['badTrials'][i_trial] == 1 or
            math.isnan(trial_infos['responseTime'][i_trial])):
            valid = False
        else:
            valid = True
        valid_trials.append(valid)
        if valid:
            for key in keys:
                new_trial_infos[key].append(trial_infos[key][i_trial])

    # # For each trial and each neuron, convert the spike times into a rate
    # n_time = 100
    # Rates = np.zeros((n_unit, n_trial, n_time))  # (-2.5s, 3.5s)
    # mins = list()
    # maxs = list()
    # for i_trial in range(n_trial):
    #     for i_neuron in range(n_unit):
    #         # Convert the spikes into a rate
    #         spikes_tmp = spikes[i_trial, i_neuron]
    #         try:
    #             mins.append(np.min(spikes_tmp))
    #             maxs.append(np.max(spikes_tmp))
    #         except ValueError:
    #             pass
    #         # rate = convert_to_rate(spikes_tmp)
    #         # Rates[i_trial, i_neuron] = rate
    i_trial= 2
    i_unit = 0
    spikes_unit = spikes[i_trial, i_unit]
    # trial_info = trial_infos[i_trial]
    # unit_info = unit_infos[i_unit]