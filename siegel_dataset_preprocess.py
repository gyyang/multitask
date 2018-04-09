"""Siegel dataset preprocessing.

Standardize the Siegel dataset
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

DATASETPATH = './datasets/siegel_dataset'


def _load_spikes(fname):
    """Load spiking data from Siegel dataset."""
    fname_full = os.path.join(DATASETPATH, 'sorted', fname)

    # Analyze a single file
    mat_dict = loadmat(fname_full, squeeze_me=True, struct_as_record=False)

    # Get spike times
    # spikes is an array (trials, neurons) of 1-D array
    spikes = mat_dict['spikeTimes']
    return spikes


def _load_tables(fname, table_name):
    """Load tables from Siegel dataset and convert to dictionary.

    Args:
        fname: str, file name
        table_name: str, can be 'trialinfo', 'unitinfo', and 'electrodeinfo'

    Returns:
        table_dict: dictionary, for each (key, val) pair, val is an array
    """
    fname_full = os.path.join(DATASETPATH, table_name, fname[:6] + '.csv')

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


def _get_valid_trials(trial_infos):
    """Get valid trials.

    Args:
        trial_infos: dict of arrays.

    Returns:
        new_trial_infos: dict of arrays, only contain valid trials
        valid_trials: list of bools, indicating the valid trials
    """
    new_trial_infos = defaultdict(list)
    keys = trial_infos.keys()
    n_trial = len(trial_infos[keys[0]])
    valid_trials = list()
    for i_trial in range(n_trial):
        if (trial_infos['badTrials'][i_trial] == 1 or
            math.isnan(trial_infos['responseTime'][i_trial]) or
            trial_infos['responseTime'][i_trial] < 0.2
        ):
            valid = False
        else:
            valid = True
        valid_trials.append(valid)
        if valid:
            for key in keys:
                new_trial_infos[key].append(trial_infos[key][i_trial])
    return new_trial_infos, valid_trials


def _expand_task_var(task_var):
    """Little helper function that calculate a few more things."""
    task_var['context'] = list()
    for r in task_var['rule']:
        if r == 'dir':
            task_var['context'].append(+1)
        else:
            task_var['context'].append(-1)
    # TODO(gryang): Take care of boundary case
    task_var['stim_dir_sign'] = (np.array(task_var['direction']>0)).astype(int)*2-1
    task_var['stim_col2dir_sign'] = (np.array(task_var['color']>90)).astype(int)*2-1
    return task_var


def load_data():
    datasetpath = os.path.join(DATASETPATH, 'sorted')

    files = os.listdir(datasetpath)
    files = [f for f in files if '1' in f]

    f = files[0]

    trial_infos = _load_tables(f, 'trialinfo')
    trial_infos, valid_trials = _get_valid_trials(trial_infos)
    task_var = _expand_task_var(trial_infos)

    unit_infos = _load_tables(f, 'unitinfo')
    electrode_infos = _load_tables(f, 'electrodeinfo')

    spikes = _load_spikes(f)
    spikes = spikes[valid_trials, :]

    n_trial, n_unit = spikes.shape

    assert len(trial_infos.values()[0]) == n_trial
    assert len(unit_infos.values()[0]) == n_unit

    bin_size = 0.05  # unit: second
    bins = np.arange(-0.05, 0.2, bin_size)
    n_time = len(bins) - 1

    data = list()
    for i_unit in range(n_unit):
        rates = np.zeros((n_trial, n_time))
        for i_trial in range(n_trial):
            spikes_unit = spikes[i_trial, i_unit]
            # Compute PSTH
            hist, bin_edges = np.histogram(spikes_unit, bins=bins)
            rates[i_trial, :] = hist / bin_size
        unit_dict = {
            'task_var': task_var,  # TODO(gryang): change keys
            'rate': rates
        }
        data.append(unit_dict)
    return data

# def _spike_to_rate(spikes_unit, times):
#     """Convert spikes to rate.
#
#     Args:
#         spikes_unit: list of float, a list of spike times
#         times: list of float, a list of time points, default unit second
#
#     Returns:
#         rates: list of float, a list of rate, default unit spike/second, Hz
#             rates will be the same size as times
#     """


if __name__ == '__main__':
    # rate1s, rate2s = get_mante_data()
    data = load_data()




