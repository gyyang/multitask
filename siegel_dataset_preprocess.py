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


if __name__ == '__main__':
    # rate1s, rate2s = get_mante_data()

    datasetpath = os.path.join(DATASETPATH, 'sorted')

    files = os.listdir(datasetpath)
    files = [f for f in files if '1' in f]

    f = files[0]

    trial_infos = _load_tables(f, 'trialinfo')
    unit_infos = _load_tables(f, 'unitinfo')
    electrode_infos = _load_tables(f, 'electrodeinfo')

    spikes = _load_spikes(f)
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
    i_trial = 2
    i_unit = 0
    spikes_unit = spikes[i_trial, i_unit]
    # trial_info = trial_infos[i_trial]
    # unit_info = unit_infos[i_unit]