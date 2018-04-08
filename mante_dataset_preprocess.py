"""Mante dataset preprocessing.

Standardize the Mante dataset.
"""

from __future__ import division

import os
import csv
from collections import defaultdict
import math
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

DATASETPATH = './datasets/mante_dataset'


def load_mante_data(smooth=True):
    """Load Mante data into raw format.

    Args:
        smooth: bool, whether to load smoothed data

    Returns:
        data: a list of mat_struct, storing info for each unit
    """
    if smooth:
        fname = 'dataT_smooth.mat'
    else:
        fname = 'dataT.mat'

    fname = os.path.join(DATASETPATH, fname)

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


if __name__ == '__main__':
    rate1s, rate2s = get_mante_data()