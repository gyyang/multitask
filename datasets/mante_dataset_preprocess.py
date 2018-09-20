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


def _expand_task_var(task_var):
    """Little helper function that calculate a few more things."""
    task_var['stim_dir_sign'] = (task_var['stim_dir']>0).astype(int)*2-1
    task_var['stim_col2dir_sign'] = (task_var['stim_col2dir']>0).astype(int)*2-1
    return task_var


def load_data(smooth=True, single_units=False, animal='ar'):
    """Load Mante data into raw format.

    Args:
        smooth: bool, whether to load smoothed data
        single_units: bool, if True, only analyze single units
        animal: str, 'ar' or 'fe'

    Returns:
        data: standard format, list of dict of arrays/dict
            list is over neurons
            dict is for response array and task variable dict
            response array has shape (n_trial, n_time)
    """
    if smooth:
        fname = 'dataTsmooth_' + animal + '.mat'
    else:
        fname = 'dataT_' + animal + '.mat'

    fname = os.path.join(DATASETPATH, fname)

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)

    dataT = mat_dict['dataT'].__dict__ # as dictionary

    data = dataT['unit']
    # time = dataT['time']

    if single_units:
        single_units = get_single_units(animal)
        ind_single_units = np.where(single_units)[0]
        data = [data[i] for i in ind_single_units]

    # Convert to standard format
    new_data = list()
    n_unit = len(data)
    for i in range(n_unit):
        task_var = data[i].task_variable.__dict__
        task_var = _expand_task_var(task_var)
        unit_dict = {
            'task_var': task_var,  # turn into dictionary
            'rate': data[i].response  # (n_trial, n_time)
        }
        new_data.append(unit_dict)

    return new_data


def get_single_units(animal):
    # get single units
    fname = os.path.join('datasets', 'mante_dataset',
                         'metadata_'+animal+'.mat')
    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)
    metadata = mat_dict['metadata'].unit # as dictionary
    discriminable = np.array([(m.unitInfo.discriminability in [3,4]) for m in metadata])
    single_units = np.array([m.unitInfo.type=='s' for m in metadata])
    single_units = np.logical_and(single_units, discriminable)
    if animal == 'ar':
        assert np.sum(single_units)==181
    else:
        assert np.sum(single_units)==207
    return single_units


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