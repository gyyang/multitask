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