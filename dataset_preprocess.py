"""Dataset preprocessing.

Standardize several datasets into the same format for task variance analysis.
"""

from __future__ import division

import os
import numpy as np
from scipy.io import loadmat

DATASETPATH = './datasets'


def load_mante_data(smooth=True):
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
    n_unit = len(data)
    n_cond = 72 # condition
    n_time = data[0].response.shape[1]
    return data


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


def get_mante_data():
    data = load_mante_data()

    n_unit = len(data)

    rate1s = list()
    rate2s = list()
    random_shuffle = True
    for i_unit in range(n_unit):
        # Get trial-averaged condition-based responses (n_condition, n_time)
        rate1s.append(get_trial_avg_rate(data[i_unit], context=1,
                                         random_shuffle=random_shuffle))
        rate2s.append(get_trial_avg_rate(data[i_unit], context=-1,
                                         random_shuffle=random_shuffle))
    # (n_unit, n_condition, n_time)
    rate1s, rate2s = np.array(rate1s), np.array(rate2s)
    return rate1s, rate2s


if __name__ == '__main__':
    rate1s, rate2s = get_mante_data()
