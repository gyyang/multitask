"""Different training experiments."""

from __future__ import division

import os
import numpy as np
import train

# TODO: make this flexible
DATAPATH = os.path.join(os.getcwd(), 'data')

# def analysis(run_analysis):
#     # Run a set of standard analysis
#     if run_analysis is None:
#         run_analysis = list()
#
#     if 'var' in run_analysis:
#         # Compute variance
#         from variance import compute_variance
#         compute_variance(save_name)
#         compute_variance(save_name, random_rotation=True)
#         print('Computed variance')
#
#     if 'psy' in run_analysis:
#         # Psychometric analysis
#         import performance
#         for rule in ['dm1', 'contextdm1', 'multidm']:
#             if rule in rule_tests:
#                 performance.compute_choicefamily_varytime(save_name, rule)
#
#     if 'compare' in run_analysis:
#         # Compute similarity with data and store results
#         from contextdm_analysis import run_score
#         log['score_train'], log['score_test'] = run_score(save_name)
#         print('Data matching score : {:0.3f}'.format(log['score_test'].mean()))
#
#         with open(os.path.join('data', 'log_'+save_name+'.pkl'), 'wb') as f:
#             pickle.dump(log, f)
#
#     if 'taskset' in run_analysis:
#         # Run analysis for task representation
#         from taskset import compute_taskspace, compute_replacerule_performance
#         for setup in [1, 2]:
#             compute_taskspace(save_name, setup, restore=False)
#             compute_replacerule_performance(save_name, setup, restore=False)


def train_mante_local():
    """Local training of only the Mante task."""
    hparams = {'learning_rate': 0.001, 'in_type': 'multi'}
    train_dir = os.path.join(DATAPATH, 'debug')
    train.train(train_dir, hparams=hparams, ruleset='mante', seed=0)


def train_all_local():
    """Local training of all tasks."""
    train_dir = os.path.join(DATAPATH, 'debug')
    train.train(train_dir, ruleset='all', seed=0)


def train_vary_hparams(i):
    """Vary the hyperparameters.

    This experiment loops over a set of hyperparameters.

    Args:
        i: int, the index of the hyperparameters list
    """
    # Ranges of hyperparameters to loop over
    hp_ranges = dict()
    hp_ranges['activation'] = ['relu', 'tanh']
    hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU']
    hp_ranges['w_rec_init'] = ['diag', 'randortho']
    hp_ranges['l1_h'] = [0, 1e-4]
    hp_ranges['l2_h'] = [0, 1e-4]
    hp_ranges['l1_weight'] = [0, 1e-4]

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    indices = np.unravel_index(i, dims=dims)

    # Set up new hyperparameter
    hparams = dict()
    for key, index in zip(keys, indices):
        hparams[key] = hp_ranges[key][index]

    train_dir = os.path.join(DATAPATH, 'debug', str(i))
    train.train(train_dir, hparams, ruleset='mante', max_steps=2000)


if __name__ == '__main__':
    # train_mante_local()
    # train_all_local()
    for i in range(10):
        train_vary_hparams(i)

