"""Different training experiments."""

from __future__ import division

import os
import numpy as np

import train
import variance

# TODO: make this flexible
DATAPATH = os.path.join(os.getcwd(), 'data')

def analysis(run_analysis, model_dir):
    # Run a set of standard analysis
    if run_analysis is None:
        run_analysis = list()

    if 'var' in run_analysis:
        # Compute variance
        from variance import compute_variance
        compute_variance(model_dir)
        compute_variance(model_dir, random_rotation=True)
        print('Computed variance')

    if 'psy' in run_analysis:
        # Psychometric analysis
        import performance
        for rule in ['dm1', 'contextdm1', 'multidm']:
            performance.compute_choicefamily_varytime(model_dir, rule)

    # if 'compare' in run_analysis:
    #     # Compute similarity with data and store results
    #     from contextdm_analysis import run_score
    #     log['score_train'], log['score_test'] = run_score(model_dir)
    #     print('Data matching score : {:0.3f}'.format(log['score_test'].mean()))
    #
    #     with open(os.path.join('data', 'log_'+model_dir+'.pkl'), 'wb') as f:
    #         pickle.dump(log, f)

    if 'taskset' in run_analysis:
        # Run analysis for task representation
        from taskset import compute_taskspace, compute_replacerule_performance
        for setup in [1, 2]:
            compute_taskspace(model_dir, setup, restore=False)
            compute_replacerule_performance(model_dir, setup, restore=False)


def train_mante(seed=0, train_dir='train_mante'):
    """Training of only the Mante task."""
    hparams = {'target_perf': 0.9}
    train_dir = os.path.join(DATAPATH, train_dir, str(seed))
    train.train(train_dir, hparams=hparams, ruleset='mante', seed=seed)


def train_all(seed=0, train_dir='train_all'):
    """Training of all tasks."""
    train_dir = os.path.join(DATAPATH, train_dir, str(seed))
    hparams = {'activation': 'softplus'}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(train_dir, hparams=hparams, ruleset='all',
                rule_prob_map=rule_prob_map, seed=seed)


def train_vary_hparams(i):
    """Vary the hyperparameters.

    This experiment loops over a set of hyperparameters.

    Args:
        i: int, the index of the hyperparameters list
    """
    # Ranges of hyperparameters to loop over
    hp_ranges = dict()
    hp_ranges['activation'] = ['softplus','relu', 'tanh', 'retanh', 'power']
    hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU']
    hp_ranges['w_rec_init'] = ['diag', 'randortho']
    hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_h'] = [0]
    hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    indices = np.unravel_index(i, dims=dims)

    # Set up new hyperparameter
    hparams = dict()
    hparams['target_perf'] = 0.9#0.8
    for key, index in zip(keys, indices):
        hparams[key] = hp_ranges[key][index]

    train_dir = os.path.join(DATAPATH, 'varyhparams', str(i))
    #train.train(train_dir, hparams, ruleset='mante', max_steps=1e7)
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(train_dir, hparams, ruleset='all', rule_prob_map=rule_prob_map, max_steps=1e8)

    variance.compute_variance(train_dir)


if __name__ == '__main__':
    """ 
    train_mante()
    # train_all()
    # for i in range(10):
    #     train_vary_hparams(i)
    """ 

    #train_all()
    for i in np.arange(0,3):
        #train_vary_hparams(i)
        train_all(i)

