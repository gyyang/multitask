"""Different training experiments."""

from __future__ import division

import os
from collections import OrderedDict
import numpy as np

import tools
import train
import variance
import clustering
import data_analysis
import taskset

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


def train_mante(seed=0, model_dir='train_mante'):
    """Training of only the Mante task."""
    hp = {'target_perf': 0.9}
    model_dir = os.path.join(DATAPATH, model_dir, str(seed))
    train.train(model_dir, hp=hp, ruleset='mante', seed=seed)


def mante_tanh(seed=0, model_dir='mante_tanh'):
    """Training of only the Mante task."""
    hp = {'activation': 'tanh',
               'target_perf': 0.9}
    model_dir = os.path.join(DATAPATH, model_dir, str(seed))
    train.train(model_dir, hp=hp, ruleset='mante', seed=seed)
    # Analyses
    variance.compute_variance(model_dir)

    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)
    data_analysis.compute_var_all(model_dir)


def train_all(seed=0, model_dir='train_all'):
    """Training of all tasks."""
    model_dir = os.path.join(DATAPATH, model_dir, str(seed))
    hp = {'activation': 'softplus'}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    # train.train(model_dir, hp=hp, ruleset='all',
    #             rule_prob_map=rule_prob_map, seed=seed)
    # Analyses
    variance.compute_variance(model_dir)
    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)
    data_analysis.compute_var_all(model_dir)
    
    setups = [1, 2, 3]
    for setup in setups:
        taskset.compute_taskspace(model_dir, setup,
                                  restore=False,
                                  representation='rate')
        taskset.compute_replacerule_performance(model_dir, setup, False)


def train_seq(i):
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['c_intsyn'] = [0, 1.0]

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    # Set up new hyperparameter
    hp = dict()
    for key, index in zip(keys, indices):
        hp[key] = hp_ranges[key][index]
    hp['learning_rate'] = 0.001
    hp['w_rec_init'] = 'randortho'
    hp['easy_task'] = True
    hp['activation'] = 'relu'
    hp['ksi_intsyn'] = 0.01
    hp['max_steps'] = 4e5

    model_dir = os.path.join(DATAPATH, 'seq', str(i))
    rule_trains = [['fdgo'], ['delaygo'], ['dm1', 'dm2'], ['multidm'],
                   ['contextdm1', 'contextdm2']]
    train.train_sequential(
        model_dir,
        rule_trains,
        hp=hp,
        max_steps=hp['max_steps'],
        display_step=500,
        ruleset='all',
        seed=i // n_max,
    )


def train_vary_hp_seq(i):
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus', 'relu']
    hp_ranges['w_rec_init'] = ['randortho']
    hp_ranges['c_intsyn'] = [0, 0.1, 1.0, 10.]
    hp_ranges['ksi_intsyn'] = [0.001, 0.01, 0.1]
    hp_ranges['max_steps'] = [1e5, 2e5, 4e5]

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    # Set up new hyperparameter
    hp = dict()
    for key, index in zip(keys, indices):
        hp[key] = hp_ranges[key][index]
    hp['learning_rate'] = 0.001
    hp['w_rec_init'] = 'randortho'
    hp['easy_task'] = True

    model_dir = os.path.join(DATAPATH, 'seq_varyhp', str(i))
    rule_trains = [['fdgo'], ['delaygo'], ['dm1', 'dm2'], ['multidm'],
                   ['contextdm1', 'contextdm2']]
    train.train_sequential(
        model_dir,
        rule_trains,
        hp=hp,
        max_steps=hp['max_steps'],
        display_step=500,
        ruleset='all',
        seed=i // n_max,
    )


def train_vary_hp(i):
    """Vary the hyperparameters.

    This experiment loops over a set of hyperparameters.

    Args:
        i: int, the index of the hyperparameters list
    """
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    # hp_ranges['activation'] = ['softplus', 'relu', 'tanh', 'retanh']
    # hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU']
    # hp_ranges['w_rec_init'] = ['diag', 'randortho']
    hp_ranges['activation'] = ['softplus']
    hp_ranges['rnn_type'] = ['LeakyRNN']
    hp_ranges['w_rec_init'] = ['randortho']
    hp_ranges['l1_h'] = [0, 1e-9, 1e-8, 1e-7, 1e-6]  # TODO(gryang): Change this?
    hp_ranges['l2_h'] = [0]
    hp_ranges['l1_weight'] = [0, 1e-7, 1e-6, 1e-5]
    # TODO(gryang): add the level of overtraining

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    # Set up new hyperparameter
    hp = dict()
    for key, index in zip(keys, indices):
        hp[key] = hp_ranges[key][index]

    model_dir = os.path.join(DATAPATH, 'varyhp_reg2', str(i))
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(model_dir, hp, ruleset='all',
                rule_prob_map=rule_prob_map, seed=i // n_max)

    # Analyses
    variance.compute_variance(model_dir)
    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)
    data_analysis.compute_var_all(model_dir)


def _base_vary_hp_mante(i, hp_ranges, base_name):
    """Vary hyperparameters for mante tasks."""
    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    # Set up new hyperparameter
    hp = dict()
    for key, index in zip(keys, indices):
        hp[key] = hp_ranges[key][index]

    model_dir = os.path.join(DATAPATH, base_name, str(i))
    train.train(model_dir, hp, ruleset='mante',
                max_steps=1e7, seed=i // n_max)

    # Analyses
    variance.compute_variance(model_dir)
    
    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)
    data_analysis.compute_var_all(model_dir)


def vary_l2_init_mante(i):
    """Vary the hyperparameters and train on Mante tasks only.

    This experiment loops over a set of hyperparameters.

    Args:
        i: int, the index of the hyperparameters list
    """
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus']
    hp_ranges['rnn_type'] = ['LeakyRNN']
    hp_ranges['w_rec_init'] = ['randortho']
    hp_ranges['l2_weight_init'] = [0, 1e-4, 2*1e-4, 4*1e-4, 8*1e-4, 1.6*1e-3]
    hp_ranges['target_perf'] = [0.9]

    _base_vary_hp_mante(i, hp_ranges, base_name='vary_l2init_mante')


def vary_l2_weight_mante(i):
    """Vary the hyperparameters and train on Mante tasks only.

    This experiment loops over a set of hyperparameters.

    Args:
        i: int, the index of the hyperparameters list
    """
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus']
    hp_ranges['rnn_type'] = ['LeakyRNN']
    hp_ranges['w_rec_init'] = ['randortho']
    hp_ranges['l2_weight'] = [0, 1e-4, 2*1e-4, 4*1e-4, 8*1e-4, 1.6*1e-3]
    hp_ranges['target_perf'] = [0.9]

    _base_vary_hp_mante(i, hp_ranges, base_name='vary_l2weight_mante')


def vary_p_weight_train_mante(i):
    """Vary the hyperparameters and train on Mante tasks only.

    This experiment loops over a set of hyperparameters.

    Args:
        i: int, the index of the hyperparameters list
    """
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus']
    hp_ranges['rnn_type'] = ['LeakyRNN']
    hp_ranges['w_rec_init'] = ['randortho']
    # hp_ranges['p_weight_train'] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hp_ranges['p_weight_train'] = [0.05, 0.075]
    hp_ranges['target_perf'] = [0.9]

    _base_vary_hp_mante(i, hp_ranges, base_name='vary_pweighttrain_mante')


if __name__ == '__main__':
    """ 
    train_mante()
    # train_all()
    # for i in range(10):
    #     train_vary_hp(i)
    """ 

    #train_all()
    for i in np.arange(0,3):
        #train_vary_hp(i)
        train_all(i)

