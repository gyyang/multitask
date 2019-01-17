"""Different training experiments."""

from __future__ import division

import os
from collections import OrderedDict
import numpy as np

import tools
import train
from analysis import variance
from analysis import clustering
from analysis import data_analysis
from analysis import performance
from analysis import taskset

# TODO: make this flexible
DATAPATH = os.path.join(os.getcwd(), 'data')


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


def train_all(seed=0, root_dir='train_all'):
    """Training of all tasks."""
    model_dir = os.path.join(DATAPATH, root_dir, str(seed))
    hp = {'activation': 'softplus', 'w_rec_init': 'diag'}  # TODO: change the default back to diag
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(model_dir, hp=hp, ruleset='all',
                rule_prob_map=rule_prob_map, seed=seed)
    train_all_analysis(seed=seed, root_dir=root_dir)


def debug_train_all():
    root_dir = 'debug_train_all'
    seed = 0
    model_dir = os.path.join(DATAPATH, root_dir, str(seed))
    hp = {'activation': 'softplus', 'w_rec_init': 'diag'}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(model_dir, hp=hp, ruleset='all',
                rule_prob_map=rule_prob_map, seed=seed, max_steps=1e3)
    train_all_analysis(seed=seed, root_dir=root_dir)


def train_all_analysis(seed=0, root_dir='train_all'):
    model_dir = os.path.join(DATAPATH, root_dir, str(seed))
    # Analyses
    variance.compute_variance(model_dir)
    variance.compute_variance(model_dir, random_rotation=True)
    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)
    data_analysis.compute_var_all(model_dir)

    for rule in ['dm1', 'contextdm1', 'multidm']:
        performance.compute_choicefamily_varytime(model_dir, rule)

    setups = [1, 2, 3]
    for setup in setups:
        taskset.compute_taskspace(model_dir, setup,
                                  restore=False,
                                  representation='rate')
        taskset.compute_replacerule_performance(model_dir, setup, False)


def train_all_tanhgru(seed=0, model_dir='tanhgru'):
    """Training of all tasks with Tanh GRUs."""
    model_dir = os.path.join(DATAPATH, model_dir, str(seed))
    hp = {'activation': 'tanh',
          'rnn_type': 'LeakyGRU'}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(model_dir, hp=hp, ruleset='all',
                rule_prob_map=rule_prob_map, seed=seed)
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


def train_all_mixrule(seed=0, root_dir='mixrule'):
    """Training of all tasks."""
    model_dir = os.path.join(DATAPATH, root_dir, str(seed))
    hp = {'activation': 'relu', 'w_rec_init': 'diag',
          'use_separate_input': True, 'mix_rule': True}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(model_dir, hp=hp, ruleset='all',
                rule_prob_map=rule_prob_map, seed=seed)

    # Analyses
    variance.compute_variance(model_dir)
    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)

    setups = [1, 2, 3]
    for setup in setups:
        taskset.compute_taskspace(model_dir, setup,
                                  restore=False,
                                  representation='rate')
        taskset.compute_replacerule_performance(model_dir, setup, False)


def train_all_mixrule_softplus(seed=0, root_dir='mixrule_softplus'):
    """Training of all tasks."""
    model_dir = os.path.join(DATAPATH, root_dir, str(seed))
    hp = {'activation': 'softplus', 'w_rec_init': 'diag',
          'use_separate_input': True, 'mix_rule': True}
    rule_prob_map = {'contextdm1': 5, 'contextdm2': 5}
    train.train(model_dir, hp=hp, ruleset='all',
                rule_prob_map=rule_prob_map, seed=seed)

    # Analyses
    variance.compute_variance(model_dir)
    log = tools.load_log(model_dir)
    analysis = clustering.Analysis(model_dir, 'rule')
    log['n_cluster'] = analysis.n_cluster
    tools.save_log(log)

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


def pretrain(setup, seed):
    """Get pre-trained networks."""
    hp = dict()
    hp['learning_rate'] = 0.001
    hp['w_rec_init'] = 'diag'
    hp['easy_task'] = False
    hp['activation'] = 'relu'
    hp['max_steps'] = 2*1e6
    hp['l1_h'] = 1e-8
    hp['target_perf'] = 0.97
    hp['n_rnn'] = 128
    hp['use_separate_input'] = True

    model_dir = os.path.join(DATAPATH, 'pretrain', 'setup'+str(setup), str(seed))
    if setup == 0:
        rule_trains = ['contextdm1', 'contextdm2', 'contextdelaydm2']
    elif setup == 1:
        rule_trains = ['fdgo', 'fdanti', 'delaygo']
    else:
        raise ValueError

    train.train(model_dir,
          hp=hp,
          max_steps=hp['max_steps'],
          display_step=500,
          ruleset='all',
          rule_trains=rule_trains,
          rule_prob_map=None,
          seed=seed,
          )


def posttrain(pretrain_setup, posttrain_setup, trainables, seed):
    """Training based on pre-trained networks."""
    hp = {'n_rnn': 128,
          'l1_h': 1e-8,
          'target_perf': 0.97,
          'activation': 'relu',
          'max_steps': 1e6,
          'use_separate_input': True}

    if posttrain_setup == 0:
        rule_trains = ['contextdelaydm1']
    elif posttrain_setup == 1:
        rule_trains = ['delayanti']
    else:
        raise ValueError

    if trainables == 0:
        hp['trainables'] = 'all'
    elif trainables == 1:
        hp['trainables'] = 'rule'
    else:
        raise ValueError

    name = (str(pretrain_setup) + '_' + str(posttrain_setup) +
            '_' + str(trainables) + '_' + str(seed))
    model_dir = os.path.join(DATAPATH, 'posttrain', name)
    load_dir = os.path.join(DATAPATH, 'pretrain',
                            'setup' + str(pretrain_setup), str(seed))
    hp['load_dir'] = load_dir
    hp['pretrain_setup'] = pretrain_setup
    hp['posttrain_setup'] = posttrain_setup
    train.train(model_dir,
          hp=hp,
          max_steps=hp['max_steps'],
          display_step=50,
          ruleset='all',
          rule_trains=rule_trains,
          seed=seed,
          load_dir=load_dir,
          trainables=hp['trainables'],
          )


if __name__ == '__main__':
    debug_train_all()
