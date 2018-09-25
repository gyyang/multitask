"""Analyze the results after varying hyperparameters."""

from __future__ import division

from collections import defaultdict
from collections import OrderedDict
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tools

mpl.rcParams.update({'font.size': 7})

DATAPATH = os.path.join(os.getcwd(), 'data', 'posttrain')
FIGPATH = os.path.join(os.getcwd(), 'figure')


def get_avg_performance(model_dirs, rule):
    """Get average performance across trials for model_dirs."""
    perfs = defaultdict(list)
    # TODO: If a network reached the top, then perf replaced by the last one
    for model_dir in model_dirs:
        log = tools.load_log(model_dir)
        for t, perf in zip(log['trials'], log['perf_'+rule]):
            perfs[t].append(perf)

    # average performances
    trials = list(perfs.keys())
    trials = np.sort(trials)
    avg_perfs = [np.mean(perfs[t]) for t in trials]
    return avg_perfs, trials


def plot_posttrain_performance(posttrain_setup, trainables):
    from task import rule_name
    hp_target = {'posttrain_setup': posttrain_setup,
                 'trainables': trainables}
    hp_target['pretrain_setup'] = 0
    model_dirs0 = tools.find_all_models(DATAPATH, hp_target)
    hp_target['pretrain_setup'] = 1
    model_dirs1 = tools.find_all_models(DATAPATH, hp_target)

    fs = 7
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    color0, color1 = 'xkcd:blue', 'xkcd:red'
    hp = tools.load_hp(model_dirs0[0])
    rule = hp['rule_trains'][0]  # depends on posttrain setup
    for model_dir in model_dirs0:
        log = tools.load_log(model_dir)
        ax.plot(np.array(log['trials']) / 1000., log['perf_' + rule], color=color0, alpha=0.1)
    avg_perfs, trials = get_avg_performance(model_dirs0, rule)
    l0 = ax.plot(trials / 1000., avg_perfs, color=color0, label='0')

    for model_dir in model_dirs1:
        log = tools.load_log(model_dir)
        ax.plot(np.array(log['trials']) / 1000., log['perf_' + rule], color=color1, alpha=0.1)
    avg_perfs, trials = get_avg_performance(model_dirs1, rule)
    l1 = ax.plot(trials / 1000., avg_perfs, color=color1, label='1')
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total trials (1,000)', fontsize=fs, labelpad=2)
    ax.set_yticks([0, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    lg = ax.legend(title='Pretrained set', ncol=2, loc=4,
                    frameon=False)

    plt.ylabel('Perf. of ' + rule_name[rule])
    # plt.title('Training ' + hp_target['trainables'])
    plt.savefig('figure/Posttrain_post{:d}train{:s}.pdf'.format(
        posttrain_setup, trainables), transparent=True)
    # plt.show()


if __name__ == '__main__':
    pass

