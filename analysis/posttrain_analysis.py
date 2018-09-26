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
    """Get average performance across trials for model_dirs.

    Some networks converge earlier than others. For those converged early,
    choose the last performance for later performance
    """
    perfs = defaultdict(list)

    trials = []
    for model_dir in model_dirs:
        log = tools.load_log(model_dir)
        trials += list(log['trials'])
    trials = np.sort(np.unique(trials))

    for model_dir in model_dirs:
        log = tools.load_log(model_dir)
        for t in trials:
            if t in log['trials']:
                ind = log['trials'].index(t)
            else:
                ind = -1
            perfs[t].append(log['perf_' + rule][ind])
        # for t, perf in zip(log['trials'], log['perf_'+rule]):
        #     perfs[t].append(perf)

    # average performances
    trials = list(perfs.keys())
    trials = np.sort(trials)
    avg_perfs = [np.mean(perfs[t]) for t in trials]
    return avg_perfs, trials


def plot_posttrain_performance(posttrain_setup, trainables):
    from task import rule_name
    hp_target = {'posttrain_setup': posttrain_setup,
                 'trainables': trainables}
    fs = 7
    fig = plt.figure(figsize=(1.5, 1.2))
    ax = fig.add_axes([0.25, 0.3, 0.7, 0.65])

    colors = ['xkcd:blue', 'xkcd:red']
    for pretrain_setup in [1, 0]:
        c = colors[pretrain_setup]
        l = ['B', 'A'][pretrain_setup]
        hp_target['pretrain_setup'] = pretrain_setup
        model_dirs = tools.find_all_models(DATAPATH, hp_target)
        hp = tools.load_hp(model_dirs[0])
        rule = hp['rule_trains'][0]  # depends on posttrain setup
        for model_dir in model_dirs:
            log = tools.load_log(model_dir)
            ax.plot(np.array(log['trials']) / 1000.,
                    log['perf_' + rule], color=c, alpha=0.1)
        avg_perfs, trials = get_avg_performance(model_dirs, rule)
        l0 = ax.plot(trials / 1000., avg_perfs, color=c, label=l)

    ax.set_ylim([0, 1])
    ax.set_xlabel('Total trials (1,000)', fontsize=fs, labelpad=2)
    ax.set_yticks([0, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # lg = ax.legend(title='Pretrained set', ncol=2, loc=4,
    #                 frameon=False)

    plt.ylabel('Perf. of ' + rule_name[rule])
    # plt.title('Training ' + hp_target['trainables'])
    plt.savefig('figure/Posttrain_post{:d}train{:s}.pdf'.format(
        posttrain_setup, trainables), transparent=True)
    # plt.show()


if __name__ == '__main__':
    pass

