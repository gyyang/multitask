"""
Main file for generating results in the paper:
Clustering and compositionality of task representations
in a neural network trained to perform many cognitive tasks
Yang GR et al. 2017 BioRxiv
"""
from __future__ import absolute_import

import tools
import performance
import standard_analysis
import antitask
import clustering
import variance
import taskset
import varyhparams


def obsolete_cont_train(c, ksi, seed, save_name, seq=True):
    """Sequantial training setup for paper

    Args:
        c, ksi : parameters for continual learning
        seed: int, random seed.
        save_name: string, name of file to be saved.
        seq: bool, sequential or not.
    """
    from train import train
    param_intsyn = (c, ksi)

    nunit = 256
    ruleset = 'all'

    if seq:
        # Sequential training
        rule_trains = [
                'fdgo', 'delaygo', 'dm1', 'dm2', ('contextdm1', 'contextdm2'),
                'multidm', 'delaydm1', 'delaydm2',
                ('contextdelaydm1', 'contextdelaydm2'), 'multidelaydm']
    else:
        rule_trains = [(
                'fdgo', 'delaygo', 'dm1', 'dm2', 'contextdm1', 'contextdm2',
                'multidm', 'delaydm1', 'delaydm2', 'contextdelaydm1',
                'contextdelaydm2', 'multidelaydm')]

    rule_tests = [
            'fdgo', 'delaygo', 'dm1', 'dm2', 'contextdm1', 'contextdm2',
            'multidm', 'delaydm1', 'delaydm2',
            'contextdelaydm1', 'contextdelaydm2', 'multidelaydm']

    learning_rate = 0.001  # learning is much better with smaller learning rate
    activation = 'relu'  # softplus has lots of difficulty learning context-dms
    easy_task = True   # Network has difficulty learning harder version

    train(save_name,
          ruleset=ruleset,
          n_hidden=nunit,
          learning_rate=learning_rate,
          target_perf=None,
          seed=seed,
          activation=activation,
          rnn_type='LeakyRNN',
          l1_h=0.0001,
          training_iters=100000,
          display_step=500,
          param_intsyn=param_intsyn,
          rule_trains=rule_trains,
          rule_tests=rule_tests,
          run_analysis=['var'],
          easy_task=easy_task)


# Directories of the models and the sample model
# Change these to your directories
root_dir = './data/train_all'
model_dir = root_dir + '/0'

# =============================================================================
# root_dir = './data/varyhparams'
# hp_target = {'activation': 'softplus',
#              'rnn_type': 'LeakyGRU',
#              'w_rec_init': 'randortho',
#              'l1_h': 0,
#              'l1_weight': 0}
# model_dir, _ = tools.find_model(root_dir, hp_target)
# =============================================================================

# =============================================================================
# root_dir = './data/varyhparams'
# hp_target = {'activation': 'tanh',
#              'rnn_type': 'LeakyRNN',
#              'w_rec_init': 'diag',
#              'l1_h': 0}
# root_dir, _ = tools.find_all_models(root_dir, hp_target)
# =============================================================================

# root_dir = './data/mantetemp'
# root_dir = model_dir

# variance.compute_variance(root_dir)

# Performance Analysis-----------------------------------------------------
# standard_analysis.schematic_plot(model_dir=model_dir) # Generate schematic
# performance.plot_performanceprogress(model_dir) # Plot performance during training
# performance.psychometric_choice(model_dir) # Psychometric for dm
# performance.psychometric_choiceattend(model_dir, no_ylabel=True)
# performance.psychometric_choiceint(model_dir, no_ylabel=True)

# TODO(gryang): the following remains to be fixed
# model_dirs = t
# tools.valid_model_dirs('*256paper')
# for model_dir in model_dirs[3:4]:
#     for rule in ['dm1', 'contextdm1', 'multidm']:
#         # performance.compute_choicefamily_varytime(model_dir, rule)
#         performance.plot_choicefamily_varytime(model_dir, rule)
# performance.psychometric_delaychoice_varytime(model_dir, 'delaydm1')

# Analysis of Anti tasks---------------------------------------------------
# ATA = antitask.Analysis(model_dir)
# ATA.plot_example_unit()
# ATA.plot_lesions()
# ATA.plot_inout_connections()
# ATA.plot_rec_connections()
# ATA.plot_rule_connections()

# Clustering Analysis------------------------------------------------------

CA = clustering.Analysis(model_dir, data_type='rule')
# CA.plot_example_unit()
# CA.plot_cluster_score()
# CA.plot_cluster_score(save_name=hp_target['activation'])
# CA.plot_variance()
# CA.plot_2Dvisualization('PCA')
# CA.plot_2Dvisualization('MDS')
# CA.plot_2Dvisualization('tSNE')
# CA.plot_lesions()
# CA.plot_connectivity_byclusters()


# CA = clustering.Analysis(model_dir, data_type='epoch')
# CA.plot_variance()
# CA.plot_2Dvisualization()

# FTV Analysis-------------------------------------------------------------
# variance.plot_hist_varprop_selection(root_dir)
# variance.plot_hist_varprop_all(root_dir)

# Varying hyperparameter analysis------------------------------------------
# varyhparams.plot_n_clusters()
# varyhparams.plot_hist_varprop_tanh()

# Task Representation------------------------------------------------------
# tsa = taskset.TaskSetAnalysis(model_dir)
# tsa.compute_and_plot_taskspace(
#         epochs=['stim1'], dim_reduction_type='PCA')

# Compositional Representation---------------------------------------------
# setups = [1] # Go, Anti family
# setups = [2] # Ctx DM family
# =============================================================================
# setups = [1, 2]
# for setup in setups:
#     pass
#     taskset.plot_taskspace_group(root_dir, setup=setup,
#                                  restore=True, representation='rate')
#     taskset.plot_taskspace_group(root_dir, setup=setup,
#                                  restore=True, representation='weight')
#     taskset.plot_replacerule_performance_group(
#             root_dir, setup=setup, restore=True)
# =============================================================================

# Continual Learning Analysis----------------------------------------------
# TODO(gryang): Remains to be fixed
# performance.get_allperformance('*cont')
# performance.plot_performanceprogress_cont(('0_2cont', '0_0cont'))
# performance.plot_finalperformance_cont('*2cont', '*0cont')
# for save_pattern in ['*3cont']:
#     variance.plot_hist_varprop_selection(save_pattern)
# performance.plot_performanceprogress(model_dir='0_3seqlowlr') # Plot performance during training
