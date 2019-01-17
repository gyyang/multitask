"""
Main file for generating results in the paper:
Clustering and compositionality of task representations
in a neural network trained to perform many cognitive tasks
Yang GR et al. 2017 BioRxiv
"""
from __future__ import absolute_import

import tools
from analysis import performance
from analysis import standard_analysis
from analysis import clustering
from analysis import variance
from analysis import taskset
from analysis import varyhp
from analysis import data_analysis
from analysis import contextdm_analysis
from analysis import posttrain_analysis


# Directories of the models and the sample model
# Change these to your directories
# root_dir = './data/tanhgru'
root_dir = './data/train_all'
model_dir = root_dir + '/1'


# # Performance Analysis-----------------------------------------------------
# standard_analysis.schematic_plot(model_dir=model_dir)
# performance.plot_performanceprogress(model_dir)
# performance.psychometric_choice(model_dir)  # Psychometric for dm
# performance.psychometric_choiceattend(model_dir, no_ylabel=True)
# performance.psychometric_choiceint(model_dir, no_ylabel=True)
#
# for rule in ['dm1', 'contextdm1', 'multidm']:
#     performance.plot_choicefamily_varytime(model_dir, rule)
# performance.psychometric_delaychoice_varytime(model_dir, 'delaydm1')
#
#
# # Clustering Analysis------------------------------------------------------
model_dir = root_dir + '/1'
# CA = clustering.Analysis(model_dir, data_type='rule')
# CA.plot_example_unit()
# CA.plot_cluster_score()
# CA.plot_variance()
# CA.plot_2Dvisualization('PCA')
# CA.plot_2Dvisualization('MDS')
# CA.plot_2Dvisualization('tSNE')
# CA.plot_lesions()
# CA.plot_connectivity_byclusters()
#
#
CA = clustering.Analysis(model_dir, data_type='epoch')
CA.plot_variance()
# CA.plot_2Dvisualization('tSNE')
#
#
# # Varying hyperparameter analysis------------------------------------------
# varyhp_root_dir = './data/varyhp'
# n_clusters, hp_list = varyhp.get_n_clusters(varyhp_root_dir)
# varyhp.plot_n_clusters(n_clusters, hp_list)
# varyhp.plot_n_cluster_hist(n_clusters, hp_list)
#
#
# # FTV Analysis-------------------------------------------------------------
# variance.plot_hist_varprop_selection(root_dir)
# variance.plot_hist_varprop_selection('./data/tanhgru')
# variance.plot_hist_varprop_all(root_dir, plot_control=True)
#
#
# # ContextDM analysis-------------------------------------------------------
# ua = contextdm_analysis.UnitAnalysis(model_dir)
# ua.plot_inout_connections()
# ua.plot_rec_connections()
# ua.plot_rule_connections()
# ua.prettyplot_hist_varprop()
#
# contextdm_analysis.plot_performance_choicetasks(model_dir, grouping='var')
# contextdm_analysis.plot_performance_2D_all(model_dir, 'contextdm1')
#
#
# # Task Representation------------------------------------------------------
# tsa = taskset.TaskSetAnalysis(model_dir)
# tsa.compute_and_plot_taskspace(epochs=['stim1'], dim_reduction_type='PCA')
#
#
# # Compositional Representation---------------------------------------------
# setups = [1, 2, 3]
# for setup in setups:
#     taskset.plot_taskspace_group(root_dir, setup=setup,
#                                  restore=True, representation='rate')
#     taskset.plot_taskspace_group(root_dir, setup=setup,
#                                  restore=True, representation='weight')
#     taskset.plot_replacerule_performance_group(
#             root_dir, setup=setup, restore=True)

# name = 'tanhgru'
# name = 'mixrule'
# name = 'mixrule_softplus'
# setups = [1, 2]
# d = './data/' + name
# for setup in setups:
#     taskset.plot_taskspace_group(d, setup=setup,
#                                  restore=False, representation='rate',
#                                  fig_name_addon=name)
    # taskset.plot_taskspace_group(d, setup=setup,
    #                              restore=True, representation='weight',
    #                              fig_name_addon=name)
    # taskset.plot_replacerule_performance_group(
    #     d, setup=setup, restore=False, fig_name_addon=name)


## Continual Learning Analysis----------------------------------------------
# hp_target0 = {'c_intsyn': 0, 'ksi_intsyn': 0.01,
#               'activation': 'relu', 'max_steps': 4e5}
# hp_target1 = {'c_intsyn': 1, 'ksi_intsyn': 0.01,
#               'activation': 'relu', 'max_steps': 4e5}
# model_dirs0 = tools.find_all_models('data/seq/', hp_target0)
# model_dirs1 = tools.find_all_models('data/seq/', hp_target1)
# model_dirs0 = tools.select_by_perf(model_dirs0, perf_min=0.8)
# model_dirs1 = tools.select_by_perf(model_dirs1, perf_min=0.8)
# performance.plot_performanceprogress_cont((model_dirs0[0], model_dirs1[2]))
# performance.plot_finalperformance_cont(model_dirs0, model_dirs1)
# data_analysis.plot_fracvar_hist_byhp(hp_vary='c_intsyn', mode='all_var', legend=False)
# data_analysis.plot_fracvar_hist_byhp(hp_vary='p_weight_train', mode='all_var')


## Data analysis------------------------------------------------------------
# Note that these wouldn't work without the data file
# data_analysis.plot_all('mante_single_ar')
# data_analysis.plot_all('mante_single_fe')
# data_analysis.plot_all('mante_ar')
# data_analysis.plot_all('mante_fe')

## Post-training of pre-trained networks------------------------------------
# for posttrain_setup in range(2):
#     for trainables in ['all', 'rule']:
#         posttrain_analysis.plot_posttrain_performance(posttrain_setup, trainables)
