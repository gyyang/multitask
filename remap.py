"""
Analysis of remap units
"""

from __future__ import division

import os
import numpy as np
import pickle
import time
import copy
from collections import OrderedDict
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work
from task import *
from run import Run, plot_singleneuron_intime
from network import get_perf


save_addon = 'tf_latest_500'
with open('data/clustering'+save_addon+'.pkl','rb') as f:
    result = pickle.load(f)
labels          = result['labels']
label_prefs     = result['label_prefs'] # Map the label to its preferred rule
h_normvar_all   = result['h_normvar_all']
ind_orig        = result['ind_orig']
rules           = result['rules']

########################## Get Remap Units ####################################
# Directly search
# This will be a stricter subset of the remap modules found in clustering results
rules_remap = np.array([REMAP, INHREMAP, DELAYREMAP])
rules_nonremap = np.array([r for r in rules if r not in rules_remap])
ind_rules_remap = [rules.index(r) for r in rules_remap]
ind_rules_nonremap = [rules.index(r) for r in rules_nonremap]
h_normvar_all_remap = h_normvar_all[:, ind_rules_remap].sum(axis=1)
h_normvar_all_nonremap = h_normvar_all[:, ind_rules_nonremap].sum(axis=1)

#==============================================================================
# plt.figure()
# _ = plt.hist(h_normvar_all_remap, bins=50)
# plt.xlabel('Proportion of variance in remap tasks')
# plt.show()
#==============================================================================

ind_remap = np.where(h_normvar_all_remap>0.9)[0]
ind_remap_orig = ind_orig[ind_remap] # Indices of remap units in the original matrix

# Use clustering results (tend to be loose)
# label_remap    = np.where(label_prefs==INHREMAP)[0][0]
# ind_remap      = np.where(labels==label_remap)[0]
# ind_remap_orig = ind_orig[ind_remap] # Indices of remap units in the original matrix

########################## Analyze Remap Units ################################

# plot_singleneuron_intime(save_addon, ind_remap_orig[-2], [INHREMAP, INHGO], save=True, ylabel_firstonly = True)


###################### Plot single unit connection ##########################
with Run(save_addon) as R:
    w_out, b_out, w_rec, w_in, b_rec = R.w_out, R.b_out, R.w_rec, R.w_in, R.b_rec
    config = R.config
N_RING = config['N_RING']

neuron = ind_remap_orig[-2] # Example unit

if False:
    # Connection with ring input and output
    w_in0 = w_in[neuron, 1:1+N_RING]
    w_out0 = w_out[:,neuron][1:]

    fs = 6
    fig = plt.figure(figsize=(1.5,0.8))
    ax = fig.add_axes([0.3,0.25,0.6,0.55])
    ax.plot(w_in0, color=sns.xkcd_palette(['green'])[0], label='From input')
    ax.plot(w_out0, color=sns.xkcd_palette(['blue'])[0], label='To output')
    plt.ylabel('Conn. weight', fontsize=fs, labelpad=1)
    plt.xlabel('Degree', fontsize=fs, labelpad=-6)
    plt.xticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$','',r'360$\degree$'])
    plt.title('Unit {:d} '.format(neuron), fontsize=fs)

    wmax = max((w_in0.max(),w_out0.max()))
    wmin = min((w_in0.min(),w_out0.min()))
    plt.ylim([wmin-0.1*(wmax-wmin),wmax+0.7*(wmax-wmin)])
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.locator_params(axis='y',nbins=5)
    lg = plt.legend(ncol=1,bbox_to_anchor=(1.0,1.15),frameon=False,
                    fontsize=fs,labelspacing=0.3,loc=1)
    plt.savefig('figure/connweight_remap_unit'+str(neuron)+save_addon+'.pdf', transparent=True)
    plt.show()


if True:
    remaprule_to_remap = w_in[:, 2*N_RING+rules_remap][ind_remap_orig, :].flatten()
    nonremaprule_to_remap = w_in[:, 2*N_RING+rules_nonremap][ind_remap_orig, :].flatten()

    fs = 6
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.3,0.3,0.6,0.4])
    ax.boxplot([nonremaprule_to_remap, remaprule_to_remap], showfliers=False)
    ax.set_xticklabels(['Non-remap', 'Remap'])
    ax.set_xlabel('Input from rule units', fontsize=fs, labelpad=3)
    ax.set_ylabel('Conn. weight', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.locator_params(axis='y',nbins=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
#    ax.set_xlim([-0.5, 1.5])
    plt.savefig('figure/connweightrule_remap_unit'+str(neuron)+save_addon+'.pdf', transparent=True)
    plt.show()


# ########### Plot Causal Manipulation Results  #############################
if False:
    perfs, perfs_lesion_remap = list(), list()
    for lesion_units, perfs_store in zip([None, ind_remap_orig],
                                         [perfs, perfs_lesion_remap]):
        with Run(save_addon, lesion_units=lesion_units) as R:
            config = R.config
            for rule in rules:
                task = generate_onebatch(rule=rule, config=config, mode='test')
                y_hat = R.f_y_from_x(task.x)
                perf = get_perf(y_hat, task.y_loc)
                perfs_store.append(perf.mean())

    perf_diff = np.array(perfs_lesion_remap) - np.array(perfs)
    perf_diff_remaprules    = perf_diff[ind_rules_remap].mean()
    perf_diff_nonremaprules = perf_diff[ind_rules_nonremap].mean()

    perfs, perfs_lesion_remap = np.array(perfs), np.array(perfs_lesion_remap)
    perf_plots = [np.mean(perfs[ind_rules_nonremap]), np.mean(perfs[ind_rules_remap]),
                  np.mean(perfs_lesion_remap[ind_rules_nonremap]), np.mean(perfs_lesion_remap[ind_rules_remap])]

    fs = 6
    width = 0.3
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.3,0.3,0.6,0.4])
    b0 = ax.bar(np.arange(2)-width, [np.mean(perfs[ind_rules_nonremap]), np.mean(perfs[ind_rules_remap])],
           width=width, color=sns.xkcd_palette(['orange'])[0], edgecolor='none')
    b1 = ax.bar(np.arange(2), [np.mean(perfs_lesion_remap[ind_rules_nonremap]), np.mean(perfs_lesion_remap[ind_rules_remap])],
           width=width, color=sns.xkcd_palette(['green'])[0], edgecolor='none')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['Non-remap', 'Remap'])
    ax.set_xlabel('Rules', fontsize=fs, labelpad=3)
    ax.set_ylabel('performance', fontsize=fs)
    lg = ax.legend((b0, b1), ('Control', 'Remap units lesioned'),
                   fontsize=fs, ncol=1, bbox_to_anchor=(1,1.7),
                   labelspacing=0.2, loc=1, frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.locator_params(axis='y',nbins=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #    ax.set_xlim([-0.5, 1.5])
    plt.savefig('figure/perflesion_remap_units'+save_addon+'.pdf', transparent=True)
    plt.show()


# # A.plot_unit_connection(inds)
# ga = GeneralAnalysis(save_addon=A.save_addon_original)
# success = False
# for ind in inds_remap:
#     res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING])
#     if res.rvalue<-0.9 and res.pvalue<0.01:
#         success = True
#         break
#
# if success:
#     rules = [INHGO, DELAYREMAP, INHREMAP]
#     ga.run_test(rules=rules)
#     ga.plot_singleneuron_intime(neuron_id=ind, rules=rules, save_fig=True)
# else:
#     raise ValueError('Did not success to find a typical remap unit in this network')
#
#

# ########### Plot histogram of correlation coefficient #####################
# slopes = list()
# rvalues = list()
# pvalues = list()
# remap_units = list()
# HDIMs = range(190,207)
# for j, HDIM in enumerate(HDIMs):
#     N_RING = 16
#     save_addon = 'chanceabbott_'+str(HDIM)+'_'+str(N_RING)
#     A = SumStatAnalysis('rule', save_addon)
#     inds_remap = A.find_remap_units()
#
#     # A.plot_unit_connection(inds)
#     ga = GeneralAnalysis(save_addon=A.save_addon_original)
#     for ind in range(HDIM):
#         res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING])
#         slopes.append(res.slope)
#         rvalues.append(res.rvalue)
#         pvalues.append(res.pvalue)
#         remap_units.append(ind in inds_remap)
#
#     if j == 0:
#         conn_rule_to_all = ga.Win[:, 1+2*N_RING:] # connection from rule inputs to all units
#         conn_rule_to_remap = ga.Win[inds_remap, 1+2*N_RING:] # connection from rule inputs to remap units
#     else:
#         conn_rule_to_all = np.concatenate((conn_rule_to_all, ga.Win[:, 1+2*N_RING:]))
#         conn_rule_to_remap = np.concatenate((conn_rule_to_remap, ga.Win[inds_remap, 1+2*N_RING:]))
#
#
# slopes, rvalues, pvalues, remap_units = np.array(slopes), np.array(rvalues), np.array(pvalues), np.array(remap_units)
#
# # plot_value, plot_range = slopes, (-4,4)
# plot_value, plot_range = rvalues, (-1,1)
# thres = 0.01 ##TODO: Find out how to set this threshold
# for i in [0,1]:
#     if i == 0:
#         units = remap_units
#         title = 'Remap'
#     else:
#         units = (1-remap_units).astype(bool)
#         title = 'Non-remap'
#
#     fig = plt.figure(figsize=(1.5,1.2))
#     ax = fig.add_axes([0.3,0.3,0.6,0.5])
#     hist, bins_edge = np.histogram(plot_value[units], range=plot_range, bins=30)
#     ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=sns.xkcd_palette(['navy blue'])[0], edgecolor='none')
#     hist, bins_edge = np.histogram(plot_value[units*(pvalues<thres)], range=plot_range, bins=30)
#     ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
#     plt.xlabel('corr. coeff.', fontsize=7, labelpad=1)
#     plt.ylabel('counts', fontsize=7)
#     plt.locator_params(nbins=3)
#     plt.title(title+' units ({:d} nets)'.format(len(HDIMs)) , fontsize=7)
#     ax.tick_params(axis='both', which='major', labelsize=7)
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     plt.savefig('figure/'+title+'_unit_cc.pdf', transparent=True)
#     plt.show()
#
# ########### Plot connections averaged across networks #####################
# fig = plt.figure(figsize=(2.5,2))
# ax = fig.add_axes([0.2,0.5,0.75,0.4])
# ax.plot(conn_rule_to_all.mean(axis=0)[ga.rules], 'o-', markersize=3, label='all', color=sns.xkcd_palette(['black'])[0])
# ax.plot(conn_rule_to_remap.mean(axis=0)[ga.rules], 'o-', markersize=3, label='remap', color=sns.xkcd_palette(['red'])[0])
#
# plt.xticks(range(len(ga.rules)), [rule_name[rule] for rule in ga.rules], rotation=90)
# plt.ylabel('Mean conn. from rule', fontsize=7)
# plt.title('Average of {:d} networks'.format(len(HDIMs)), fontsize=7)
# #plt.ylim(bottom=0.0, top=1.05)
# plt.xlim([-0.5, len(ga.rules)-0.5])
# ax.tick_params(axis='both', which='major', labelsize=7)
# leg = plt.legend(title='to units',fontsize=7,frameon=False, loc=2, numpoints=1,
#                  bbox_to_anchor=(0.3,0.6), labelspacing=0.3)
# plt.setp(leg.get_title(),fontsize=7)
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.locator_params(axis='y', nbins=4)
# plt.savefig('figure/conn_ruleinput_to_remap.pdf', transparent=True)
# plt.show()


