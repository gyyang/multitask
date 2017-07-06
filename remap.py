"""
Analysis of anti units
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

save = True
fast_eval = True

data_type  = 'rule'
save_addon = 'allrule_softplus_0_300test'
with open('data/variance'+data_type+save_addon+'.pkl','rb') as f:
    res = pickle.load(f)
h_var_all = res['h_var_all']
rules     = res['keys']

# First only get active units. Total variance across tasks larger than 1e-3
ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
# ind_active = np.where(h_var_all.sum(axis=1) > 0.)[0]
h_var_all  = h_var_all[ind_active, :]

# Normalize by the total variance across tasks
h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

########################## Get Anti Units ####################################
# Directly search
# This will be a stricter subset of the anti modules found in clustering results
rules_anti         = np.array([FDANTI, DELAYANTI])
rules_nonanti      = np.array([r for r in rules if r not in rules_anti])

ind_rules_anti     = [rules.index(r) for r in rules_anti]
ind_rules_nonanti  = [rules.index(r) for r in rules_nonanti]

h_normvar_all_anti     = h_normvar_all[:, ind_rules_anti].sum(axis=1)
h_normvar_all_nonanti  = h_normvar_all[:, ind_rules_nonanti].sum(axis=1)

plt.figure()
_ = plt.hist(h_normvar_all_anti, bins=50)
plt.xlabel('Proportion of variance in anti tasks')
plt.show()

ind_anti = np.where(h_normvar_all_anti>0.5)[0]
ind_nonanti = np.where(h_normvar_all_anti<=0.5)[0]
ind_anti_orig = ind_active[ind_anti] # Indices of anti units in the original matrix
ind_nonanti_orig = ind_active[ind_nonanti]
# Use clustering results (tend to be loose)
# label_anti    = np.where(label_prefs==FDANTI)[0][0]
# ind_anti      = np.where(labels==label_anti)[0]
# ind_anti_orig = ind_orig[ind_anti] # Indices of anti units in the original matrix

########################## Analyze Anti Units ################################

if True:
    plot_singleneuron_intime(save_addon, ind_anti_orig[2], [FDANTI, FDGO], save=True, ylabel_firstonly = True)


###################### Plot single unit connection ##########################
with Run(save_addon) as R:
    w_out, b_out, w_rec, w_in, b_rec = R.w_out, R.b_out, R.w_rec, R.w_in, R.b_rec
    config = R.config
N_RING = config['N_RING']

neuron = ind_anti_orig[5] # Example unit


########################### Plotting Input/Output Connectivity #################
if True:
    w_in_ = (w_in[:, 1:N_RING+1]+w_in[:, 1+N_RING:2*N_RING+1])/2.
    w_out_ = w_out[1:, :].T

    for ind_group, unit_type in zip([ind_anti_orig, ind_nonanti_orig],
                                    ['Anti units', 'Non-Anti Units']):
        # ind_group = ind_anti_orig
        n_group    = len(ind_group)
        w_in_group = np.zeros((n_group, N_RING))
        w_out_group = np.zeros((n_group, N_RING))

        for i, ind in enumerate(ind_group):
            tmp_in           = w_in_[ind, :]
            tmp_out           = w_out_[ind, :]

            # Sort by input weights
            ind_max       = np.argmax(tmp_in)

            w_in_group[i, :] = np.roll(tmp_in, int(N_RING/2)-ind_max)
            w_out_group[i, :] = np.roll(tmp_out, int(N_RING/2)-ind_max)

        w_in_ave = w_in_group.mean(axis=0)
        w_out_ave = w_out_group.mean(axis=0)

        fs = 6
        fig = plt.figure(figsize=(1.5, 1.0))
        ax = fig.add_axes([.3, .3, .6, .6])
        ax.plot(w_in_ave, color='black', label='In')
        ax.plot(w_out_ave, color='red', label='Out')
        ax.set_xticks([int(N_RING/2)])
        ax.set_xticklabels(['Preferred dir.'])
        # ax.set_xlabel(xlabel, fontsize=fs, labelpad=3)
        ax.set_ylabel('Conn. weight', fontsize=fs)
        lg = ax.legend(fontsize=fs, bbox_to_anchor=(1.1,1.1),
                       labelspacing=0.2, loc=1, frameon=False)
        # plt.setp(lg.get_title(),fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_title(unit_type, fontsize=fs, y=0.9)
        plt.locator_params(axis='y',nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if save:
            plt.savefig('figure/conn_'+unit_type+'.pdf', transparent=True)

########################### Plotting Rule Connectivity ########################
if True:
    for ind, unit_type in zip([ind_anti_orig, ind_nonanti_orig],
                              ['Anti units', 'Non-Anti units']):
        b1 = w_in[:, 2*N_RING+1+rules_anti][ind, :].flatten()
        b2 = w_in[:, 2*N_RING+1+rules_nonanti][ind, :].flatten()

        fs = 6
        fig = plt.figure(figsize=(1.5,1.2))
        ax = fig.add_axes([0.3,0.3,0.6,0.4])
        ax.boxplot([b1, b2], showfliers=False)
        ax.set_xticklabels(['Anti', 'Non-Anti'])
        ax.set_xlabel('Input from rule units', fontsize=fs, labelpad=3)
        ax.set_ylabel('Conn. weight', fontsize=fs)
        ax.set_title('To '+unit_type, fontsize=fs, y=0.9)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(axis='y',nbins=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if save:
            plt.savefig('figure/connweightrule_'+unit_type+'.pdf', transparent=True)
        plt.show()

########################### Plotting Recurrent Connectivity ###################
if True:
    from scipy.stats import binned_statistic

    w_in_ = (w_in[:, 1:N_RING+1]+w_in[:, 1+N_RING:2*N_RING+1])/2.
    w_out_ = w_out[1:, :].T

    inds = [ind_nonanti_orig, ind_anti_orig]
    names = ['Non-Anti', 'Anti']

    i_pairs = [(0,0), (0,1), (1,0), (1,1)]

    pref_diffs_list = list()
    w_recs_list = list()

    w_rec_bygroup = np.zeros((2,2))

    for i_pair in i_pairs:
        ind1, ind2 = inds[i_pair[0]], inds[i_pair[1]]
        # For each neuron get the preference based on input weight
        # sort by weights
        w_sortby = w_in_
        # w_sortby = w_out_
        prefs1 = np.argmax(w_sortby[ind1, :], axis=1)*2.*np.pi/N_RING
        prefs2 = np.argmax(w_sortby[ind2, :], axis=1)*2.*np.pi/N_RING

        # Compute the pairwise distance based on preference
        # Then get the connection weight between pairs
        pref_diffs = list()
        w_recs = list()
        for i, ind_i in enumerate(ind1):
            for j, ind_j in enumerate(ind2):
                if ind_j == ind_i:
                    # Excluding self connections, which tend to be positive
                    continue
                pref_diffs.append(get_dist(prefs1[i]-prefs2[j]))
                # pref_diffs.append(prefs1[i]-prefs2[j])
                w_recs.append(w_rec[ind_j, ind_i])
        pref_diffs, w_recs = np.array(pref_diffs), np.array(w_recs)
        pref_diffs_list.append(pref_diffs)
        w_recs_list.append(w_recs)

        w_rec_bygroup[i_pair[1], i_pair[0]] = np.mean(w_recs[pref_diffs<np.pi/6.])
    

    fs = 6
    vmax = np.ceil(np.max(w_rec_bygroup)*100)/100.
    vmin = np.floor(np.min(w_rec_bygroup)*100)/100.
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.2, 0.1, 0.5, 0.5])
    im = ax.imshow(w_rec_bygroup, cmap='coolwarm',
                   aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position("top")
    plt.yticks(range(len(names)), names,
               rotation=90, va='center', fontsize=fs)
    plt.xticks(range(len(names)), names,
               rotation=0, ha='center', fontsize=fs)
    ax.tick_params('both', length=0)
    ax.set_xlabel('From', fontsize=fs, labelpad=2)
    ax.set_ylabel('To', fontsize=fs, labelpad=2)
    for loc in ['bottom','top','left','right']:
        ax.spines[loc].set_visible(False)
    
    ax = fig.add_axes([0.72, 0.1, 0.03, 0.5])
    cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Rec. weight', fontsize=fs, labelpad=-7)
    plt.tick_params(axis='both', which='major', labelsize=fs)
    if save:
        plt.savefig('figure/conn_anti_task_recurrent.pdf', transparent=True)



# Continue to plot the recurrent connections by difference in input weight
if False:
    fs = 6
    fig = plt.figure(figsize=(3, 3.0))
    ax = fig.add_axes([.3, .3, .6, .6])

    for i, i_pair in enumerate(i_pairs):
        bin_means, bin_edges, binnumber = binned_statistic(
            pref_diffs_list[i], w_recs_list[i], bins=6, statistic='mean')

        ax.plot(bin_means, label=names[i_pair[0]]+' to '+names[i_pair[1]])
        ax.set_xticks([int(N_RING/2)])
        ax.set_xticklabels(['preferred input loc.'])
        # ax.set_xlabel(xlabel, fontsize=fs, labelpad=3)
        ax.set_ylabel('conn. weight', fontsize=fs)
        lg = ax.legend(fontsize=fs, bbox_to_anchor=(1.1,1.1),
                       labelspacing=0.2, loc=1, frameon=False)
        # plt.setp(lg.get_title(),fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        # ax.set_title(names[i_pair[0]]+' to '+names[i_pair[1]], fontsize=fs, y=0.9)
        plt.locator_params(axis='y',nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # if save:
        #     plt.savefig('figure/conn_'+unit_type+'_'+save_addon+'.pdf', transparent=True)


# ########### Plot Causal Manipulation Results  #############################
if False:
    perfs, perfs_lesion_anti = list(), list()
    for lesion_units, perfs_store in zip([None, ind_anti_orig],
                                         [perfs, perfs_lesion_anti]):
        with Run(save_addon, lesion_units=lesion_units, fast_eval=fast_eval) as R:
            config = R.config
            for rule in rules:
                task = generate_onebatch(rule=rule, config=config, mode='test')
                y_hat = R.f_y_from_x(task.x)
                perf = get_perf(y_hat, task.y_loc)
                perfs_store.append(perf.mean())

    perf_diff = np.array(perfs_lesion_anti) - np.array(perfs)
    perf_diff_antirules    = perf_diff[ind_rules_anti].mean()
    perf_diff_nonantirules = perf_diff[ind_rules_nonanti].mean()

    perfs, perfs_lesion_anti = np.array(perfs), np.array(perfs_lesion_anti)
    perf_plots = [np.mean(perfs[ind_rules_nonanti]), np.mean(perfs[ind_rules_anti]),
                  np.mean(perfs_lesion_anti[ind_rules_nonanti]), np.mean(perfs_lesion_anti[ind_rules_anti])]

    fs = 6
    width = 0.3
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.3,0.3,0.6,0.4])
    b0 = ax.bar(np.arange(2)-width, [np.mean(perfs[ind_rules_nonanti]), np.mean(perfs[ind_rules_anti])],
           width=width, color=sns.xkcd_palette(['orange'])[0], edgecolor='none')
    b1 = ax.bar(np.arange(2), [np.mean(perfs_lesion_anti[ind_rules_nonanti]), np.mean(perfs_lesion_anti[ind_rules_anti])],
           width=width, color=sns.xkcd_palette(['green'])[0], edgecolor='none')
    ax.plot([-width/2, width/2], [np.mean(perfs[ind_rules_nonanti]), np.mean(perfs_lesion_anti[ind_rules_nonanti])],
           '.-', color='gray')
    ax.plot([1-width/2, 1+width/2], [np.mean(perfs[ind_rules_anti]), np.mean(perfs_lesion_anti[ind_rules_anti])],
            '.-', color='gray')
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(['Non-Anti', 'Anti'])
    ax.set_xlabel('Tasks', fontsize=fs, labelpad=3)
    ax.set_ylabel('Performance', fontsize=fs)
    lg = ax.legend((b0, b1), ('Control', 'Anti units all lesioned'),
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
    if save:
        plt.savefig('figure/perflesion_anti_units.pdf', transparent=True)
    plt.show()


# # A.plot_unit_connection(inds)
# ga = GeneralAnalysis(save_addon=A.save_addon_original)
# success = False
# for ind in inds_anti:
#     res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING])
#     if res.rvalue<-0.9 and res.pvalue<0.01:
#         success = True
#         break
#
# if success:
#     rules = [FDGO, DELAYANTI, FDANTI]
#     ga.run_test(rules=rules)
#     ga.plot_singleneuron_intime(neuron_id=ind, rules=rules, save_fig=True)
# else:
#     raise ValueError('Did not success to find a typical anti unit in this network')
#
#

# ########### Plot histogram of correlation coefficient #####################
# slopes = list()
# rvalues = list()
# pvalues = list()
# anti_units = list()
# HDIMs = range(190,207)
# for j, HDIM in enumerate(HDIMs):
#     N_RING = 16
#     save_addon = 'chanceabbott_'+str(HDIM)+'_'+str(N_RING)
#     A = SumStatAnalysis('rule', save_addon)
#     inds_anti = A.find_anti_units()
#
#     # A.plot_unit_connection(inds)
#     ga = GeneralAnalysis(save_addon=A.save_addon_original)
#     for ind in range(HDIM):
#         res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING])
#         slopes.append(res.slope)
#         rvalues.append(res.rvalue)
#         pvalues.append(res.pvalue)
#         anti_units.append(ind in inds_anti)
#
#     if j == 0:
#         conn_rule_to_all = ga.Win[:, 1+2*N_RING:] # connection from rule inputs to all units
#         conn_rule_to_anti = ga.Win[inds_anti, 1+2*N_RING:] # connection from rule inputs to anti units
#     else:
#         conn_rule_to_all = np.concatenate((conn_rule_to_all, ga.Win[:, 1+2*N_RING:]))
#         conn_rule_to_anti = np.concatenate((conn_rule_to_anti, ga.Win[inds_anti, 1+2*N_RING:]))
#
#
# slopes, rvalues, pvalues, anti_units = np.array(slopes), np.array(rvalues), np.array(pvalues), np.array(anti_units)
#
# # plot_value, plot_range = slopes, (-4,4)
# plot_value, plot_range = rvalues, (-1,1)
# thres = 0.01 ##TODO: Find out how to set this threshold
# for i in [0,1]:
#     if i == 0:
#         units = anti_units
#         title = 'Anti'
#     else:
#         units = (1-anti_units).astype(bool)
#         title = 'Non-anti'
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
# ax.plot(conn_rule_to_anti.mean(axis=0)[ga.rules], 'o-', markersize=3, label='anti', color=sns.xkcd_palette(['red'])[0])
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
# plt.savefig('figure/conn_ruleinput_to_anti.pdf', transparent=True)
# plt.show()
