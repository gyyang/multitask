"""
Analysis of the delay-matching tasks
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
from run import Run
from network import get_perf
from slowpoints import search_slowpoints

save = True # TEMP


def gen_taskparams(tar1_loc, n_tar, n_rep):
    batch_size = n_rep * n_tar**2
    batch_shape = (n_tar, n_tar, n_rep)
    ind_tar_mod1, ind_tar_mod2, ind_rep = np.unravel_index(range(batch_size),batch_shape)

    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi) % (2*np.pi)

    # tar_cohs = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])*0.3
    # tar_cohs = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])*0.5
    tar_cohs = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])*0.5
    tar_mod1_cohs = np.array([tar_cohs[i] for i in ind_tar_mod1])
    tar_mod2_cohs = np.array([tar_cohs[i] for i in ind_tar_mod2])

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs,
              'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
              'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
              'tar1_mod2_strengths' : 1 + tar_mod2_cohs,
              'tar2_mod2_strengths' : 1 - tar_mod2_cohs,
              'tar_time'    : 1000}
              # If tar_time is long (~1600), we can reproduce the curving trajectories
    return params, batch_size

class UnitAnalysis(object):
    def __init__(self, save_addon, fast_eval=True):
        data_type  = 'rule'
        fname = 'data/variance'+data_type+save_addon
        with open(fname+'.pkl','rb') as f:
            res = pickle.load(f)
        h_var_all = res['h_var_all']
        keys      = res['keys']

        rules = [DMSGO, DMCGO]
        ind_rules = [keys.index(rule) for rule in rules]
        h_var_all = h_var_all[:, ind_rules]

        # First only get active units. Total variance across tasks larger than 1e-3
        ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
        h_var_all  = h_var_all[ind_active, :]

        # Normalize by the total variance across tasks
        h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

        ind_lesions = dict()

        ind_lesions['1']  = np.where(h_normvar_all[:,0]>0.9)[0]
        ind_lesions['12'] = np.where(np.logical_and(h_normvar_all[:,0]>0.2, h_normvar_all[:,0]<0.8))[0]
        ind_lesions['2']  = np.where(h_normvar_all[:,0]<0.1)[0]

        ind_lesions_orig = {key: ind_active[val] for key, val in ind_lesions.iteritems()}

        self.save_addon         = save_addon
        self.ind_lesions        = ind_lesions
        self.ind_lesions_orig   = ind_lesions_orig
        self.fast_eval          = fast_eval
        self.h_normvar_all      = h_normvar_all
        self.rules              = rules
        self.ind_active         = ind_active
        self.colors = dict(zip(['intact', '1', '2', '12'],
                               sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))

    def prettyplot_hist_varprop(self):
        # Similar to the function from variance.py, but prettier
        # Plot the proportion of variance for the first rule
        rules = self.rules

        fs = 6
        fig = plt.figure(figsize=(2.0,1.2))
        ax = fig.add_axes([0.2,0.3,0.5,0.5])
        data_plot = self.h_normvar_all[:, 0]
        hist, bins_edge = np.histogram(data_plot, bins=30, range=(0,1))
        ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
               color='gray', edgecolor='none')
        bs = list()
        for i, group in enumerate(['1', '2', '12']):
            data_plot = self.h_normvar_all[self.ind_lesions[group], 0]
            hist, bins_edge = np.histogram(data_plot, bins=30, range=(0,1))
            b_tmp = ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
                   color=self.colors[group], edgecolor='none', label=group)
            bs.append(b_tmp)
        plt.locator_params(nbins=3)
        xlabel = 'VarRatio({:s}, {:s})'.format(rule_name[rules[0]], rule_name[rules[1]])
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylim(bottom=-0.02*hist.max())
        ax.set_xlim([-0.1,1.1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='major', labelsize=fs, length=2)
        lg = ax.legend(bs, ['1','2','12'], title='Group',
                       fontsize=fs, ncol=1, bbox_to_anchor=(1.5,1.0),
                       loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        plt.savefig('figure/prettyplot_hist_varprop'+
                rule_name[rules[0]].replace(' ','')+
                rule_name[rules[1]].replace(' ','')+
                save_addon+'.pdf', transparent=True)


    def plot_performance_2D(self, rule, lesion_group=None, **kwargs):
        if lesion_group is None:
            lesion_units = None
            lesion_group_name = 'intact'
        elif lesion_group == '1+2':
            lesion_units = np.concatenate((self.ind_lesions_orig['1'],self.ind_lesions_orig['2']))
            lesion_group_name = 'lesion groups 1 & 2'
        else:
            lesion_units = self.ind_lesions_orig[lesion_group]
            lesion_group_name = 'lesion group ' + lesion_group

        n_rep = 20
        n_tar_loc = 10 # increase repeat by increasing this
        batch_size = n_rep * n_tar_loc**2
        batch_shape = (n_rep, n_tar_loc,n_tar_loc)
        ind_rep, ind_tar_loc1, ind_tar_loc2 = np.unravel_index(range(batch_size),batch_shape)

        # Looping target location
        tar1_locs = 2*np.pi*ind_tar_loc1/n_tar_loc
        tar2_locs = 2*np.pi*ind_tar_loc2/n_tar_loc

        params = {'tar1_locs' : tar1_locs,
                  'tar2_locs' : tar2_locs}

        with Run(self.save_addon, lesion_units=lesion_units, fast_eval=self.fast_eval) as R:
            task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
            y_sample = R.f_y_from_x(task.x)

        ## TODO: Need better way to determine performance
        if rule in [DMSGO, DMCGO]:
            match_response = y_sample[-1, :, 0] < 0.5 # Last time point, fixation unit, match if go
        elif rule in [DMSNOGO, DMCNOGO]:
            match_response = y_sample[-1, :, 0] > 0.5
        match_response = match_response.reshape(batch_shape)
        match_response = match_response.mean(axis=0)

        kwargs = dict()
        fs = 6
        fig = plt.figure(figsize=(1.5,1.5))
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        im = ax.imshow(match_response, cmap='BrBG', origin='lower',
                       aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Mod 2 loc.', fontsize=fs, labelpad=-3)
        plt.xticks([0, n_tar_loc-1], ['0', '360'],
                   rotation=0, va='center', fontsize=fs)
        if 'ylabel' in kwargs and kwargs['ylabel']==False:
            plt.yticks([])
        else:
            ax.set_ylabel('Mod 1 loc.', fontsize=fs, labelpad=-3)
            plt.yticks([0, n_tar_loc-1], [0, 360],
                       rotation=0, va='center', fontsize=fs)
        plt.title(rule_name[rule] + '\n' + lesion_group_name, fontsize=fs)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)

        if 'colorbar' in kwargs and kwargs['colorbar']==False:
            pass
        else:
            ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
            cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
            cb.outline.set_linewidth(0.5)
            cb.set_label('Prop. of match', fontsize=fs, labelpad=-3)
            plt.tick_params(axis='both', which='major', labelsize=fs)

        # plt.savefig('figure/'+rule_name[rule].replace(' ','')+
        #             '_perf2D_lesion'+str(lesion_group)+
        #             self.save_addon+'.pdf', transparent=True)
        plt.show()

        
save_addon = 'allrule_weaknoise_320'
# save_addon = 'allrule_weaknoise_400'
ua = UnitAnalysis(save_addon)
# ua.prettyplot_hist_varprop()
# for rule in [DMSGO, DMCGO]:
#     for lesion_group in ['1', '2', '12', '1+2']:
#         ua.plot_performance_2D(rule=rule, lesion_group=lesion_group, ylabel=False, colorbar=False)



n_rep = 1
n_tar_loc = 10 # increase repeat by increasing this
batch_size = n_rep * n_tar_loc**2
batch_shape = (n_rep, n_tar_loc,n_tar_loc)
ind_rep, ind_tar_loc1, ind_tar_loc2 = np.unravel_index(range(batch_size),batch_shape)

# Looping target location
tar1_locs = 2*np.pi*ind_tar_loc1/n_tar_loc
tar2_locs = 2*np.pi*ind_tar_loc2/n_tar_loc

params = {'tar1_locs' : tar1_locs,
          'tar2_locs' : tar2_locs}

# rule = DMSGO
rule = DMCGO
# lesion_group = '1'
# lesion_group = '2'
lesion_group = '12'
# lesion_group = None

if lesion_group is None:
    lesion_units = None
    lesion_group_name = 'intact'
elif lesion_group == '1+2':
    lesion_units = np.concatenate((ua.ind_lesions_orig['1'],ua.ind_lesions_orig['2']))
    lesion_group_name = 'lesion groups 1 & 2'
else:
    lesion_units = ua.ind_lesions_orig[lesion_group]
    lesion_group_name = 'lesion group ' + lesion_group
    
with Run(ua.save_addon, lesion_units=lesion_units, fast_eval=ua.fast_eval) as R:
    task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
    y_sample = R.f_y_from_x(task.x)

if rule in [DMSGO, DMCGO]:
    match_response = y_sample[-1, :, 0] < 0.5 # Last time point, fixation unit, match if go
elif rule in [DMSNOGO, DMCNOGO]:
    match_response = y_sample[-1, :, 0] > 0.5
match_response = match_response.reshape(batch_shape)
match_response = match_response.mean(axis=0)

kwargs = dict()
fs = 6
fig = plt.figure(figsize=(1.5,1.5))
ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
im = ax.imshow(match_response, cmap='BrBG', origin='lower',
               aspect='auto', interpolation='nearest', vmin=0, vmax=1)
ax.set_xlabel('Mod 2 loc.', fontsize=fs, labelpad=-3)
plt.xticks([0, n_tar_loc-1], ['0', '360'],
           rotation=0, va='center', fontsize=fs)
if 'ylabel' in kwargs and kwargs['ylabel']==False:
    plt.yticks([])
else:
    ax.set_ylabel('Mod 1 loc.', fontsize=fs, labelpad=-3)
    plt.yticks([0, n_tar_loc-1], [0, 360],
               rotation=0, va='center', fontsize=fs)
plt.title(rule_name[rule] + '\n' + lesion_group_name, fontsize=fs)
ax.tick_params('both', length=0)
for loc in ['bottom','top','left','right']:
    ax.spines[loc].set_visible(False)

if 'colorbar' in kwargs and kwargs['colorbar']==False:
    pass
else:
    ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
    cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Prop. of match', fontsize=fs, labelpad=-3)
    plt.tick_params(axis='both', which='major', labelsize=fs)

# plt.savefig('figure/'+rule_name[rule].replace(' ','')+
#             '_perf2D_lesion'+str(lesion_group)+
#             self.save_addon+'.pdf', transparent=True)
plt.show()
