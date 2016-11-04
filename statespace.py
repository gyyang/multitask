"""
State space analysis for decision tasks
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

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'


save_addon = 'tf_latest_500'
rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]


print('Starting standard analysis of the CHOICEATTEND task...')
with Run(save_addon) as R:

    n_tar_loc = 100 # increase repeat by increasing this
    n_tar = 2
    batch_size = n_tar_loc * n_tar**2
    batch_shape = (n_tar_loc,n_tar,n_tar)
    ind_tar_loc, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

    tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
    tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

    tar_str_range = 0.4
    tar1_mod1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar_mod1/(n_tar-1)
    tar2_mod1_strengths = 2 - tar1_mod1_strengths
    tar1_mod2_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar_mod2/(n_tar-1)
    tar2_mod2_strengths = 2 - tar1_mod2_strengths

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs,
              'tar1_mod1_strengths' : tar1_mod1_strengths,
              'tar2_mod1_strengths' : tar2_mod1_strengths,
              'tar1_mod2_strengths' : tar1_mod2_strengths,
              'tar2_mod2_strengths' : tar2_mod2_strengths,
              'tar_time'    : 800}

    h_samples = dict()
    for rule in rules:
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
        # Only study target epoch
        epoch = task.epochs['tar1']
        h_samples[rule] = R.f_h(task.x)[epoch[0]:epoch[1],...][::20,...]

    for rule in [DMCGO]: # for control
        task  = generate_onebatch(rule, R.config, 'test')
        h_samples[rule] = R.f_h(task.x)[::50,...]

nt, nb, nh = h_samples[CHOICEATTEND_MOD1].shape
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(h_samples[CHOICEATTEND_MOD1].reshape((-1, nh)))

h_trans = dict()
for rule in rules:
    h_trans[rule] = pca.transform(h_samples[rule].reshape((-1, nh)))
ev = pca.explained_variance_
evr = pca.explained_variance_ratio_
# plt.figure()
# plt.plot(h_trans[:,0], h_trans[:,1], '.')
# plt.show()

# Plot multiple rules:
# colors = sns.color_palette('dark', len(rules))
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i, rule in enumerate(rules):
#     h_plot = h_trans[rule]
#     h_plot = h_plot.reshape((nt, nb, nh))
#     #h_plot = h_plot[:,tar1_locs==np.unique(tar1_locs)[1],:]
#     h_plot = h_plot.reshape((-1,nh))
#     ax.plot(h_plot[:,0], h_plot[:,1], h_plot[:,2], '.', markersize=2, color=colors[i])
#     ax.plot([h_plot[0,0]], [h_plot[0,1]], [h_plot[0,2]], 'd', markersize=5, color=colors[i])
# plt.show()

# Plot multiple conditions
rule = CHOICEATTEND_MOD1
task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
y_locs = task.y_loc[-1]

#separate_bys = tar1_locs
#separate_bys = tar1_mod1_strengths
separate_bys = y_locs

#colors = sns.color_palette('dark', len(np.unique(separate_bys)))
colors = sns.color_palette("husl", len(np.unique(separate_bys)))
h_tran = h_trans[rule].reshape((nt, nb, nh))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, s in enumerate(np.unique(separate_bys)):
    h_plot = h_tran[:,separate_bys==s,:].reshape((-1,nh))
    ax.plot(h_plot[:,0], h_plot[:,1], h_plot[:,2], '.', markersize=2, color=colors[i])
    ax.plot([h_plot[0,0]], [h_plot[0,1]], [h_plot[0,2]], 'd', markersize=5, color=colors[i])
plt.show()