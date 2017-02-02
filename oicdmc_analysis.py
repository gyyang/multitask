"""
Analysis of the OIC and DMC tasks
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

def gen_taskparams(n_tar, n_rep):
    batch_size = n_rep * n_tar**2
    batch_shape = (n_tar, n_tar, n_rep)
    ind_tar_mod1, ind_tar_mod2, ind_rep = np.unravel_index(range(batch_size),batch_shape)

    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi) % (2*np.pi)

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs}
              # If tar_time is long (~1600), we can reproduce the curving trajectories
    return params, batch_size

save_addon = 'oicdmconly_weaknoise_test'
rules = [OIC, DMC]

# Analyzing the sample period
n_rep = 1
n_tar_loc = 12
batch_size = n_rep * n_tar_loc
batch_shape = (n_rep, n_tar_loc)
ind_rep, ind_tar_loc = np.unravel_index(range(batch_size),batch_shape)

tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
tar1_cats = tar1_locs<np.pi # Category of target 1

tar2_locs = [0]     * batch_size
tar3_locs = [np.pi] * batch_size

params = {'tar1_locs' : tar1_locs,
          'tar2_locs' : tar2_locs,
          'tar3_locs' : tar3_locs}

with Run(save_addon, fast_eval=True) as R:
    config = R.config

    rule = OIC
    task = generate_onebatch(rule, config, 'psychometric', params=params)
    x_sample = task.x
    h_sample = R.f_h(x_sample)
    y_sample = R.f_y(h_sample)



h_sample_cat1 = h_sample[:, tar1_cats==1, :]
h_sample_cat2 = h_sample[:, tar1_cats==0, :]


for plot_ind in range(config['HDIM']):
    fig = plt.figure(figsize=(2,2))
    t_plot = np.arange(h_sample.shape[0])*config['dt']/1000
    ax = fig.add_axes([.2, .2, .7, .7])
    _ = ax.plot(t_plot, h_sample_cat1[:, :, plot_ind], color='blue')
    _ = ax.plot(t_plot, h_sample_cat2[:, :, plot_ind], color='red')