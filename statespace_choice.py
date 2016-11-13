"""
State space analysis for the simple choice task
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
from slowpoints import find_slowpoints


save_addon = 'tf_latest_400'
rule = CHOICE_MOD1

def f_sub_mean(x):
    # subtract mean activity across batch conditions
    assert(len(x.shape)==3)
    x_mean = x.mean(axis=1)
    for i in range(x.shape[1]):
        x[:,i,:] -= x_mean
    return x

# Regressors
Choice, Mod1 = range(2)
regr_names = ['Choice', 'Mod 1']

subtract_t_mean=False
z_score = False

tar1_loc  = 0

def gen_taskparams(n_tar, n_rep, tar_str_range):
    batch_size = n_rep * n_tar
    batch_shape = (n_rep, n_tar)
    ind_rep, ind_tar = np.unravel_index(range(batch_size),batch_shape)

    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

    tar1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar/(n_tar-1)
    tar2_strengths = 2 - tar1_strengths

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs,
              'tar1_strengths' : tar1_strengths,
              'tar2_strengths' : tar2_strengths,
              'tar_time'    : 10000}

    return params, batch_size


with Run(save_addon) as R:
    config = R.config

    params, batch_size = gen_taskparams(n_tar=6, n_rep=1, tar_str_range=0.4)

    X = np.zeros((batch_size, 2)) # regressors

    task  = generate_onebatch(rule, R.config, 'psychometric', params=params, noise_on=False)
    # Only study target epoch
    epoch = task.epochs['tar1']
    h_sample = R.f_h(task.x)
    y_sample = R.f_y(h_sample)
    y_sample_loc = R.f_y_loc(y_sample)

    perfs = get_perf(y_sample, task.y_loc)
    # y_choice is 1 for choosing tar1_loc, otherwise -1
    y_choice = 2*(get_dist(y_sample_loc[-1]-tar1_loc)<np.pi/2) - 1

    h_sample = h_sample[epoch[0]:epoch[1],...][::50,...] # every 50 ms

    H = h_sample
    Perfs = perfs

    tar_cohs = params['tar1_strengths'] - params['tar2_strengths']
    X = np.array([y_choice, tar_cohs]).T

    if subtract_t_mean:
        H = f_sub_mean(H)

# Include only active units
nt, nb, nh = H.shape
h = H.reshape((-1, nh))
ind_active = np.where(h.var(axis=0) > 1e-4)[0]
ind_orig   = np.arange(nh)[ind_active]
h = h[:, ind_active]

# Z-score response (will have a strong impact on results)
if z_score:
    h = h - h.mean(axis=0)
    h = h/h.std(axis=0)

h = h.reshape((nt, nb, h.shape[-1]))


################################### Regression ################################
from sklearn import linear_model
# Create linear regression object

# Train the model using the training sets
Y = np.swapaxes(h, 0, 1)
Y = Y.reshape((Y.shape[0], -1))
regr = linear_model.LinearRegression()
regr.fit(X, Y)
coef = regr.coef_
coef = coef.reshape((h.shape[0],h.shape[2],X.shape[1])) # Time, Units, Coefs
# Get coeff at time when norm is maximum
coef_maxt = np.zeros((h.shape[2],X.shape[1])) # Units, Coefs
for i in range(h.shape[2]):
    ind = np.argmax(np.sum(coef[:,i,:]**2,axis=1))
    coef_maxt[i, :] = coef[ind,i,:]
# Orthogonalize
q, r = np.linalg.qr(coef_maxt)

h_tran = np.dot(h, q) # Transform


################ Pretty Plotting of State-space Results #######################
plot_onlycorrect = False # Only plotting correct trials
fs = 6

Perfs = Perfs.astype(bool)

fig = plt.figure(figsize=(2,2))
ax = fig.add_axes([0.2,0.2,0.7,0.7])

sep_by = Mod1
ch_list = [-1,1]
colors = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=6)
for i, s in enumerate(np.unique(X[:,sep_by])):
    for ch in ch_list: # Choice
        ind = (X[:,sep_by]==s)*(X[:,0]==ch)
        if plot_onlycorrect:
            ind *= Perfs

        if np.any(ind):
            h_plot = h_tran[:,ind,:]

            h_plot = h_plot.mean(axis=1)
            ax.plot(h_plot[:,0], h_plot[:,1],
                    '.-', markersize=2, color=colors[i], markeredgewidth=0.2)
############################### Find slow points ##############################

# Choosing starting points
params, batch_size = gen_taskparams(n_tar=6, n_rep=1, tar_str_range=0.2)
# Find slow points
# Generate coherence 0 inputs
# This has to be different from Mante et al. 2013,
# because our inputs are always positive, and can appear at different locations
pnt_trans = list()
task  = generate_onebatch(rule, config, 'psychometric', noise_on=False, params=params)
ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),(6,6))
for i in range(batch_size):
    res = find_slowpoints(save_addon, input=task.x[1000,i,:])
    print(res.success, res.fun)
    pnt_tran = np.dot(res.x[ind_active], q) # Transform
    pnt_trans.append(pnt_tran)
pnt_trans = np.array(pnt_trans)
ax.plot(pnt_trans[:,Choice], pnt_trans[:,Mod1], 'o', markersize=2, color='red')

plt.savefig('figure/fixpoint_simplechoice_statespace'+save_addon+'.pdf', transparent=True)
plt.show()




