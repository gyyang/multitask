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
from network import get_perf

#==============================================================================
# mpl.rcParams['xtick.direction'] = 'out'
# mpl.rcParams['ytick.direction'] = 'out'
# 
#==============================================================================

save_addon = 'tf_latest_400'
rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
#rules = [CHOICEATTEND_MOD1]

def f_sub_mean(x):
    # subtract mean activity across batch conditions
    assert(len(x.shape)==3)
    x_mean = x.mean(axis=1)
    for i in range(x.shape[1]):
        x[:,i,:] -= x_mean
    return x

# Regressors
Choice, Mod1, Mod2, Rule_mod1, Rule_mod2, Rule_int = range(6)
regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule attend 1', 'Rule attend 2', 'Rule int']

subtract_t_mean=True
z_score = True

print('Starting standard analysis of the CHOICEATTEND task...')
with Run(save_addon) as R:

    n_tar = 6
    n_rep = 30
    batch_size = n_rep * n_tar**2
    batch_shape = (n_rep, n_tar, n_tar)
    ind_rep, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

    tar1_loc  = 0
    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

    tar_str_range = 0.2
    tar1_mod1_cohs = -tar_str_range/2+tar_str_range*ind_tar_mod1/(n_tar-1)
    tar1_mod2_cohs = -tar_str_range/2+tar_str_range*ind_tar_mod2/(n_tar-1)
    tar1_mod1_strengths = 1 + tar1_mod1_cohs
    tar2_mod1_strengths = 1 - tar1_mod1_cohs
    tar1_mod2_strengths = 1 + tar1_mod2_cohs
    tar2_mod2_strengths = 1 - tar1_mod2_cohs

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs,
              'tar1_mod1_strengths' : tar1_mod1_strengths,
              'tar2_mod1_strengths' : tar2_mod1_strengths,
              'tar1_mod2_strengths' : tar1_mod2_strengths,
              'tar2_mod2_strengths' : tar2_mod2_strengths,
              'tar_time'    : 1600}
              # If tar_time is long (~1600), we can reproduce the curving trajectories

    X = np.zeros((len(rules)*batch_size, 3+3)) # regressors
    trial_rules = np.zeros(len(rules)*batch_size)
    H = np.array([])
    for i, rule in enumerate(rules):
        task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
        # Only study target epoch
        epoch = task.epochs['tar1']
        h_sample = R.f_h(task.x)
        y_sample = R.f_y(h_sample)
        y_sample_loc = R.f_y_loc(y_sample)

        perfs = get_perf(y_sample, task.y_loc)
        # y_choice is 1 for choosing tar1_loc, otherwise -1
        y_choice = 2*(get_dist(y_sample_loc[-1]-tar1_loc)<np.pi/2) - 1

        h_sample = h_sample[epoch[0]:epoch[1],...][::50,...] # every 50 ms
        if i == 0:
            H = h_sample
        else:
            H = np.concatenate((H, h_sample), axis=1)

        X[i*batch_size:(i+1)*batch_size, :3] = np.array([y_choice, tar1_mod1_cohs, tar1_mod2_cohs]).T
        X[i*batch_size:(i+1)*batch_size, 3+i] = 1
        trial_rules[i*batch_size:(i+1)*batch_size] = rule
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

def show3D(h_tran, separate_by, pcs=(0,1,2), **kwargs):
    if separate_by == 'tar1_mod1_strengths':
        separate_bys = tar1_mod1_strengths
        colors = sns.color_palette("RdBu_r", len(np.unique(separate_bys)))
    elif separate_by == 'tar1_mod2_strengths':
        separate_bys = tar1_mod2_strengths
        colors = sns.color_palette("BrBG", len(np.unique(separate_bys)))
    else:
        raise ValueError

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, s in enumerate(np.unique(separate_bys)):
        h_plot = h_tran[:,separate_bys==s,:]
        for j in range(h_plot.shape[1]):
            ax.plot(h_plot[:,j,pcs[0]], h_plot[:,j,pcs[1]], h_plot[:,j,pcs[2]],
                    '.-', markersize=2, color=colors[i])
    if 'azim' in kwargs:
        ax.azim = kwargs['azim']
    if 'elev' in kwargs:
        ax.elev = kwargs['elev']
    ax.elev = 62
    plt.show()

def test_PCA_ring():
    # Test PCA with simple ring representation
    n_t = 3
    n_loc = 256
    n_ring = 64
    h_simple = np.zeros((n_t, n_loc, n_ring))

    pref = np.arange(0,2*np.pi,2*np.pi/n_ring) # preferences
    locs = np.arange(n_loc)/n_loc*2*np.pi
    ts   = np.arange(n_t)/n_t
    for i, loc in enumerate(locs):
        dist = get_dist(loc-pref) # periodic boundary
        dist /= np.pi/8
        h_simple[:,i,:] = 0.8*np.exp(-dist**2/2)
        h_simple[:,i,:] = (h_simple[:,i,:].T * ts).T

    from sklearn.decomposition import PCA
    pca = PCA()
    h_tran = pca.fit_transform(h_simple.reshape((-1, n_ring))).reshape((n_t, n_loc, n_ring))
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    colors = sns.color_palette("husl", n_loc)

    h_tran += np.random.randn(*h_tran.shape)*0.004

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([.2,.2,.7,.7])
    for i in range(n_loc):
        ax.plot(h_tran[:,i,0], h_tran[:,i,1], '-', linewidth=0.3, color=colors[i])
    ax.set_aspect('equal')
    plt.savefig('figure/temp.pdf')
    plt.show()

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_axes([.2,.2,.7,.7])
    ax.plot(evr, 'o-')
    plt.show()


# nt, nb, nh = h_samples[CHOICEATTEND_MOD1].shape
# from sklearn.decomposition import PCA
# pca = PCA()
# #pca.fit(h_samples[CHOICEATTEND_MOD1].reshape((-1, nh)))
# pca.fit(np.concatenate((h_samples[CHOICEATTEND_MOD1].reshape((-1, nh)),h_samples[CHOICEATTEND_MOD2].reshape((-1, nh))), axis=0))
#
# h_trans = dict()
# for rule in rules:
#     h_trans[rule] = pca.transform(h_samples[rule].reshape((-1, nh)))

#h_tran = h_trans[CHOICEATTEND_MOD1].reshape((nt, nb, -1))
#show3D(h_tran, 'tar1_mod1_strengths', azim=-62, elev=62)
#show3D(h_tran, 'tar1_mod2_strengths', azim=-62, elev=62)

#h_tran = h_trans[CHOICEATTEND_MOD2].reshape((nt, nb, -1))
#show3D(h_tran, 'tar1_mod1_strengths', azim=-62, elev=62)
#show3D(h_tran, 'tar1_mod2_strengths', azim=-62, elev=62)



# Regression

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

# Show 2-D
plot_eachcurve = False
fig, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(3,3))
for i_row, rule in enumerate(rules):
    for i_col in range(3):
        ax = axarr[i_row,i_col]
        ch_list = [-1,1]
        if i_col == 0:
            pcs = [0,1] # Choice, Mod1
            separate_by = 'tar1_mod1_strengths'
        elif i_col == 1:
            if i_row == 0:
                pcs = [0,2]
                separate_by = 'tar1_mod1_strengths'
            else:
                pcs = [0,1] # Choice, Mod1
                separate_by = 'tar1_mod2_strengths'
            ax.set_title(rule_name[rule], fontsize=7, y=0.9)
        else:
            pcs = [0,2] # Choice, Mod1
            separate_by = 'tar1_mod2_strengths'

        if separate_by == 'tar1_mod1_strengths':
            sep_by = Mod1
            colors = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=len(np.unique(X[:,sep_by])))
        elif separate_by == 'tar1_mod2_strengths':
            sep_by = Mod2
            colors = sns.diverging_palette(145, 280, sep=1, s=99, l=30, n=len(np.unique(X[:,sep_by])))
        else:
            raise ValueError

        ax.locator_params(nbins=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel(regr_names[pcs[0]], fontsize=7)
        ax.set_ylabel(regr_names[pcs[1]], fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)


        for i, s in enumerate(np.unique(X[:,sep_by])):
            for ch in ch_list: # Choice
                if ch == -1:
                    kwargs = {'markerfacecolor' : colors[i], 'linewidth' : 1}
                    #mfc=colors[i]
                else:
                    kwargs = {'markerfacecolor' : 'white', 'linewidth' : 0.5}
                    #mfc = 'white'
                ind = (X[:,sep_by]==s)*(trial_rules==rule)*(X[:,0]==ch)
                if np.any(ind):
                    h_plot = h_tran[:,ind,:]
                    if plot_eachcurve:
                        for j in range(h_plot.shape[1]):
                            axarr[i_row,i_col].plot(h_plot[:,j,pcs[0]], h_plot[:,j,pcs[1]],
                                    '.-', markersize=2, color=colors[i])
                    else:
                        h_plot = h_plot.mean(axis=1)
                        axarr[i_row,i_col].plot(h_plot[:,pcs[0]], h_plot[:,pcs[1]],
                                '.-', markersize=2, color=colors[i], markeredgewidth=0.2, **kwargs)
plt.tight_layout(pad=1.0)
plt.savefig('figure/choicetasks_statespace.pdf', transparent=True)
plt.show()

