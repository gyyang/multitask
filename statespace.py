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
# from slowpoints import find_slowpoints


save_addon = 'tf_latest_500'
rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]

# Regressors
Choice, Mod1, Mod2, Rule_mod1, Rule_mod2, Rule_int = range(6)
regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule attend 1', 'Rule attend 2', 'Rule int']

z_score = True


tar1_loc  = 0

def gen_taskparams(n_tar, n_rep, tar_str_range):
    batch_size = n_rep * n_tar**2
    batch_shape = (n_rep, n_tar, n_tar)
    ind_rep, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

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
    return params, batch_size


with Run(save_addon) as R:
    config = R.config

    params, batch_size = gen_taskparams(n_tar=6, n_rep=10, tar_str_range=0.2)

    X = np.zeros((len(rules)*batch_size, 3+3)) # regressors
    trial_rules = np.zeros(len(rules)*batch_size)
    H = np.array([])
    Perfs = np.array([])
    for i, rule in enumerate(rules):
        print('Starting standard analysis of the '+rule_name[rule]+' task...')
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
            Perfs = perfs
        else:
            H = np.concatenate((H, h_sample), axis=1)
            Perfs = np.concatenate((Perfs, perfs))

        tar1_mod1_cohs = params['tar1_mod1_strengths'] - params['tar2_mod1_strengths']
        tar1_mod2_cohs = params['tar1_mod2_strengths'] - params['tar2_mod2_strengths']
        X[i*batch_size:(i+1)*batch_size, :3] = np.array([y_choice, tar1_mod1_cohs, tar1_mod2_cohs]).T
        X[i*batch_size:(i+1)*batch_size, 3+i] = 1
        trial_rules[i*batch_size:(i+1)*batch_size] = rule

# Include only active units
nt, nb, nh = H.shape
h = H.reshape((-1, nh))
ind_active = np.where(h.var(axis=0) > 1e-4)[0]
ind_orig   = np.arange(nh)[ind_active]
h = h[:, ind_active]

# Z-score response (will have a strong impact on results)
if z_score:
    meanh = h.mean(axis=0)
    stdh = h.std(axis=0)
    h = h - meanh
    h = h/stdh

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


####################### Find Fixed & Slow Points ##############################
def find_slowpoints(save_addon, input, start_points=None, find_fixedpoints=True, dtype='float64'):
    if find_fixedpoints:
        # Finding fixed points require less tolerange
        print('Findings fixed points...')
        options = {'ftol':1e-10, 'gtol': 1e-10} # for L-BFGS-B
        # options = {'ftol':1e-7, 'gtol': 1e-7} # for L-BFGS-B
        # options = {'xtol': 1e-10}
    else:
        # Finding slow points allow more tolerange
        print('Findings slow points...')
        options = {'ftol':1e-4, 'gtol': 1e-4} # for L-BFGS-B
        # options = {'xtol': 1e-5}
    with Run(save_addon) as R:
        w_rec = R.w_rec.astype(dtype)
        w_in  = R.w_in.astype(dtype)
        b_rec = R.b_rec.astype(dtype)

    nh = len(b_rec)
    # Add constant input to baseline
    input = input.astype(dtype)
    b_rec = b_rec + np.dot(w_in, input)

    def dgdx(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        dfdx = 1/(1+1/expy)
        return (-F + np.dot(w_rec.T, F*dfdx))

    def g(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        return np.sum(F**2)/2

    if start_points is None:
        start_points = [np.ones(nh)]

    res_list = list()
    for start_point in start_points:
        start_point = start_point.astype(dtype)
        # res = minimize(g, start_point, method='Newton-CG', jac=dgdx, options=options)
        res = minimize(g, start_point, method='L-BFGS-B', jac=dgdx,
                       bounds=[(0,100)]*nh, options=options)
        # ftol may be important for how slow points are
        # If I pick gtol=1e-7, ftol=1e-20. Then regardless of starting points
        # I find only one fixed point, which depends on the input to the network
        res_list.append(res)
    return res_list

# Find Fixed points
# Choosing starting points
fixed_points_trans_all = dict()
slow_points_trans_all  = dict()
for rule in rules:
    params = {'tar1_locs' : [0],
              'tar2_locs' : [np.pi],
              'tar1_mod1_strengths' : [1],
              'tar2_mod1_strengths' : [1],
              'tar1_mod2_strengths' : [1],
              'tar2_mod2_strengths' : [1],
              'tar_time'    : 1600}
    print(rule_name[rule])
    task  = generate_onebatch(rule, config, 'psychometric', noise_on=False, params=params)

    #tmp = np.array([h[-1, X[:,0]==ch, :].mean(axis=0) for ch in [1, -1]])
    tmp = list()
    for ch in [-1, 1]:
        h_tmp = h[-1, X[:,0]==ch, :]
        ind = np.argmax(np.sum(h_tmp**2, 1))
        tmp.append(h_tmp[ind, :])
    tmp = np.array(tmp)

    if z_score:
        tmp *= stdh
        tmp += meanh
    start_points = np.zeros((tmp.shape[0], nh))
    start_points[:, ind_active] = tmp

    # Find fixed points
    res_list = find_slowpoints(save_addon, task.x[1000,0,:],
                               start_points=start_points, find_fixedpoints=True)
    fixed_points_raws  = list()
    fixed_points_trans = list()
    for i, res in enumerate(res_list):
        print(res.success, res.message, res.fun)
        fixed_points_raws.append(res.x)
        # fixed_points = start_points[i,ind_active]
        fixed_points = res.x[ind_active]
        if z_score:
            fixed_points -= meanh
            fixed_points /= stdh
        fixed_points_tran = np.dot(fixed_points, q)
        fixed_points_trans.append(fixed_points_tran)

    fixed_points_raws  = np.array(fixed_points_raws)
    fixed_points_trans = np.array(fixed_points_trans)
    fixed_points_trans_all[rule] = fixed_points_trans

    # Find slow points
    # The starting conditions will be equally sampled points in between two fixed points
    n_slow_points = 100 # actual points will be this minus 1
    mix_weight = np.array([np.arange(1,n_slow_points), n_slow_points-np.arange(1,n_slow_points)], dtype='float').T/n_slow_points

    # start_points = np.dot(mix_weight, fixed_points_raws)
    start_points = np.dot(mix_weight, start_points)

    # start_points+= np.random.randn(*start_points.shape) # Randomly perturb starting points
    # start_points *= np.random.uniform(0, 2, size=start_points.shape) # Randomly perturb starting points

    # start_points = np.random.rand(100, nh)*3

    res_list = find_slowpoints(save_addon, task.x[1000,0,:],
                               start_points=start_points, find_fixedpoints=False)

    slow_points_trans = list()
    for i, res in enumerate(res_list):
        # print(res.fun)
        # slow_points = start_points[i,ind_active]
        slow_points = res.x[ind_active]
        if z_score:
            slow_points -= meanh
            slow_points /= stdh
        slow_points_tran = np.dot(slow_points, q)
        slow_points_trans.append(slow_points_tran)
    slow_points_trans = np.array(slow_points_trans)
    slow_points_trans_all[rule] = slow_points_trans

################ Pretty Plotting of State-space Results #######################
plot_eachcurve = False
plot_onlycorrect = True # Only plotting correct trials
fs = 6

Perfs = Perfs.astype(bool)

colors1 = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=6)
colors2 = sns.diverging_palette(280, 145, sep=1, s=99, l=30, n=6)

fig, axarr = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(3,2))
for i_col, rule in enumerate(rules):
    for i_row in range(2):
        ax = axarr[i_row, i_col]
        ch_list = [-1,1]
        if i_row == 0:
            pcs = [Choice, Mod1] # Choice, Mod1
            separate_by = 'tar1_mod1_strengths'
            ax.set_title(rule_name[rule], fontsize=fs, y=0.8)
        else:
            pcs = [Choice, Mod2] # Choice, Mod1
            separate_by = 'tar1_mod2_strengths'
        ax.set_ylim([h_tran[:,:,pcs[1]].min(),h_tran[:,:,pcs[1]].max()])
        ax.set_xlim([h_tran[:,:,pcs[0]].min(),h_tran[:,:,pcs[0]].max()])

        if separate_by == 'tar1_mod1_strengths':
            sep_by = Mod1
            colors = colors1
        elif separate_by == 'tar1_mod2_strengths':
            sep_by = Mod2
            colors = colors2
        else:
            raise ValueError

        ax.axis('off')
        if i_col == 0:
            anc = [h_tran[:,:,pcs[0]].min()+1, h_tran[:,:,pcs[1]].max()-5] # anchor point
            ax.plot([anc[0], anc[0]], [anc[1]-5, anc[1]-1], color='black', lw=1.0)
            ax.plot([anc[0]+1, anc[0]+5], [anc[1], anc[1]], color='black', lw=1.0)
            ax.text(anc[0], anc[1], regr_names[pcs[0]], fontsize=fs, va='bottom')
            ax.text(anc[0], anc[1], regr_names[pcs[1]], fontsize=fs, rotation=90, ha='right', va='top')

        for i, s in enumerate(np.unique(X[:,sep_by])):
            for ch in ch_list: # Choice
                if ch == -1:
                    kwargs = {'markerfacecolor' : colors[i], 'linewidth' : 1}
                else:
                    kwargs = {'markerfacecolor' : 'white', 'linewidth' : 0.5}
                ind = (X[:,sep_by]==s)*(trial_rules==rule)*(X[:,0]==ch)
                if plot_onlycorrect:
                    ind *= Perfs

                if not np.any(ind):
                    continue

                h_plot = h_tran[:,ind,:]
                if plot_eachcurve:
                    for j in range(h_plot.shape[1]):
                        ax.plot(h_plot[:,j,pcs[0]], h_plot[:,j,pcs[1]],
                                '.-', markersize=2, color=colors[i])
                else:
                    h_plot = h_plot.mean(axis=1)
                    ax.plot(h_plot[:,pcs[0]], h_plot[:,pcs[1]],
                            '.-', markersize=2, color=colors[i], markeredgewidth=0.2, **kwargs)

                # Compute and plot slow points
                ax.plot(slow_points_trans_all[rule][:,pcs[0]],
                        slow_points_trans_all[rule][:,pcs[1]],
                        '+', markersize=1, mew=0.2, color=sns.xkcd_palette(['magenta'])[0])

                ax.plot(fixed_points_trans_all[rule][:,pcs[0]],
                        fixed_points_trans_all[rule][:,pcs[1]],
                        'x', markersize=2, mew=0.5, color=sns.xkcd_palette(['red'])[0])



plt.tight_layout(pad=0.0)

# Plot labels
for i_row in range(2):
    if i_row == 0:
        ax = fig.add_axes([0.25,0.45,0.2,0.1])
        colors = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=6)
    else:
        ax = fig.add_axes([0.25,0.05,0.2,0.1])
        colors = sns.diverging_palette(145, 280, sep=1, s=99, l=30, n=6)

    for i in range(6):
        kwargs = {'markerfacecolor' : colors[i], 'linewidth' : 1}
        ax.plot([i], [0], '.-', color=colors[i], markersize=4, markeredgewidth=0.5, **kwargs)
    ax.axis('off')
    ax.text(2.5, 1, 'Strong Weak Strong', fontsize=5, va='bottom', ha='center')
    ax.text(2.5, -1, 'To choice 1    To choice 2', fontsize=5, va='top', ha='center')
    ax.set_xlim([-1,6])
    ax.set_ylim([-3,3])

plt.savefig('figure/fixpoint_choicetasks_statespace'+save_addon+'.pdf', transparent=True)
plt.show()




