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


save_addon = 'tf_latest_400'
rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]

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


with Run(save_addon) as R:

    n_tar = 6
    n_rep = 10
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


############################### Find slow points ##############################
def find_slowpoints(save_addon, rule, x0_list):
    print('Findings slow points...')
    # Generate coherence 0 inputs
    # This has to be different from Mante et al. 2013,
    # because our inputs are always positive, and can appear at different locations
    params = {'tar1_locs' : [0],
              'tar2_locs' : [np.pi],
              'tar1_mod1_strengths' : [1.2],
              'tar2_mod1_strengths' : [0.8],
              'tar1_mod2_strengths' : [0.8],
              'tar2_mod2_strengths' : [1.2],
              'tar_time'    : 1600}

    with Run(save_addon) as R:
        w_rec = R.w_rec
        w_in  = R.w_in
        b_rec = R.b_rec
        task  = generate_onebatch(rule, R.config, 'psychometric', noise_on=False, params=params)

    nh = len(b_rec)
    # Add constant input to baseline
    b_rec = b_rec + np.dot(w_in, task.x[1000,0,:])

    def dgdx(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        dfdx = 1/(1+1/expy)
        return (-F + np.dot(w_rec.T, F*dfdx))/nh

    def g(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        return np.mean(F**2)/2

    # It's 10% faster to provide ejac and fun separately
    # Default seeting Newton-CG is 100% slower but more accurate than L-BFGS-B
    # But much much better than SLSQP
    res_list = list()
    for x0 in x0_list:
        # res = minimize(g, x0, method='Newton-CG', jac=dgdx)
        res = minimize(g, x0, method='L-BFGS-B', jac=dgdx,
                       bounds=[(0,100)]*nh, options={'ftol':1e-20, 'gtol': 1e-7})
        # ftol may be important for how slow points are
        # If I pick gtol=1e-7, ftol=1e-20. Then regardless of starting points
        # I find only one fixed point, which depends on the input to the network
        #
        res_list.append(res)
    return res_list



################ Pretty Plotting of State-space Results #######################
plot_eachcurve = False
plot_onlycorrect = True # Only plotting correct trials
fs = 6

Perfs = Perfs.astype(bool)

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
            colors = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=len(np.unique(X[:,sep_by])))
        elif separate_by == 'tar1_mod2_strengths':
            sep_by = Mod2
            colors = sns.diverging_palette(145, 280, sep=1, s=99, l=30, n=len(np.unique(X[:,sep_by])))
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

                if np.any(ind):
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
                    if i_row == 0:
                        if i_col == 0:
                            # Choosing starting points
                            # x0_list = list()
                            # for k in range(10):
                            #     x0 = np.zeros(nh)
                            #     x0[ind_active] = q[:,0] * k
                            #     x0_list.append(x0)

                            tmp = h[:, ind, :].mean(axis=1)
                            tmp = tmp[::3,:]
                            x0_list = np.zeros((tmp.shape[0], nh))
                            x0_list[:, ind_active] = tmp

                            x0_list = np.random.rand(10, nh)*4

                            # Find slow points
                            res_list = find_slowpoints(save_addon, rule, x0_list)
                            pnt_trans = list()
                            for res, x0 in zip(res_list, x0_list):
                                print(res.fun)
                                x0_tran = np.dot(x0[ind_active], q)
                                pnt_tran = np.dot(res.x[ind_active], q) # Transform
                                # axarr[0, i_col].plot([x0_tran[Choice],pnt_tran[Choice]],
                                #                      [x0_tran[Mod1],pnt_tran[Mod1]],
                                #                      '+-', markersize=1, mew=0.2, lw=0.25, color='red')
                                # axarr[1, i_col].plot([x0_tran[Choice],pnt_tran[Choice]],
                                #                      [x0_tran[Mod2],pnt_tran[Mod2]],
                                #                      '+-', markersize=1, mew=0.2, lw=0.25, color='red')
                                pnt_trans.append(pnt_tran)

                            pnt_trans = np.array(pnt_trans)
                            axarr[0, i_col].plot(pnt_trans[:,Choice], pnt_trans[:,Mod1], '+', markersize=1, mew=0.2, color='red')
                            axarr[1, i_col].plot(pnt_trans[:,Choice], pnt_trans[:,Mod2], '+', markersize=1, mew=0.2, color='red')


plt.tight_layout(pad=0.0)

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

plt.savefig('figure/temp_choicetasks_statespace'+save_addon+'.pdf', transparent=True)
plt.show()




