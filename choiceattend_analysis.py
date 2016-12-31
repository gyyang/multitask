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
from slowpoints import find_slowpoints

def gen_taskparams(tar1_loc, n_tar, n_rep):
    batch_size = n_rep * n_tar**2
    batch_shape = (n_rep, n_tar, n_tar)
    ind_rep, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

    tar1_locs = np.ones(batch_size)*tar1_loc
    tar2_locs = (tar1_locs+np.pi) % (2*np.pi)

    # tar_cohs = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])*0.3
    tar_cohs = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])*0.5
    tar_mod1_cohs = np.array([tar_cohs[i] for i in ind_tar_mod1])
    tar_mod2_cohs = np.array([tar_cohs[i] for i in ind_tar_mod2])

    params = {'tar1_locs' : tar1_locs,
              'tar2_locs' : tar2_locs,
              'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
              'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
              'tar1_mod2_strengths' : 1 + tar_mod2_cohs,
              'tar2_mod2_strengths' : 1 - tar_mod2_cohs,
              'tar_time'    : 1600}
              # If tar_time is long (~1600), we can reproduce the curving trajectories
    return params, batch_size

class ChoiceAttAnalysis(object):
    def __init__(self, save_addon, **kwargs):
        default_setting = {
            'save_addon'         : save_addon,
            'analyze_threerules' : False,
            'analyze_allunits'   : False,
            'redefine_choice'    : True,
            'regress_product'    : False, # regression of interaction terms
            'z_score'            : True,
            'fast_eval'          : False}

        self.setting = default_setting
        for key, val in kwargs.iteritems():
            self.setting[key] = val
        print('Current analysis setting:')
        for key, val in default_setting.iteritems():
            print(key + ' : ' + str(val))

        if self.setting['analyze_threerules']:
            self.rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT]
            # Regressors
            self.regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule attend 1', 'Rule attend 2', 'Rule int']
        else:
            self.rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
            # Regressors
            self.regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']
        n_rule = len(self.rules)
        n_regr = len(self.regr_names)

        tar1_loc  = 0


        with Run(save_addon, sigma_rec=0, fast_eval=self.setting['fast_eval']) as R:
        # with Run(save_addon) as R:
            self.config = R.config
            w_out  = R.w_out
            w_in   = R.w_in
            w_rec  = R.w_rec

            params, batch_size = gen_taskparams(tar1_loc, n_tar=6, n_rep=1)
            self.params = params

            X = np.zeros((n_rule*batch_size, n_regr)) # regressors
            trial_rules = np.zeros(n_rule*batch_size)
            H = np.array([])
            Perfs = np.array([]) # Performance

            for i, rule in enumerate(self.rules):
                print('Starting standard analysis of the '+rule_name[rule]+' task...')
                task  = generate_onebatch(rule, R.config, 'psychometric', params=params, noise_on=True)
                # Only study target epoch
                epoch = task.epochs['tar1']
                h_sample = R.f_h(task.x)
                y_sample = R.f_y(h_sample)
                y_sample_loc = R.f_y_loc(y_sample)

                perfs = get_perf(y_sample, task.y_loc)
                # y_choice is 1 for choosing tar1_loc, otherwise -1
                y_choice = 2*(get_dist(y_sample_loc[-1]-tar1_loc)<np.pi/2) - 1

                every_t = int(50/self.config['dt'])

                h_sample = h_sample[epoch[0]:epoch[1],...][::every_t,...] # every 50 ms
                if i == 0:
                    H = h_sample
                    Perfs = perfs
                else:
                    H = np.concatenate((H, h_sample), axis=1)
                    Perfs = np.concatenate((Perfs, perfs))

                tar_mod1_cohs = params['tar1_mod1_strengths'] - params['tar2_mod1_strengths']
                tar_mod2_cohs = params['tar1_mod2_strengths'] - params['tar2_mod2_strengths']
                X[i*batch_size:(i+1)*batch_size, :3] = \
                    np.array([y_choice, tar_mod1_cohs/tar_mod1_cohs.max(), tar_mod2_cohs/tar_mod2_cohs.max()]).T
                if self.setting['analyze_threerules']:
                    X[i*batch_size:(i+1)*batch_size, 3+i] = 1
                else:
                    X[i*batch_size:(i+1)*batch_size, 3] = 1-i*2
                trial_rules[i*batch_size:(i+1)*batch_size] = rule

        self.X = X
        self.H = H
        self.performances = Perfs
        self.trial_rules = trial_rules

        # Get preferences (+1 if activity is higher for choice=+1)
        preferences = (H[:, X[:,0]== 1, :].mean(axis=(0,1)) >
                       H[:, X[:,0]==-1, :].mean(axis=(0,1)))*2-1

        # Include only active units
        nt, nb, nh = H.shape # time, batch, hidden units
        h = H.reshape((-1, nh))

        if self.setting['analyze_allunits']:
            ind_active = range(nh)
        else:
            ind_active = np.where(h.var(axis=0) > 1e-4)[0]


        h = h[:, ind_active]
        preferences = preferences[ind_active]

        w_in  = w_in[ind_active, :]
        w_rec = w_rec[ind_active, :][:, ind_active]
        w_out = w_out[:, ind_active]

        # Z-score response (will have a strong impact on results)
        if self.setting['z_score']:
            self.meanh = h.mean(axis=0)
            self.stdh = h.std(axis=0)
            h = h - self.meanh
            h = h/self.stdh

        h = h.reshape((nt, nb, h.shape[-1]))


        # Temporarily disabled
        # Load clustering results
        # Get task-specific units for Att1 and Att2
        # data_type = 'rule'
        # with open('data/variance'+data_type+save_addon+'.pkl','rb') as f:
        #     res = pickle.load(f)
        # # Normalize by the total variance across tasks
        # res['h_normvar_all'] = (res['h_var_all'].T/np.sum(res['h_var_all'], axis=1)).T
        # res['ind_active'] = ind_active # temp
        # self.res = res

        # thre = 0.5 # threshold for variance ratio
        # thre = 0.8
        # ind_specifics = dict()
        # ind_specifics[-1] = range(self.h.shape[-1]) # The rest
        # for rule in [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]:
        #     h_normvar_all = res['h_normvar_all'][:, res['keys'].index(rule)]
        #     # Find indices and convert them
        #     ind_tmp = np.where(h_normvar_all>thre)[0]
        #     ind_tmp_active = res['ind_active'][ind_tmp]
        #     ind_specifics[rule] = list()
        #     for i in ind_tmp_active:
        #         j = np.where(ind_active==i)[0]
        #         if len(j) > 0:
        #             ind_specifics[rule].append(j[0])
        #             ind_specifics[-1].pop(ind_specifics[-1].index(j[0]))



        ################################### Regression ################################
        from sklearn import linear_model
        # Looping over units
        # Although this is slower, it's still quite fast, and more flexible
        coef_maxt = np.zeros((h.shape[2], X.shape[1])) # Units, Coefs
        for i in range(h.shape[-1]):
            # For each unit, relabel choice, mod 1,2 such that preferred choice is 1
            Xi = X.copy() # slicing method doesn't work for numpy array
            if self.setting['redefine_choice']:
                Xi[:, :3] = Xi[:, :3]*preferences[i]
            # Notice this also has to be taken care of when plotting state-space

            Y = np.swapaxes(h[:,:,i], 0, 1)
            regr = linear_model.LinearRegression() # Create linear regression object

            if self.setting['regress_product']:
                Xi_ = np.zeros((Xi.shape[0], n_regr + n_regr*(n_regr-1)/2))
                Xi_[:, :n_regr] = Xi
                k = 0
                for l in range(n_regr-1):
                    for m in range(l+1, n_regr):
                        Xi_[:, n_regr+k] = Xi[:, l] * Xi[:, m]
                        k += 1
            else:
                Xi_ = Xi

            regr.fit(Xi_, Y)

            coef = regr.coef_
            ind = np.argmax(np.sum(coef**2,axis=1))
            coef_maxt[i, :] = coef[ind,:]



        # Orthogonalize
        q, r = np.linalg.qr(coef_maxt)

        self.ind_active = ind_active
        self.h = h
        self.preferences = preferences
        self.coef_maxt = coef_maxt
        self.q = q
        self.h_tran = np.dot(h, q) # Transform

    def find_slowpoints(self):
    ####################### Find Fixed & Slow Points ##############################
        nt, nb, nh = self.H.shape # time, batch, hidden units
        # Find Fixed points
        # Choosing starting points
        self.fixed_points_trans_all = dict()
        self.slow_points_trans_all  = dict()
        for rule in self.rules:
            params = {'tar1_locs' : [0],
                      'tar2_locs' : [np.pi],
                      'tar1_mod1_strengths' : [1],
                      'tar2_mod1_strengths' : [1],
                      'tar1_mod2_strengths' : [1],
                      'tar2_mod2_strengths' : [1],
                      'tar_time'    : 600}
            print(rule_name[rule])
            task  = generate_onebatch(rule, self.config, 'psychometric', noise_on=False, params=params)
            epoch = task.epochs['tar1']

            #tmp = np.array([h[-1, X[:,0]==ch, :].mean(axis=0) for ch in [1, -1]])
            tmp = list()
            for ch in [-1, 1]:
                h_tmp = self.h[-1, self.X[:,0]==ch, :]
                ind = np.argmax(np.sum(h_tmp**2, 1))
                tmp.append(h_tmp[ind, :])
            tmp = np.array(tmp)

            if self.setting['z_score']:
                tmp *= self.stdh
                tmp += self.meanh
            start_points = np.zeros((tmp.shape[0], nh))
            print(start_points.shape)
            start_points[:, self.ind_active] = tmp

            # Find fixed points
            res_list = find_slowpoints(save_addon, task.x[epoch[1]-1,0,:],
                                       start_points=start_points, find_fixedpoints=True)
            fixed_points_raws  = list()
            fixed_points_trans = list()
            for i, res in enumerate(res_list):
                print(res.success, res.message, res.fun)
                fixed_points_raws.append(res.x)
                # fixed_points = start_points[i,ind_active]
                fixed_points = res.x[self.ind_active]
                if self.setting['z_score']:
                    fixed_points -= self.meanh
                    fixed_points /= self.stdh
                fixed_points_tran = np.dot(fixed_points, self.q)
                fixed_points_trans.append(fixed_points_tran)

            fixed_points_raws  = np.array(fixed_points_raws)
            fixed_points_trans = np.array(fixed_points_trans)
            self.fixed_points_trans_all[rule] = fixed_points_trans

            # Find slow points
            # The starting conditions will be equally sampled points in between two fixed points
            n_slow_points = 100 # actual points will be this minus 1
            mix_weight = np.array([np.arange(1,n_slow_points), n_slow_points-np.arange(1,n_slow_points)], dtype='float').T/n_slow_points

            # start_points = np.dot(mix_weight, fixed_points_raws)
            start_points = np.dot(mix_weight, start_points)

            # start_points+= np.random.randn(*start_points.shape) # Randomly perturb starting points
            # start_points *= np.random.uniform(0, 2, size=start_points.shape) # Randomly perturb starting points

            # start_points = np.random.rand(100, nh)*3

            res_list = find_slowpoints(save_addon, task.x[epoch[1]-1,0,:],
                                       start_points=start_points, find_fixedpoints=False)

            slow_points_trans = list()
            for i, res in enumerate(res_list):
                # print(res.fun)
                # slow_points = start_points[i,ind_active]
                slow_points = res.x[self.ind_active]
                if self.setting['z_score']:
                    slow_points -= self.meanh
                    slow_points /= self.stdh
                slow_points_tran = np.dot(slow_points, self.q)
                slow_points_trans.append(slow_points_tran)
            slow_points_trans = np.array(slow_points_trans)
            self.slow_points_trans_all[rule] = slow_points_trans

    def plot_statespace(self, plot_slowpoints=True):
        # TODO: Need to take into account the flipping of choice labeling
        if self.setting['redefine_choice']:
            raise ValueError('Redefine choice not supported for state space plots yet.')

        if plot_slowpoints:
            try:
                _ = self.slow_points_trans_all
            except AttributeError:
                self.find_slowpoints()

        ################ Pretty Plotting of State-space Results #######################
        plot_onlycorrect = True # Only plotting correct trials
        fs = 6

        Perfs = self.performances.astype(bool)

        colors1 = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=6)
        colors2 = sns.diverging_palette(280, 145, sep=1, s=99, l=30, n=6)

        fig, axarr = plt.subplots(2, len(self.rules), sharex=True, sharey='row', figsize=(len(self.rules)*1,2))
        for i_col, rule in enumerate(self.rules):
            for i_row in range(2):
                ax = axarr[i_row, i_col]
                ch_list = [-1,1]
                if i_row == 0:
                    pcs = [0, 1] # Choice, Mod1
                    sep_by = 1 # Separate by Mod 1
                    colors = colors1
                    ax.set_title(rule_name[rule], fontsize=fs, y=0.8)
                else:
                    pcs = [0, 2] # Choice, Mod1
                    sep_by = 2 # Separate by Mod 2
                    colors = colors2

                # ax.set_ylim([self.h_tran[:,:,pcs[1]].min(),self.h_tran[:,:,pcs[1]].max()])
                # ax.set_xlim([self.h_tran[:,:,pcs[0]].min(),self.h_tran[:,:,pcs[0]].max()])
                ax.axis('off')


                if i_col == 0:
                    anc = [self.h_tran[:,:,pcs[0]].min()+1, self.h_tran[:,:,pcs[1]].max()-5] # anchor point
                    ax.plot([anc[0], anc[0]], [anc[1]-5, anc[1]-1], color='black', lw=1.0)
                    ax.plot([anc[0]+1, anc[0]+5], [anc[1], anc[1]], color='black', lw=1.0)
                    ax.text(anc[0], anc[1], self.regr_names[pcs[0]], fontsize=fs, va='bottom')
                    ax.text(anc[0], anc[1], self.regr_names[pcs[1]], fontsize=fs, rotation=90, ha='right', va='top')

                # Loop over coherences of Mod 1 or 2
                for i, s in enumerate(np.unique(self.X[:,sep_by])):
                    for ch in ch_list: # Choice
                        if ch == 1:
                            kwargs = {'markerfacecolor' : 'white', 'linewidth' : 0.5}
                        else:
                            kwargs = {'markerfacecolor' : colors[i], 'linewidth' : 1}

                        if self.setting['redefine_choice']:
                            h_plot = np.zeros((self.h.shape[0],self.h.shape[2])) # Time, Regress
                            # If choices are redefined, then we need to get the
                            # activity of units that prefer original tar1 and tar2 separately
                            ind1 = (self.X[:,sep_by]== s)*(self.trial_rules==rule)*(self.X[:,0]== ch) # for batch
                            if plot_onlycorrect:
                                ind1 *= Perfs

                            if np.any(ind1):
                                # Average across these batches
                                tmp = self.h[:,ind1,:].mean(axis=1)
                                h_plot[:, self.preferences== 1] = tmp[:, self.preferences== 1]

                            ind2 = (self.X[:,sep_by]==-s)*(self.trial_rules==rule)*(self.X[:,0]==-ch) # for batch
                            if plot_onlycorrect:
                                ind2 *= Perfs

                            if np.any(ind2):
                                tmp = self.h[:,ind2,:].mean(axis=1)
                                h_plot[:, self.preferences==-1] = tmp[:, self.preferences==-1]

                            if not (np.any(ind1) or np.any(ind2)):
                                continue

                        else:
                            ind = (self.X[:,sep_by]==s)*(self.trial_rules==rule)*(self.X[:,0]==ch) # for batch
                            if plot_onlycorrect:
                                ind *= Perfs

                            if not np.any(ind):
                                continue

                            h_plot = self.h[:,ind,:].mean(axis=1)

                        h_plot = np.dot(h_plot, self.q) # Transform

                        ax.plot(h_plot[:,pcs[0]], h_plot[:,pcs[1]],
                                '.-', markersize=2, color=colors[i], markeredgewidth=0.2, **kwargs)

                        if not plot_slowpoints:
                            continue

                        # Compute and plot slow points
                        ax.plot(self.slow_points_trans_all[rule][:,pcs[0]],
                                self.slow_points_trans_all[rule][:,pcs[1]],
                                '+', markersize=1, mew=0.2, color=sns.xkcd_palette(['magenta'])[0])

                        ax.plot(self.fixed_points_trans_all[rule][:,pcs[0]],
                                self.fixed_points_trans_all[rule][:,pcs[1]],
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

    def plot_betaweights(self):
        ############################## Plot beta weights ##############################
        # Plot beta weights
        # coef_ = coef.reshape((-1, 6))
        coef_ = self.coef_maxt


        # # Plot all comparisons
        # fig, axarr = plt.subplots(6, 6, sharex='col', sharey='row', figsize=(4,4))
        # colors = sns.xkcd_palette(['grey', 'red', 'blue'])
        # # colors = sns.xkcd_palette(['red'] * 3)
        # fs = 6
        # for i in range(6):
        #     for j in range(6):
        #         ax = axarr[i, j]
        #         for rule, c in zip([-1, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2], colors):
        #             ind = ind_specifics[rule]
        #             ax.plot(coef_[ind,j], coef_[ind, i], 'o', color=c, ms=1, mec='white', mew=0.15)
        #         ax.axis('off')
        #         if i == 5:
        #             med_x = np.median(coef_[:,j])
        #             ran_x = (np.max(coef_[:,j]) - np.min(coef_[:,j]))/5
        #             bot_y = np.min(coef_[:,i])
        #             ax.plot([med_x-ran_x, med_x+ran_x], [bot_y, bot_y], color='black', lw=1.0)
        #             ax.text(med_x, bot_y, regr_names[j], fontsize=fs, va='top', ha='center')
        #         if j == 0:
        #             med_y = np.median(coef_[:,i])
        #             ran_y = (np.max(coef_[:,i]) - np.min(coef_[:,i]))/5
        #             left_x = np.min(coef_[:,j])
        #             ax.plot([left_x, left_x], [med_y-ran_y, med_y+ran_y], color='black', lw=1.0)
        #             ax.text(left_x, med_y, regr_names[i], fontsize=fs, rotation=90, ha='right', va='center')
        # plt.savefig('figure/beta_weights_full.pdf', transparent=True)
        # plt.show()

        if self.setting['analyze_threerules']:
            n_regr = 4
        else:
            n_regr = 3

        # Plot important comparisons
        fig, axarr = plt.subplots(1, n_regr, sharex='col', sharey='row', figsize=(n_regr*2,2))
        colors = sns.xkcd_palette(['grey', 'red', 'blue'])
        # colors = sns.xkcd_palette(['red'] * 3)
        fs = 6
        i = 0
        for j in range(1, 1+n_regr):
            ax = axarr[j-1]

            # for rule, c in zip([-1, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2], colors):
            #     ind = ind_specifics[rule]
            #     ax.plot(coef_[ind,j], coef_[ind, i], 'o', color=c, ms=1.5, mec='white', mew=0.2)

            ax.plot(coef_[:,j], coef_[:, i], 'o', color='black', ms=1.5, mec='white', mew=0.2)

            # ax.axis('off')

            # med_x = np.median(coef_[:,j])
            # ran_x = (np.max(coef_[:,j]) - np.min(coef_[:,j]))/5
            # bot_y = np.min(coef_[:,i])
            # ax.plot([med_x-ran_x, med_x+ran_x], [bot_y, bot_y], color='black', lw=1.0)
            # ax.text(med_x, bot_y, self.regr_names[j], fontsize=fs, va='top', ha='center')

            # if j == 0:
            #     med_y = np.median(coef_[:,i])
            #     ran_y = (np.max(coef_[:,i]) - np.min(coef_[:,i]))/5
            #     left_x = np.min(coef_[:,j])
            #     ax.plot([left_x, left_x], [med_y-ran_y, med_y+ran_y], color='black', lw=1.0)
            #     ax.text(left_x, med_y, self.regr_names[i], fontsize=fs, rotation=90, ha='right', va='center')
            ax.set_xlabel(self.regr_names[j], fontsize=fs)
            if j == 1:
                ax.set_ylabel(self.regr_names[i], fontsize=fs)

            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.locator_params(nbins=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        plt.tight_layout()

        plt.savefig('figure/beta_weights_sub.pdf', transparent=True)
        plt.show()

    def temp(self):
        ######################### Study connection weights ##############################
        w_in_specifics = w_in[ind_specifics[CHOICEATTEND_MOD2], :]
        n_ring = config['N_RING']
        rules = [FIXATION, GO, INHGO, DELAYGO,\
            CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
            CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,\
            REMAP, INHREMAP, DELAYREMAP,\
            DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]

        fs = 6
        fig = plt.figure()
        plt.plot(w_in_specifics[:, 2*n_ring+1:].mean(axis=0), 'o-')
        plt.xticks(rules, [rule_name[rule] for rule in rules], rotation=90, fontsize=fs)
        plt.show()

        fig = plt.figure()
        _ = plt.plot(w_in_specifics[:, 1:n_ring+1].T)
        plt.show()

        fig = plt.figure()
        _ = plt.plot(w_in_specifics[:, n_ring+1:2*n_ring+1].T)
        plt.show()

        ############################### Study variance ################################
        data_type = 'rule'
        with open('data/variance'+data_type+save_addon+'.pkl','rb') as f:
            res = pickle.load(f)
        # Normalize by the total variance across tasks
        res['h_normvar_all'] = (res['h_var_all'].T/np.sum(res['h_var_all'], axis=1)).T
        res['ind_active'] = np.arange(nh) # temp


        for rule in [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]:
            h_normvar_all = res['h_normvar_all'][:, res['keys'].index(rule)]

        plt.hist(res['h_normvar_all'][:, res['keys'].index(CHOICEATTEND_MOD1)], bins=20)
        plt.show()

        rule = DELAYREMAP
        fig = plt.figure(figsize=(1.5,1.2))
        ax = fig.add_axes([0.3,0.3,0.6,0.5])
        hist, bins_edge = np.histogram(res['h_normvar_all'][:, res['rules'].index(rule)], bins=30, range=(0,1))
        ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
               color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
        plt.xlim([-0.05, 1.05])
        plt.ylim([hist.max()*-0.05, hist.max()*1.1])
        plt.xlabel(r'Variance ratio', fontsize=7, labelpad=1)
        plt.ylabel('counts', fontsize=7)
        plt.locator_params(nbins=3)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # plt.savefig('figure/hist_totalvar.pdf', transparent=True)
        plt.show()


# save_addon = 'allrule_weaknoise_300'
# caa = ChoiceAttAnalysis(save_addon, analyze_threerules=False,
#                         analyze_allunits=False, fast_eval=True, redefine_choice=False)
# caa.plot_betaweights()
# caa.plot_statespace(plot_slowpoints=True)


fast_eval = True
data_type  = 'rule'
save_addon = 'allrule_weaknoise_300'
fname = 'data/variance'+data_type+save_addon
with open(fname+'.pkl','rb') as f:
    res = pickle.load(f)
h_var_all = res['h_var_all']
keys      = res['keys']

rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
ind_rules = [keys.index(rule) for rule in rules]
h_var_all = h_var_all[:, ind_rules]

# First only get active units. Total variance across tasks larger than 1e-3
ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
h_var_all  = h_var_all[ind_active, :]

# Normalize by the total variance across tasks
h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T


ind_1  = np.where(h_normvar_all[:,0]>0.8)[0]
ind_12 = np.where((h_normvar_all[:,0]>0.4)*(h_normvar_all[:,0]<0.6))[0]
ind_2  = np.where(h_normvar_all[:,0]<0.2)[0]

ind_1_orig  = ind_active[ind_1]
ind_12_orig = ind_active[ind_12]
ind_2_orig  = ind_active[ind_2]


perfs, perfs_lesion = list(), list()
for lesion_units, perfs_store in zip([None, ind_2_orig],
                                     [perfs, perfs_lesion]):
    with Run(save_addon, lesion_units=lesion_units, fast_eval=fast_eval) as R:
        config = R.config
        for rule in rules:
            task = generate_onebatch(rule=rule, config=config, mode='test')
            y_hat = R.f_y_from_x(task.x)
            perf = get_perf(y_hat, task.y_loc)
            perfs_store.append(perf.mean())


perfs, perfs_lesion = np.array(perfs), np.array(perfs_lesion)

fs = 6
width = 0.3
fig = plt.figure(figsize=(1.5,1.2))
ax = fig.add_axes([0.3,0.3,0.6,0.4])
b0 = ax.bar(np.arange(2)-width, [perfs[0], perfs[1]],
       width=width, color=sns.xkcd_palette(['orange'])[0], edgecolor='none')
b1 = ax.bar(np.arange(2), [perfs_lesion[0], perfs_lesion[1]],
       width=width, color=sns.xkcd_palette(['green'])[0], edgecolor='none')
ax.set_xticks(np.arange(2))
ax.set_xticklabels([rule_name[r] for r in rules])
ax.set_xlabel('Tasks', fontsize=fs, labelpad=3)
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
# if save:
#     plt.savefig('figure/perflesion_remap_units'+save_addon+'.pdf', transparent=True)
plt.show()