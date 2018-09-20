"""Analysis of the context-dependent decision making tasks"""

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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import tensorflow as tf
from task import generate_trials, get_dist
from network import Model

# Constants
CONTEXTS   = (1, -1)                # Two contexts, attend to mod 1 & 2 respectively
MOD1COHIDS = MOD2COHIDS = (3, 2, 1, -1, -2, -3)  # IDs of mod 1 and 2 coherences

################################ Helper functions #############################

def get_active_units(H, setting):
    '''Analyze only active units'''

    if setting['analyze_allunits']:
        # Analyze all units with non-zero variance
        ind_active = np.where(H[-1].var(axis=0) > 0)[0] # current setting
    else:
        # The way to select these units are important
        ind_active = np.where(H[-1].var(axis=0) > 1e-3)[0] # current setting
        # ind_active = np.where(H[-1].var(axis=0) > 0)[0] # current setting
        # ind_active = np.where(H.reshape((-1, nh)).var(axis=0) > 1e-10)[0]
        # ind_active = np.where(H.var(axis=1).mean(axis=0) > 1e-4)[0]

    H = H[..., ind_active]

    return H, ind_active


def z_score(H):
    '''z-score the activity for each unit'''
    nt, nb, nh = H.shape
    # Transform to matrix
    H = H.reshape((-1, nh))
    # z-score
    meanh = H.mean(axis=0)
    stdh  = H.std(axis=0)
    H = H - meanh
    H = H/stdh
    # Transform back
    H = H.reshape((nt, nb, nh))
    return H, meanh, stdh


def get_preferences(H, y_target_choice, y_actual_choice):
    '''The preferences of neurons'''
    # method of the Mante task
    preferences = (H[:, y_actual_choice== 1, :].mean(axis=(0,1)) >
                   H[:, y_actual_choice==-1, :].mean(axis=(0,1)))*2-1

    # preferences = (H[:, (y_actual_choice== 1)*(perfs==1), :].mean(axis=(0,1)) >
    #                H[:, (y_actual_choice==-1)*(perfs==1), :].mean(axis=(0,1)))*2-1


    # preferences = (H[:, y_target_choice== 1, :].mean(axis=(0,1)) >
    #                H[:, y_target_choice==-1, :].mean(axis=(0,1)))*2-1

    # preferences = (H[-1, y_actual_choice== 1, :].mean(axis=0) >
    #                H[-1, y_actual_choice==-1, :].mean(axis=0))*2-1

    # preferences = (H[-1, y_target_choice== 1, :].mean(axis=0) >
    #                H[-1, y_target_choice==-1, :].mean(axis=0))*2-1

    return preferences

##################### Running Mante task simulation ###########################

def gen_taskparams(stim1_loc, n_stim, n_rep=1, n_stimloc=10):
    if stim1_loc is not None:
        # Do not loop over stim1_loc
        # Generate task parameterse for state-space analysis
        batch_size = n_rep * n_stim**2
        batch_shape = (n_stim, n_stim, n_rep)
        ind_stim_mod1, ind_stim_mod2, ind_rep = np.unravel_index(range(batch_size),batch_shape)

        stim1_locs = np.ones(batch_size)*stim1_loc
    else:
        # Loop over stim1_loc
        batch_size = n_rep * n_stim**2 * n_stimloc
        batch_shape = (n_stim, n_stim, n_stimloc, n_rep)
        ind_stim_mod1, ind_stim_mod2, ind_stimloc, ind_rep = \
            np.unravel_index(range(batch_size),batch_shape)

        stim1_locs = 2*np.pi*ind_stimloc/n_stimloc

    stim2_locs = (stim1_locs+np.pi) % (2*np.pi)

    stim_cohs = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])*0.4
    # IDs for the coherences. No need to be tied to actual coherences
    stim_coh_ids   = np.array([-3, -2, -1, +1, +2, +3]) # integers, useful for comparison
    stim_coh_norms = stim_cohs/stim_cohs.max() # normalize
    stim_mod1_cohs = np.array([stim_cohs[i] for i in ind_stim_mod1])
    stim_mod2_cohs = np.array([stim_cohs[i] for i in ind_stim_mod2])

    params = {'stim1_locs' : stim1_locs,
              'stim2_locs' : stim2_locs,
              'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
              'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
              'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
              'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
              'mod1_coh_ids' : np.array([stim_coh_ids[i] for i in ind_stim_mod1]),
              'mod2_coh_ids' : np.array([stim_coh_ids[i] for i in ind_stim_mod2]),
              'mod1_coh_norms' : np.array([stim_coh_norms[i] for i in ind_stim_mod1]),
              'mod2_coh_norms' : np.array([stim_coh_norms[i] for i in ind_stim_mod2]),
              'stim_time'    : 750} # 750 ms the same as Mante et al. 2013

    return params, batch_size


def _run_simulation(model, setting):
    '''Generate simulation data for all trials'''

    config = model.config

    # Get rules and regressors
    rules = ['contextdm1', 'contextdm2']
    # regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']

    n_rule = len(rules)

    stim1_loc_list = np.arange(12)/12.*2*np.pi # loop over different stimulus location

    #################### Generate all trials ##############################
    Data = list()

    for stim1_loc in stim1_loc_list:
        # Generate task parameters used
        params, batch_size = gen_taskparams(stim1_loc, n_stim=6, n_rep=setting['n_rep'])

        # Because looping over two rules
        stim1_locs = np.tile(params['stim1_locs'], n_rule)
        mod1_coh_ids = np.tile(params['mod1_coh_ids'], n_rule)
        mod2_coh_ids = np.tile(params['mod2_coh_ids'], n_rule)
        mod1_coh_norms = np.tile(params['mod1_coh_norms'], n_rule)
        mod2_coh_norms = np.tile(params['mod2_coh_norms'], n_rule)
        assert n_rule == 2
        contexts = np.repeat([1, -1], batch_size) # +1 for Att 1, -1 for Att 2

        x     = list() # Network input
        y_loc = list() # Network target output location

        for rule in rules:
            # Generating task information
            trial = generate_trials(rule, config, 'psychometric',
                                      params=params, noise_on=False)
            x.append(trial.x)
            y_loc.extend(trial.y_loc[-1]) # target location at last time point

        x     = np.concatenate(x, axis=1)
        y_loc = np.array(y_loc)

        # Get neural activity
        H = model.get_h(x)
        # Actual response location at last time point
        y_hat_loc = model.get_y_loc(model.get_y_from_h(H))[-1]

        # Analyze activity
        Activity = H
        # Activity = x

        # Only analyze the target epoch
        epoch = trial.epochs['stim1']
        # It's important to get rid of the initial transient
        # Otherwise, rule inputs will dominate, because rules are always there
        t_start, t_end = epoch[0]+int(setting['t_ignore']/config['dt']), epoch[1]
        # t_start, t_end = epoch[0], epoch[1]
        Activity = Activity[t_start:t_end,...]

        if ('record_dt' in setting) and setting['record_dt'] != config['dt']:
            # Downsample in time with record_dt
            assert setting['record_dt'] > config['dt']
            record_every = int(setting['record_dt']/config['dt'])
            Activity = Activity[::record_every,...]


        # Analyze active units
        Activity, ind_active = get_active_units(Activity, setting) # get active units

        # Z-scoring response across time and trials (can have a strong impact on results)
        if setting['z_score']:
            # meanh and stdh need to be recorded for slow-point analysis
            Activity, meanh, stdh = z_score(Activity)


        # Get actual and target choice for each trial
        # y_choice is 1 for choosing stim1_loc, otherwise -1
        y_actual_choice = 2*(get_dist(y_hat_loc-stim1_locs)<np.pi/2) - 1
        y_target_choice = 2*(get_dist(y_loc    -stim1_locs)<np.pi/2) - 1
        correct         = (y_actual_choice==y_target_choice).astype(int)
        if setting['redefine_choice']:
            # Get neuronal preferences (+1 if activity is higher for actual_choice=+1 than -1)
            preferences = get_preferences(Activity, y_target_choice, y_actual_choice)
        else:
            preferences = np.ones(Activity.shape[-1])

        assert setting['n_rep'] == 1 # Otherwise averaging should be done beforehand


        for neuron in range(Activity.shape[-1]):
            Data.append({
                'Activity'          : Activity[:,:,neuron].T, # (n_trial, n_timepoint)
                'Context'           : contexts,
                # Redefine choice, (+1 for the preferred choice)
                'ActualChoiceRedef' : preferences[neuron]*y_actual_choice,
                'Mod1CohID'         : preferences[neuron]*mod1_coh_ids,
                'Mod2CohID'         : preferences[neuron]*mod2_coh_ids,
                'Mod1Coh'           : preferences[neuron]*mod1_coh_norms,
                'Mod2Coh'           : preferences[neuron]*mod2_coh_norms,
                'ActualChoice'      : y_actual_choice,
                'TargetChoice'      : y_target_choice,
                'Correct'           : correct,
            })


    return Data


def run_simulation(save_name, setting):
    '''Generate simulation data for all trials'''
    tf.reset_default_graph()
    model = Model(save_name, sigma_rec=setting['sigma_rec'], dt=10)

    with tf.Session() as sess:
        model.restore(sess)
        Data = _run_simulation(model, setting)

    return Data

#################### Preprocessing ############################################

def regression(Data):
    '''Linear regression of activities using four regressors'''

    from sklearn import linear_model

    regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Context']
    regr_keys  = ['ActualChoiceRedef', 'Mod1Coh', 'Mod2Coh', 'Context']
    n_regr = len(regr_names)


    # Time-independent coefficient vectors (n_unit, n_regress)
    coef_maxt = np.zeros((len(Data), n_regr))

    # Looping over units
    # Although this is slower, it's still quite fast, and it's clearer and more flexible
    for neuron, data in enumerate(Data):
        # Regressors
        regrs = np.array([data[key] for key in regr_keys]).T

        # Linear regression
        regr_model = linear_model.LinearRegression()
        regr_model.fit(regrs, data['Activity'])

        # Get time-independent coefficient vector
        coef = regr_model.coef_
        ind = np.argmax(np.sum(coef**2, axis=1))
        coef_maxt[neuron, :] = coef[ind,:]

    # Orthogonalize with QR decomposition
    # Matrix q represents the orthogonalized task-related axes
    coef_maxt_ortho, R = np.linalg.qr(coef_maxt)
    coef_maxt_ortho *= np.sign(np.diag(R)) # this will standardize the axis directions
    return coef_maxt_ortho


def get_cond_ind(data, cond):
    '''
    Get indices that satisfy the given condition
    Each condition is given by a sequence (Choice, Mod1CohID, Mod2CohID, Context, Correct)
    Each element of the sequence can be None, 'pos', 'neg', or a value

    Return a numpy array of booleans
    '''
    n_trial = len(data['Mod1CohID'])
    cond_ind = np.ones(n_trial, dtype=bool) # initialize

    for c, key in zip(cond, ['ActualChoiceRedef', 'Mod1CohID', 'Mod2CohID', 'Context', 'Correct']):
        if c is None:
            # include
            pass
        elif c in ['pos', 'positive']:
            cond_ind *= (data[key] > 0)
        elif c in ['neg', 'negative']:
            cond_ind *= (data[key] < 0)
        else:
            cond_ind *= (data[key] == c)

    return cond_ind


def get_cond_ind_16_dim(data, cond):#maddy
    '''
    Get indices that satisfy the given condition
    Each condition is given by a sequence (Mod1Cohsign, Mod2Cohsign, Context, Correct)
    Each element of the sequence can be +1 or -1
    Return a numpy array of booleans
    '''
    n_trial = len(data['Mod1CohID'])
    cond_ind = np.ones(n_trial, dtype=bool) # initialize

    for c, key in zip(cond, ['Mod1CohID', 'Mod2CohID', 'Context', 'Correct']):
        if c > 0:
            cond_ind *= (data[key] > 0)
        elif c < 0:
            cond_ind *= (data[key] < 0)
        else:#c == 0
            cond_ind *= (data[key] == c)

    return cond_ind


def get_conditions():
    '''
    Get conditions for the state space analysis

    Four types of sorting.
    For each context, by three relevant and six irrelevant conditions.
    Correct trials only.
    '''

    # Each condition is given by a sequence (Choice, Mod1Coh, Mod2Coh, Context, Correct)
    conds = list()
    conds.extend([(np.sign(mod1coh), mod1coh, None   , +1, +1) for mod1coh in MOD1COHIDS])
    conds.extend([(+1              , None   , mod2coh, +1, +1) for mod2coh in MOD2COHIDS])
    conds.extend([(-1              , None   , mod2coh, +1, +1) for mod2coh in MOD2COHIDS])
    conds.extend([(np.sign(mod2coh), None   , mod2coh, -1, +1) for mod2coh in MOD2COHIDS])
    conds.extend([(+1              , mod1coh, None   , -1, +1) for mod1coh in MOD1COHIDS])
    conds.extend([(-1              , mod1coh, None   , -1, +1) for mod1coh in MOD1COHIDS])

    return conds


def get_cond_16_dim():#maddy. 
    '''
    Get conditions for dim checking - map all trials to 16 bins. 

    Four types of sorting.
    '''
    bv = (+1, -1)
    # Each condition is given by a sequence (Mod1Cohsign, Mod2Cohsign, Context, Correct)
    conds = list()
    conds.extend([(np.sign(mod1coh), +1   , +1, +1) for mod1coh in bv])
    conds.extend([(np.sign(mod1coh), -1   , +1, +1) for mod1coh in bv])    
    conds.extend([(np.sign(mod1coh), +1   , +1, 0) for mod1coh in bv])
    conds.extend([(np.sign(mod1coh), -1   , +1, 0) for mod1coh in bv])    
    conds.extend([(np.sign(mod1coh), +1   , -1, +1) for mod1coh in bv])
    conds.extend([(np.sign(mod1coh), -1   , -1, +1) for mod1coh in bv])    
    conds.extend([(np.sign(mod1coh), +1   , -1, 0) for mod1coh in bv])
    conds.extend([(np.sign(mod1coh), -1   , -1, 0) for mod1coh in bv])    
    
    return conds


def condition_averaging(Data, conds, flatten=False):
    '''
    Condition-based averaging of data

    Args:
        Data : list of dictionary
        conds : list of conditions to average the data for
        For each condition, average across all trials with this condition

    Returns:
        if flatten: a matrix (n_time*n_condition, n_neuron)
        if not flatten: a matrix (n_time, n_condition, n_neuron)
    '''

    n_time = Data[0]['Activity'].shape[1]
    n_cond = len(conds)
    n_neuron = len(Data)
    # Condition-averaged response
    Data_avgcond = np.zeros((n_time, n_cond, n_neuron))

    for neuron, data in enumerate(Data):#data is list of dict. 
        for i_cond, cond in enumerate(conds): 
            
            cond_ind = get_cond_ind(data, cond)  
            
            if np.sum(cond_ind)>0:
                # Some neurons have no trials for particular conditions
                # In this case, the condition-averaged activity is kept at 0
                Data_avgcond[:, i_cond, neuron] = data['Activity'][cond_ind, :].mean(axis=0)

    if flatten:
        # Flatten first two dimensions
        Data_avgcond = Data_avgcond.reshape((-1, Data_avgcond.shape[-1]))

    return Data_avgcond
    

def condition_averaging_split_trte(Data, conds):#maddy changed. split train, test.     
    '''
    Condition-based averaging of data

    Args:
        Data : list of dictionary
        conds : list of conditions to average the data for
        For each condition, average across all trials with this condition

    Returns:
        2 matrices - train and test of shape (n_time, n_condition, n_neuron)
    '''

    n_time = Data[0]['Activity'].shape[1]
    n_cond = len(conds)
    n_neuron = len(Data)
    # Condition-averaged response
    Data_avgcond = np.zeros((n_time, n_cond, n_neuron))
    Data_avgcond_tr = np.zeros((n_time, n_cond, n_neuron)) #maddy below 2 lines. 
    Data_avgcond_te = np.zeros((n_time, n_cond, n_neuron))

    for neuron, data in enumerate(Data):#data is list of dict. 
        for i_cond, cond in enumerate(conds): 
            #maddy. list of 16 conds - each binary len 4 vector. ?            
            #cond_ind = get_cond_ind(data, cond) #old. 
            cond_ind = get_cond_ind_16_dim(data, cond) #maddy changed. 
            
            #maddy. for given neuron, do any of the trials satisfy given condition? 
            if np.sum(cond_ind)>0:
                # Some neurons have no trials for particular conditions
                # In this case, the condition-averaged activity is kept at 0
                Data_avgcond[:, i_cond, neuron] = data['Activity'][cond_ind, :].mean(axis=0)
                #avrg across conditions
                
                #look at trials that satisfy given condition - then split into train and test
                #70% train say. eventually get train and test matrices ntime, ncond, nunits. 
                #then implement rest compute lin classifier etc in dimensionality.
                #if split_traintest:
                p_train = .7
                cond_ind = np.where(cond_ind)[0]
                np.random.shuffle(cond_ind)

                n_trial_cond = len(cond_ind)
                n_trial_cond_train = int(n_trial_cond*p_train)
                #print neuron, i_cond, n_trial_cond_train, n_trial_cond
                
                ind_cond_train = cond_ind[:n_trial_cond_train]
                ind_cond_test  = cond_ind[n_trial_cond_train:]
                #only consider mean if slices are not empty. 
                if n_trial_cond_train < n_trial_cond and n_trial_cond_train > 0:
                    Data_avgcond_tr[:, i_cond, neuron] = np.mean(data['Activity'][ind_cond_train, :], axis=0)
                    Data_avgcond_te[:, i_cond, neuron] = np.mean(data['Activity'][ind_cond_test, :], axis=0)

    return Data_avgcond_tr, Data_avgcond_te

############################ Plotting #########################################

def plot_statespace(Data_proj, conds):
    '''Plotting the state space in the style of Mante 2013'''
    import seaborn.apionly as sns # If you don't have this, then some colormaps won't work

    ################ Pretty Plotting of State-space Results #######################
    fs = 6
    colors1 = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=6)
    colors2 = sns.diverging_palette(280, 145, sep=1, s=99, l=30, n=6)

    fig, axarr = plt.subplots(2, len(CONTEXTS),
                              sharex=True, sharey='row', figsize=(len(CONTEXTS)*1,2))

    for i_col, context in enumerate(CONTEXTS):
        for i_row in range(2):
            # Different ways of separation, either by Mod1 or Mod2
            # Also different subspaces shown

            ax = axarr[i_row, i_col]
            ax.axis('off')

            if i_row == 0:
                # Separate by coherence of Mod 1
                cohs = MOD1COHIDS

                # Show subspace (Choice, Mod1)
                pcs = [0, 1]

                # Color set
                colors = colors1

                # Rule title
                # ax.set_title(rule_name[rule], fontsize=fs, y=0.8)

            else:
                # Separate by coherence of Mod 2
                cohs = MOD2COHIDS

                # Show subspace (Choice, Mod2)
                pcs = [0, 2]

                # Color set
                colors = colors2


            # Scale bars
            # if i_col == 0:
            #     anc = [self.H_tran[:,:,pcs[0]].min()+1, self.H_tran[:,:,pcs[1]].max()-5] # anchor point
            #     ax.plot([anc[0], anc[0]], [anc[1]-5, anc[1]-1], color='black', lw=1.0)
            #     ax.plot([anc[0]+1, anc[0]+5], [anc[1], anc[1]], color='black', lw=1.0)
            #     ax.text(anc[0], anc[1], self.regr_names[pcs[0]], fontsize=fs, va='bottom')
            #     ax.text(anc[0], anc[1], self.regr_names[pcs[1]], fontsize=fs, rotation=90, ha='right', va='top')

            # ind = self.get_regr_ind(rule=rule) # for batch
            # ax.plot(self.H_tran[:,ind,pcs[0]], self.H_tran[:,ind,pcs[1]],
            #     '-', color='gray', alpha=0.5, linewidth=0.5)

            # Loop over coherences to choice 1, from high to low
            for i_coh, coh in enumerate(cohs):
                if i_row==i_col:
                    choices = [np.sign(coh)]
                else:
                    choices = [1, -1]
                # Loop over choices
                for choice in choices:

                    if choice == 1:
                        # Solid circles
                        kwargs = {'markerfacecolor' : list(colors[i_coh]), 'linewidth' : 1}
                    else:
                        # Empty circles
                        kwargs = {'markerfacecolor' : 'white', 'linewidth' : 0.5}

                    # Each condition is given by a sequence (Choice, Mod1Coh, Mod2Coh, Context, Correct)
                    if i_row == 0:
                        cond = (choice, coh, None, context, +1)
                    else:
                        cond = (choice, None, coh, context, +1)

                    i_cond = conds.index(cond)

                    ax.plot(Data_proj[:, i_cond, pcs[0]], Data_proj[:, i_cond, pcs[1]],
                            '.-', markersize=2, color=colors[i_coh], markeredgewidth=0.2, **kwargs)


                    # if not plot_slowpoints:
                    #     continue
                    #
                    # # Plot slow points
                    # ax.plot(self.slow_points_trans_all[rule][:,pcs[0]],
                    #         self.slow_points_trans_all[rule][:,pcs[1]],
                    #         '+', markersize=1, mew=0.2, color=sns.xkcd_palette(['magenta'])[0])
                    #
                    # ax.plot(self.fixed_points_trans_all[rule][:,pcs[0]],
                    #         self.fixed_points_trans_all[rule][:,pcs[1]],
                    #         'x', markersize=2, mew=0.5, color=sns.xkcd_palette(['red'])[0])

    plt.tight_layout(pad=0.0)

    # Plot labels
    for i_row in range(2):
        if i_row == 0:
            ax = fig.add_axes([0.25,0.45,0.2,0.1])
            colors = colors1
        else:
            ax = fig.add_axes([0.25,0.05,0.2,0.1])
            colors = colors2

        for i in range(6):
            kwargs = {'markerfacecolor' : list(colors[i]), 'linewidth' : 1}
            ax.plot([i], [0], '.-', color=colors[i], markersize=4, markeredgewidth=0.5, **kwargs)
        ax.axis('off')
        ax.text(2.5, 1, 'Strong Weak Strong', fontsize=5, va='bottom', ha='center')
        # During looping, we use coherence to choice 1 from high to low
        ax.text(2.5, -1, 'To choice 1    To choice 2', fontsize=5, va='top', ha='center')
        ax.set_xlim([-1,6])
        ax.set_ylim([-3,3])

    # if save:
    #     plt.savefig(os.path.join('figure',
    # 'fixpoint_choicetasks_statespace'+self.setting['save_name']+'.pdf'), transparent=True)
    # plt.show()


def plot_coef(coef):
    '''
    Plot regression coefficients (beta weights) against one another
    :return:
    '''

    regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']

    # Plot important comparisons
    fig, axarr = plt.subplots(3, 2, figsize=(4,5))
    fs = 6

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for i_plot in range(6):
        i, j = pairs[i_plot]
        ax = axarr[i_plot//2, i_plot%2]

        color = 'gray'
        # Find all units here that belong to group 1, 2, or 12 as defined in UnitAnalysis
        ax.plot(coef[:,i], coef[:,j], 'o', color=color, ms=1.5, mec='white', mew=0.2)

        # ax.plot([-2, 2], [0, 0], color='gray')
        # ax.plot([0, 0], [-2, 2], color='gray')
        ax.set_xlabel(regr_names[i], fontsize=fs)
        ax.set_ylabel(regr_names[j], fontsize=fs)
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])

        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.locator_params(nbins=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    plt.tight_layout()

    # if save:
    #     save_name = 'beta_weights_sub'
    #     plt.savefig(os.path.join('figure',save_name+self.setting['save_name']+'.pdf'), transparent=True)
    plt.show()


def compute_frac_var(Data):
    '''
    Compute the fractional variance based on the data
    This fractional variance is based on incomplete exploration of stimulus space
    similar to the actual neural recordings
    '''
    frac_var = list()
    for data in Data:
        # Variance across conditions, and average across time
        # TODO: may need to exclude early time points
        var1 = data['Activity'][data['Context']==+1, :].var(axis=0).mean()
        var2 = data['Activity'][data['Context']==-1, :].var(axis=0).mean()
        frac_var.append((var1-var2)/(var1+var2))

    return np.array(frac_var)


def plot_coefvsfracvar(coef, frac_var):
    '''Plot coefficients against the fractional variances'''
    from scipy import stats

    regr_names = ['Choice', 'Motion', 'Color', 'Rule']
    fig, axarr = plt.subplots(4, 1, figsize=(1.6,5), sharex=True)
    for i in range(4):
        ax = axarr[i]
        ax.scatter(frac_var, coef[:,i], s=3,
                   facecolor='black', edgecolor='none')

        slope, intercept, r_value, p_value, std_err = stats.linregress(frac_var, coef[:,i])
        ax.set_title('r = {:0.2f}, p = {:0.1E} '.format(r_value, p_value), fontsize=7)

        ax.set_xlim([-1.1,1.1])
        if i == 3:
            ax.set_xticks([-1,0,1])
            ax.set_xlabel('FracVar', fontsize=7)
        else:
            ax.set_xticks([-1,0,1],['']*3)
#==============================================================================
#         if i == 0:
#             # ax.set_ylim([-0.2,0.4])
#             ax.set_yticks([-0.2,0,0.4])
#         else:
#             # ax.set_ylim([-0.3,0.3])
#             # ax.set_yticks([-0.3,0,0.3])
#==============================================================================
        ax.set_ylabel(regr_names[i], fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='both', which='major', labelsize=7)

    plt.tight_layout()

    # save_name = 'fracVarvsBeta'
    # if analyze_single_units:
    #     save_name = save_name + '_SU'
    # else:
    #     save_name = save_name + '_AU'
    # if not denoise:
    #     save_name = save_name + '_nodenoise'
    # plt.savefig('figure/'+save_name+'.pdf', transparent=True)

##################### Comparison between model and data #######################

def get_default_condition():
    '''Get default condition for comparison

    Returns:
        conds : a list of conditions
        Each condition is given by a sequence (Choice, Mod1CohID, Mod2CohID, Context, Correct)
    '''

    contexts = [+1, -1]
    mod1cohids = [-3,-2,-1,1,2,3]
    mod2cohids = [-3,-2,-1,1,2,3]

    # mod1cohids = ['pos', 'neg']
    # mod2cohids = ['pos', 'neg']

    conds = list()
    for context in contexts:
        for mod1cohid in mod1cohids:
            for mod2cohid in mod2cohids:
                conds.append((None, mod1cohid, mod2cohid, context, None))

    return conds


def get_condavg_data(reuse=True):
    '''Get condition average from real data

    Args:
        reuse : if True, load from saved file. If False, recompute from data

    Returns:
        Matrix containing condition-averaged data
    '''

    if reuse:
        fname = os.path.join('data', 'ManteDataCond.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                Data_avgcond = pickle.load(f)
        else:
            return get_condavg_data(reuse=False)

    else:
        with open(os.path.join('data', 'ManteData.pkl'), 'rb') as f:
            Data = pickle.load(f)

        conds = get_default_condition()
        # Condition averaging, return (n_time, n_condition, n_neuron)
        Data_avgcond = condition_averaging(Data, conds, flatten=True)

        with open(os.path.join('data', 'ManteDataCond.pkl'), 'wb') as f:
            pickle.dump(Data_avgcond, f)

    return Data_avgcond


def get_condavg_simu(save_name):
    '''Get condition average from simulation

    Args:
        save_name : model name

    Returns:
        Matrix containing condition-averaged data
    '''

    setting = {
            'save_name'          : save_name,
            'analyze_allunits'   : False,
            'redefine_choice'    : True,
            'z_score'            : True,
            'surrogate_data'     : False,
            'sigma_rec'          : 0,
            'n_rep'              : 1,
            't_ignore'           : 0,
            'record_dt'          : 50 # ms, the same as Mante et al. 2013
            }

    DataSimu = run_simulation(save_name, setting)

    # Condition averaging, return (n_time*n_condition, n_neuron)
    conds = get_default_condition()
    DataSimu_avgcond = condition_averaging(DataSimu, conds, flatten=True)

    return DataSimu_avgcond


def get_condavg_simu_16_dim(save_name):#maddy changed. 
    '''Get condition average from simulation

    Args:
        save_name : model name

    Returns:
        Matrix containing condition-averaged data
    '''

    setting = {
            'save_name'          : save_name,
            'analyze_allunits'   : False,
            'redefine_choice'    : True,
            'z_score'            : True,
            'surrogate_data'     : False,
            'sigma_rec'          : 0,
            'n_rep'              : 1,
            't_ignore'           : 0,
            'record_dt'          : 50 # ms, the same as Mante et al. 2013
            }

    if save_name is not 'Data':
        DataSimu = run_simulation(save_name, setting)#if save_name isn't Data. 
    else:
        fname = os.path.join('data', 'ManteData.pkl')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                DataSimu = pickle.load(f) 

    # Condition averaging, return (n_time*n_condition, n_neuron)
    conds = get_cond_16_dim()
    #DataSimu_avgcond = condition_averaging(DataSimu, conds, flatten=True)
    DataSimu_train, DataSimu_test = condition_averaging_split_trte(DataSimu, conds)

    #return DataSimu_avgcond
    return DataSimu_train, DataSimu_test


def linear_fit(X, Y, dim_x=None):
    '''Get train and test score for linear regression of data

    Args:
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples, n_targets]
            Target values

        dim_x : dimension of X out of n_features used for the fit

    Returns:
        score_trains : numpy array of scores on training data
        score_tests : numpy array of scores on test data

    '''

    # pick dim_x features from x for regression
    ind_x = np.arange(X.shape[1])

    # Linear regression
    regr_model = LinearRegression()

    # Shuffle the features
    np.random.shuffle(ind_x)

    # Split train and test for cross-validation
    X_train, X_test, y_train, y_test = \
        train_test_split(X[:,ind_x[:dim_x]], Y, test_size=0.2)

    # Fit with model
    regr_model.fit(X_train, y_train)

    # Store scores for training and testing data
    score_train = regr_model.score(X_train, y_train)
    score_test  = regr_model.score(X_test, y_test)

    return score_train, score_test


def run_score(save_name, dim_x=300, n_rep=100):
    '''Get train and test score for linear regression of data

    Args:
        save_name : if 'data2data', compute the score from data to data
         otherwise, compute the score from simulating the model save_name to data

        dim_x : dimension of X out of n_features used for the fit

        n_rep : number of repetition of train/test splitting

    Returns:
        score_trains : numpy array of scores on training data
        score_tests : numpy array of scores on test data

    '''

    Data_avgcond = get_condavg_data(reuse=True) # a matrix (n_time*n_condition, n_neuron)

    if save_name == 'data2data':
        ind_x = np.arange(Data_avgcond.shape[1])
    else:
        DataSimu_avgcond = get_condavg_simu(save_name)
        Y = Data_avgcond # a matrix (n_time*n_condition, n_neuron)

        ind_x = np.arange(DataSimu_avgcond.shape[1])

    # Store results
    score_trains = np.zeros(n_rep)
    score_tests  = np.zeros(n_rep)

    # Linear regression
    regr_model = LinearRegression()

    for i_rep in range(n_rep):
        # Shuffle the features
        np.random.shuffle(ind_x)

        if save_name == 'data2data':
            X = Data_avgcond[:,ind_x[:dim_x]]
            Y = Data_avgcond[:,ind_x[dim_x:]]
        else:
            X = DataSimu_avgcond[:, ind_x[:dim_x]]

        # Split train and test for cross-validation
        X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.2)

        # Fit with model
        regr_model.fit(X_train, y_train)

        # Store scores for training and testing data
        score_train = regr_model.score(X_train, y_train)
        score_test  = regr_model.score(X_test, y_test)

        # Store results
        score_trains[i_rep] = score_train
        score_tests[i_rep] = score_test

    return score_trains, score_tests


def run_and_plot_score_vs_dimx(save_name):
    Data_avgcond = get_condavg_data(reuse=False)
    DataSimu_avgcond = get_condavg_simu(save_name)

    X, Y = DataSimu_avgcond, Data_avgcond
    # X, Y = Data_avgcond[:, :200], DataSimu_avgcond[:,-50:]
    # X, Y = Data_avgcond[:, :700], Data_avgcond[:,-50:]
    # X, Y = DataSimu_avgcond[:,:200], DataSimu_avgcond[:,-50:]

    dim_xs = np.arange(1, 200, 10)
    score_trains = list()
    score_tests = list()
    for dim_x in dim_xs:
        score_train, score_test = run_score(save_name, dim_x=dim_x)
        score_trains.append(score_train.mean())
        score_tests.append(score_test.mean())

    plt.figure()
    plt.plot(dim_xs, score_trains, 'red')
    plt.plot(dim_xs, score_tests, 'blue')
    plt.ylim([0, 1])
    plt.legend(['train', 'test'], title='error')
    plt.xlabel('Number of units used')
    plt.ylabel('r^2')

###############################################################################

def run_all_analyses(save_name):
    setting = {
            'save_name'          : save_name,
            'analyze_threerules' : False,
            'analyze_allunits'   : False,
            'redefine_choice'    : True,
            'regress_product'    : False, # regression of interaction terms
            'z_score'            : True,
            'fast_eval'          : True,
            'surrogate_data'     : False,
            'sigma_rec'          : 0,
            'select_group'       : False,
            'n_rep'              : 1,
            't_ignore'           : 0}

    rules = ['contextdm1', 'contextdm2']

    Data = run_simulation(save_name, setting)
    coef = regression(Data)
    conds = get_conditions()
    Data_avgcond = condition_averaging(Data, conds)
    Data_proj = np.dot(Data_avgcond, coef)
    plot_statespace(Data_proj, conds)
    plot_coef(coef)

    frac_var = compute_frac_var(Data)
    plot_coefvsfracvar(coef, frac_var)

    from variance import _plot_hist_varprop, _compute_hist_varprop, compute_variance
    hist, bins_edge = np.histogram(frac_var, bins=20, range=(-1,1))
    _plot_hist_varprop(hist, bins_edge, rules)

    # TODO: Check why the result is funny
    compute_variance(save_name, 'rule', rules)
    hist, bins_edge = _compute_hist_varprop(save_name, rules)
    _plot_hist_varprop(hist, bins_edge, rules)


if __name__ == '__main__':
    save_name = 'debug'

    # from train import to_savename
    # n_hidden=64
    # seed = 2
    # activation='tanh'
    # rnn_type='LeakyGRU'
    # training_iters   = 50000#150000
    # w_rec_init = 'diag' #maddy added below 5 lines.
    # l1_h        = 1.0*0.0001
    # l2_h        = 1.0*0
    # l1_weight   = 1.0*0.0001
    # l2_weight   = 0.0001*0
    # save_name = to_savename(n_hidden = n_hidden,  seed = seed, activation = activation,
    #       rnn_type = rnn_type, w_rec_init    = w_rec_init,  l1_h = l1_h,
    #       l2_h = l2_h, l1_weight   = l1_weight, l2_weight   = l2_weight)
    # #run_all_analyses(save_name)
    #
    # run_and_plot_score_vs_dimx(save_name)
    #
    # run_all_analyses(save_name)
    # # run_and_plot_score_vs_dimx(save_name)
