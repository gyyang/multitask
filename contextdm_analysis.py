"""
Analysis of the choice att tasks
"""

from __future__ import division

import os
import pickle
import copy
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from task import generate_trials, rule_name, get_rule_index, get_dist
from network import Model
import tools
import performance
from slowpoints import search_slowpoints

save = True # TEMP
COLORS = sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])


def generate_surrogate_data():
    # Generate surrogate data

    # Number of time points
    nt = 20
    t_plot = np.linspace(0, 1, nt)

    # Performance
    perfs = np.ones(batch_size)

    # Generate choice
    rel_mod = '1' if rule == 'contextdm1' else '2' # relevant modality
    rel_coh = params['stim1_mod'+rel_mod+'_strengths']-params['stim2_mod'+rel_mod+'_strengths']
    y_choice = (rel_coh>0)*2-1

    # Generate underlying low-dimensional representation
    mod1_plot = np.ones((nt, batch_size)) * (params['stim1_mod1_strengths']-params['stim2_mod1_strengths'])
    mod2_plot = np.ones((nt, batch_size)) * (params['stim1_mod2_strengths']-params['stim2_mod2_strengths'])
    choice_plot = (np.ones((nt, batch_size)).T * t_plot).T  * y_choice

    # Generate surrogate neural activity
    h_sur = np.zeros((nt, batch_size, 3))
    h_sur[:, :, 0] = mod1_plot
    h_sur[:, :, 1] = mod2_plot
    h_sur[:, :, 2] = choice_plot

    # Random orthogonal projection
    h_sample = np.dot(h_sur, random_ortho_matrix[:3, :])
    return h_sample, y_choice, perfs


class UnitAnalysis(object):
    def __init__(self, model_dir):
        """Analyze based on units."""
        data_type  = 'rule'
        fname = os.path.join(model_dir, 'variance_' + data_type)
        with open(fname + '.pkl', 'rb') as f:
            res = pickle.load(f)
        h_var_all = res['h_var_all']
        keys      = res['keys']

        rules = ['contextdm1', 'contextdm2']
        ind_rules = [keys.index(rule) for rule in rules]
        h_var_all = h_var_all[:, ind_rules]

        # First only get active units. Total variance across tasks larger than 1e-3
        ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
        # ind_active = np.where(h_var_all.sum(axis=1) > 1e-1)[0] # TEMPORARY
        h_var_all  = h_var_all[ind_active, :]

        # Normalize by the total variance across tasks
        h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

        group_ind = dict()

        group_ind['1'] = np.where(h_normvar_all[:,0] > 0.9)[0]
        group_ind['2'] = np.where(h_normvar_all[:, 0] < 0.1)[0]
        group_ind['12'] = np.where(np.logical_and(h_normvar_all[:,0] > 0.4,
                                                  h_normvar_all[:,0] < 0.6))[0]
        group_ind['1+2'] = np.concatenate((group_ind['1'], group_ind['2']))

        group_ind_orig = {key: ind_active[val] for key, val in group_ind.items()}

        self.model_dir = model_dir
        self.group_ind = group_ind
        self.group_ind_orig = group_ind_orig
        self.h_normvar_all = h_normvar_all
        self.rules = rules
        self.ind_active = ind_active
        self.colors = dict(zip([None, '1', '2', '12'],
                               sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))
        self.lesion_group_names = {None : 'intact',
                                   '1'  : 'lesion groups 1',
                                   '2'  : 'lesion groups 2',
                                   '12' : 'lesion groups 12',
                                   '1+2': 'lesion groups 1 & 2'}

    def prettyplot_hist_varprop(self):
        """Pretty version of variance.plot_hist_varprop."""
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
            data_plot = self.h_normvar_all[self.group_ind[group], 0]
            hist, bins_edge = np.histogram(data_plot, bins=30, range=(0,1))
            b_tmp = ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
                   color=self.colors[group], edgecolor='none', label=group)
            bs.append(b_tmp)
        plt.locator_params(nbins=3)
        xlabel = 'FracVar({:s}, {:s})'.format(rule_name[rules[0]], rule_name[rules[1]])
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel('Units', fontsize=fs)
        ax.set_ylim(bottom=-0.02*hist.max())
        ax.set_xlim((-0.1, 1.1))
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
                '.pdf', transparent=True)

    def plot_connectivity(self, conn_type='input'):
        """Plot connectivity while sorting by group.

        Args:
            conn_type: str, type of connectivity to plot.
        """

        # Sort data by labels and by input connectivity
        model = Model(model_dir)
        hparams = model.hparams
        with tf.Session() as sess:
            model.restore()
            w_in, w_out, w_rec = sess.run(
                [model.w_in, model.w_out, model.w_rec])
        w_in, w_rec, w_out = w_in.T, w_rec.T, w_out.T

        n_ring = hparams['n_eachring']
        groups = ['1', '2', '12']

        if conn_type == 'rec':
            # Plot recurrent connectivity
            w_rec_group = np.zeros((len(groups), len(groups)))
            for i1, group1 in enumerate(groups):
                for i2, group2 in enumerate(groups):
                    ind1 = self.group_ind_orig[group1]
                    ind2 = self.group_ind_orig[group2]
                    w_rec_group[i2, i1] = w_rec[:, ind1][ind2, :].mean()

            fs = 6
            cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_axes([.2, .2, .6, .6])
            im = ax.imshow(w_rec_group, interpolation='nearest', cmap=cmap, aspect='auto')
            ax.axis('off')

            ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
            cb = plt.colorbar(im, cax=ax)
            cb.outline.set_linewidth(0.5)
            cb.set_label('Prop. of choice 1', fontsize=fs, labelpad=2)
            plt.tick_params(axis='both', which='major', labelsize=fs)

            return

        elif conn_type == 'rule':
            # Plot input rule connectivity
            rules = ['contextdm1', 'contextdm2', 'dm1', 'dm2', 'multidm']

            w_stores = OrderedDict()

            for group in groups:
                w_store_tmp = list()
                ind = self.group_ind_orig[group]
                for rule in rules:
                    ind_rule = get_rule_index(rule, hparams)
                    w_conn = w_in[ind, ind_rule].mean(axis=0)
                    w_store_tmp.append(w_conn)

                w_stores[group] = w_store_tmp

            fs = 6
            width = 0.15
            fig = plt.figure(figsize=(3,1.5))
            ax = fig.add_axes([0.17,0.35,0.8,0.4])
            for i, group in enumerate(groups):
                b0 = ax.bar(np.arange(len(rules))+(i-1.5)*width, w_stores[group],
                       width=width, color=self.colors[group], edgecolor='none')
            ax.set_xticks(np.arange(len(rules)))
            ax.set_xticklabels([rule_name[r] for r in rules], rotation=25)
            ax.set_xlabel('From rule input', fontsize=fs, labelpad=3)
            ax.set_ylabel('conn. weight', fontsize=fs)
            lg = ax.legend(groups, fontsize=fs, ncol=3, bbox_to_anchor=(1,1.4),
                           labelspacing=0.2, loc=1, frameon=False, title='To group')
            plt.setp(lg.get_title(),fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            plt.locator_params(axis='y',nbins=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.set_xlim([-0.8, len(rules)-0.2])
            ax.plot([-0.5, len(rules)-0.5], [0, 0], color='gray', linewidth=0.5)
            plt.savefig('figure/conn_'+conn_type+'_contextdm.eps', transparent=True)
            plt.show()

            return

        # Plot input from stim or output to loc
        elif conn_type == 'input':
            w_conn = w_in[:, 1:n_ring+1]
            xlabel = 'From stim mod 1'
            lgtitle = 'To group'
        elif conn_type == 'output':
            w_conn = w_out[1:, :].T
            xlabel = 'To output'
            lgtitle = 'From group'
        else:
            ValueError('Unknown conn type')

        w_aves = dict()

        for group in groups:
            ind_group  = self.group_ind_orig[group]
            n_group    = len(ind_group)
            w_group = np.zeros((n_group, n_ring))

            for i, ind in enumerate(ind_group):
                tmp           = w_conn[ind, :]
                ind_max       = np.argmax(tmp)
                w_group[i, :] = np.roll(tmp, int(n_ring/2)-ind_max)

            w_aves[group] = w_group.mean(axis=0)

        fs = 6
        fig = plt.figure(figsize=(1.5, 1.0))
        ax = fig.add_axes([.3, .3, .6, .6])
        for group in groups:
            ax.plot(w_aves[group], color=self.colors[group], label=group)
        ax.set_xticks([int(n_ring/2)])
        ax.set_xticklabels(['preferred loc.'])
        ax.set_xlabel(xlabel, fontsize=fs, labelpad=3)
        ax.set_ylabel('conn. weight', fontsize=fs)
        lg = ax.legend(title=lgtitle, fontsize=fs, bbox_to_anchor=(1.2,1.2),
                       labelspacing=0.2, loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(axis='y',nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/conn_'+conn_type+'_contextdm.pdf', transparent=True)


def _gen_taskparams(stim1_loc, n_rep=1):
    """Generate parameters for task."""
    # Do not loop over stim1_loc
    # Generate task parameterse for state-space analysis
    n_stim = 6
    batch_size = n_rep * n_stim**2
    batch_shape = (n_stim, n_stim, n_rep)
    ind_stim_mod1, ind_stim_mod2, ind_rep = np.unravel_index(range(batch_size), batch_shape)

    stim1_locs = np.ones(batch_size)*stim1_loc

    stim2_locs = (stim1_locs+np.pi) % (2*np.pi)

    stim_cohs = np.array([-0.5, -0.15, -0.05, 0.05, 0.15, 0.5])*0.5
    stim_mod1_cohs = np.array([stim_cohs[i] for i in ind_stim_mod1])
    stim_mod2_cohs = np.array([stim_cohs[i] for i in ind_stim_mod2])

    params = {'stim1_locs' : stim1_locs,
              'stim2_locs' : stim2_locs,
              'stim1_mod1_strengths' : 1 + stim_mod1_cohs,
              'stim2_mod1_strengths' : 1 - stim_mod1_cohs,
              'stim1_mod2_strengths' : 1 + stim_mod2_cohs,
              'stim2_mod2_strengths' : 1 - stim_mod2_cohs,
              'stim_time'    : 800}

    return params, batch_size


def _sort_ind_byvariance(**kwargs):
    """Sort ind into group 1, 2, 12, and others according task variance

    Returns:
        ind_group: indices for the current matrix, not original
        ind_active_group: indices for original matrix
    """
    ua = UnitAnalysis(kwargs['model_dir'])
    
    ind_active = ua.ind_active  # TODO(gryang): check if correct

    ind_group = dict()
    ind_active_group = dict()

    for group in ['1', '2', '12', None]:
        if group is not None:
            # Find all units here that belong to group 1, 2, or 12 as defined in UnitAnalysis
            ind_group[group] = [k for k, ind in enumerate(ind_active)
                                if ind in ua.group_ind_orig[group]]
        else:
            ind_othergroup = np.concatenate(ua.group_ind_orig.values())
            ind_group[group] = [k for k, ind in enumerate(ind_active)
                                if ind not in ind_othergroup]

        # Transform to original matrix indices
        ind_active_group[group] = [ind_active[k] for k in ind_group[group]]

    return ind_group, ind_active_group


def _sort_ind_bybeta(self):
    """Sort ind into group 1, 2, 12, and others according beta weights

    Returns:
        ind_group: indices for the current matrix, not original
        ind_active_group: indices for original matrix
    """
    theta = 0

    ind_group = dict()
    ind_active_group = dict()

    for group in ['1', '2', '12', None]:
        if group == '1':
            ind = (coef[:, 1] > theta) * (coef[:, 2] < theta)
        elif group == '2':
            ind = (coef[:, 1] < theta) * (coef[:, 2] > theta)
        elif group == '12':
            ind = (coef[:, 1] > theta) * (coef[:, 2] > theta)
        elif group is None:
            ind = (coef[:, 1] < theta) * (coef[:, 2] < theta)
        else:
            raise ValueError()
        ind = np.arange(len(ind))[ind]
        ind_group[group] = ind

        # Transform to original matrix indices
        ind_active_group[group] = [ind_active[k] for k in ind]

    return ind_group, ind_active_group


def sort_ind_bygroup(grouping, **kwargs):
    """Sort indices by group."""
    if grouping == 'var':
        return _sort_ind_byvariance(**kwargs)
    elif grouping == 'beta':
        return _sort_ind_bybeta(**kwargs)
    else:
        raise ValueError()


class StateSpaceAnalysis(object):

    def __init__(self,
                 model_dir,
                 lesion_units=None,
                 select_group=None,
                 sigma_rec=None,
                 analyze_allunits=False,
                 redefine_choice=False,
                 z_score=True,
                 surrogate_data=False,
                 n_rep=1
                 ):
        """State space analysis.

        Perform state space analysis in the style of Mante, Sussillo 2013

        Args:
            lesion_units: None of list of ints, the units to lesion
            select_group: None or list of ints, the group of units to analyze
            sigma_rec: None or float, if float, override the original value
            analyze_allunits: bool, if True analyze all units, else exclude
              units with low task variance
            redefine_choice: bool, if True redefine choice such that choice 1
              is always the preferred motor response
            z_score: bool, if True z-score the data as pre-processing
            surrogate_data: bool, if True use surrogate data
            n_rep: int, the number of different stimulus locations used
        """
        # TODO(gryang): rewrite using load_data()
        raise NotImplementedError()
        # If using surrogate data, create random matrix for later use
        if surrogate_data:
            raise NotImplementedError()
            from scipy.stats import ortho_group
            with Run(model_dir, fast_eval=True) as R:
                w_rec  = R.w_rec
            nh = w_rec.shape[0]
            random_ortho_matrix = ortho_group.rvs(dim=nh)

        #################### Computing Neural Activity #########################
        # Get rules and regressors
        rules = ['contextdm1', 'contextdm2']
        regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']

        n_rule = len(rules)
        n_regr = len(regr_names)

        # stim1_loc_list = np.arange(6)/6.*2*np.pi
        coef_maxt_list = list()
        H_new_list = list()

        model = Model(model_dir, sigma_rec=sigma_rec)
        hparams = model.hparams
        with tf.Session() as sess:
            model.restore()
            if lesion_units is not None:
                model.lesion_units(sess, lesion_units)

            # Generate task parameters used
            # Target location

            H = list()
            y_sample = list()
            y_sample_loc = list()
            stim1_locs = list()
            for stim1_loc in stim1_loc_list:
                params, batch_size = _gen_taskparams(stim1_loc=stim1_loc, n_rep=n_rep)
                stim1_locs_tmp = np.tile(params['stim1_locs'], n_rule)

                x = list() # Network input
                y_loc = list() # Network target output location

                # Start computing the neural activity
                for i, rule in enumerate(rules):
                    # Generating task information
                    trial = generate_trials(rule, hparams, 'psychometric', params=params, noise_on=True)
                    x.append(trial.x)
                    y_loc.append(trial.y_loc)

                x = np.concatenate(x, axis=1)
                y_loc = np.concatenate(y_loc, axis=1)

                # Get neural activity
                fetches = [model.h, model.y_hat, model.y_hat_loc]
                H_tmp, y_sample_tmp, y_sample_loc_tmp = sess.run(
                    fetches, feed_dict={model.x: x})

                H.append(H_tmp)
                stim1_locs.extend(stim1_locs_tmp)
                y_sample_loc.append(y_sample_loc_tmp)

        H = np.concatenate(H, axis=2)  # concatenate along the unit axis
        stim1_locs = np.array(stim1_locs)
        y_sample_loc

        # Get performance and choices
        # perfs = get_perf(y_sample, y_loc)
        # y_choice is 1 for choosing stim1_loc, otherwise -1
        y_actual_choice = 2*(get_dist(y_sample_loc[-1]-stim1_locs)<np.pi/2) - 1
        y_target_choice = 2*(get_dist(y_loc[-1]-stim1_locs)<np.pi/2) - 1

        ###################### Processing activity ##################################

        # Downsample in time
        dt_new = 50
        every_t = int(dt_new/hparams['dt'])
        # Only analyze the target epoch
        epoch = trial.epochs['stim1']
        H = H[epoch[0]:epoch[1],...][int(every_t/2)::every_t,...]

        H_original = H.copy()

        # TODO: Temporary
        self.x = x[epoch[0]:epoch[1],...][::every_t,...]

        hparams['dt_new'] = dt_new

        nt, nb, nh = H.shape

        # Analyze all units or only active units
        if analyze_allunits:
            ind_active = range(nh)
        else:
            # The way to select these units are important
            ind_active = np.where(H[-1].var(axis=0) > 1e-3)[0] # current setting
            # ind_active = np.where(H[-1].var(axis=0) > 1e-5)[0] # current setting

            # ind_active = np.where(H.reshape((-1, nh)).var(axis=0) > 1e-10)[0]
            # ind_active = np.where(H.var(axis=1).mean(axis=0) > 1e-4)[0]

        if select_group is not None:
            ind_active = select_group

        H = H[:, :, ind_active]
        # TODO: Temporary
        self.H_orig_active = H.copy()

        nh = len(ind_active)  # new nh
        H = H.reshape((-1, nh))

        # Z-scoring response across time and trials (can have a strong impact on results)
        if z_score:
            self.meanh = H.mean(axis=0)
            self.stdh  = H.std(axis=0)
            H = H - self.meanh
            H = H/self.stdh

        # Transform back
        H = H.reshape((nt, nb, nh))

        # Get neuronal preferences (+1 if activity is higher for choice=+1)
        # preferences = (H[:, (y_actual_choice== 1)*(perfs==1), :].mean(axis=(0,1)) >
        #                H[:, (y_actual_choice==-1)*(perfs==1), :].mean(axis=(0,1)))*2-1

        # preferences = (H[:, y_actual_choice== 1, :].mean(axis=(0,1)) >
        #                H[:, y_actual_choice==-1, :].mean(axis=(0,1)))*2-1

        # preferences = (H[:, y_target_choice== 1, :].mean(axis=(0,1)) >
        #                H[:, y_target_choice==-1, :].mean(axis=(0,1)))*2-1

        # preferences = (H[-1, y_actual_choice== 1, :].mean(axis=0) >
        #                H[-1, y_actual_choice==-1, :].mean(axis=0))*2-1

        preferences = (H[-1, y_target_choice== 1, :].mean(axis=0) >
                       H[-1, y_target_choice==-1, :].mean(axis=0))*2-1

        ########################## Define Regressors ##################################
        # Coherences
        stim_mod1_cohs = params['stim1_mod1_strengths'] - params['stim2_mod1_strengths']
        stim_mod2_cohs = params['stim1_mod2_strengths'] - params['stim2_mod2_strengths']

        # Get task variables
        task_var = dict()
        task_var['targ_dir'] = y_actual_choice
        task_var['stim_dir']     = np.tile(stim_mod1_cohs/stim_mod1_cohs.max(), n_rule)
        task_var['stim_col2dir'] = np.tile(stim_mod2_cohs/stim_mod2_cohs.max(), n_rule)
        task_var['context']  = np.repeat([1, -1], batch_size) # +1 for Att 1, -1 for Att 2
        task_var['correct']  = (y_actual_choice==y_target_choice).astype(int)

        # Regressors (Choice, Mod1 Cohs, Mod2 Cohs, Rule)
        Regrs = np.zeros((n_rule * batch_size, n_regr))
        Regrs[:, 0] = y_target_choice
        Regrs[:, 1] = task_var['stim_dir']
        Regrs[:, 2] = task_var['stim_col2dir']
        Regrs[:, 3] = task_var['context']

        # Get unique regressors
        Regrs_new = np.vstack({tuple(row) for row in Regrs})
        # Sort it
        ind_sort = np.lexsort(Regrs_new[:, ::-1].T)
        Regrs_new = Regrs_new[ind_sort, :]

        n_cond = Regrs_new.shape[0]

        H_new = np.zeros((nt, n_cond, nh))

        for i_cond in range(n_cond):
            regr = Regrs_new[i_cond, :]

            if redefine_choice:
                for pref in [1, -1]:
                    batch_ind = ((Regrs[:,0]==pref*regr[0])*(Regrs[:,1]==pref*regr[1])*
                                 (Regrs[:,2]==pref*regr[2])*(Regrs[:,3]==regr[3]))
                    H_new[:, i_cond, preferences==pref] = H[:, batch_ind, :][:, :, preferences==pref].mean(axis=1)

            else:
                batch_ind = ((Regrs[:,0]==regr[0])*(Regrs[:,1]==regr[1])*
                             (Regrs[:,2]==regr[2])*(Regrs[:,3]==regr[3]))
                H_new[:, i_cond, :] = H[:, batch_ind, :].mean(axis=1)

        ################################### Regression ################################
        from sklearn import linear_model

        # Time-independent coefficient vectors (n_unit, n_regress)
        coef_maxt = np.zeros((nh, n_regr))

        # Looping over units
        # Although this is slower, it's still quite fast, and it's clearer and more flexible
        for i in range(nh):
            # To satisfy sklearn standard
            Y = np.swapaxes(H_new[:,:,i], 0, 1)

            # Linear regression
            regr = linear_model.LinearRegression()
            regr.fit(Regrs_new, Y)

            # Get time-independent coefficient vector
            coef = regr.coef_
            ind = np.argmax(np.sum(coef**2, axis=1))
            coef_maxt[i, :] = coef[ind,:]

        coef_maxt_list.append(coef_maxt)
        H_new_list.append(H_new)

        H_new = np.concatenate(H_new_list, axis=2) # Concatenate along units
        coef_maxt = np.concatenate(coef_maxt_list, axis=0)

        # Orthogonalize with QR decomposition
        # Matrix q represents the orthogonalized task-related axes
        q, _ = np.linalg.qr(coef_maxt)

        # Standardize the signs of axes
        H_new_tran = np.dot(H_new, q)
        if ( H_new_tran[:, Regrs_new[:, 0]== 1, 0].mean(axis=(0,1)) >
             H_new_tran[:, Regrs_new[:, 0]==-1, 0].mean(axis=(0,1)) ):
            q[:, 0] = -q[:, 0]
        if ( H_new_tran[:, Regrs_new[:, 1] < 0, 1].mean(axis=(0,1)) >
             H_new_tran[:, Regrs_new[:, 1] > 0, 1].mean(axis=(0,1)) ):
            q[:, 1] = -q[:, 1]
        if ( H_new_tran[:, Regrs_new[:, 2] < 0, 2].mean(axis=(0,1)) >
             H_new_tran[:, Regrs_new[:, 2] > 0, 2].mean(axis=(0,1)) ):
            q[:, 2] = -q[:, 2]

        H_new_tran = np.dot(H_new, q)

        self.hparams = hparams
        self.task_var = task_var
        self.Regrs_orig = Regrs
        self.Regrs = Regrs_new
        self.H = H_new
        self.H_tran = H_new_tran
        self.H_original = H_original
        self.regr_names = regr_names
        self.coef = coef_maxt
        self.rules = rules
        self.ind_active = ind_active
        self.stim1_locs = stim1_locs
        self.q = q
        self.lesion_units = lesion_units
        self.z_score = z_score
        self.model_dir = model_dir
        self.colors = dict(zip([None, '1', '2', '12'],
                           sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))


    def sort_coefs_bygroup(self, coefs_dict, grouping):
        """Sort coefs by group 1, 2, 12, and others.

        Args:
            coefs_dict: dictionary of np arrays
            grouping: str, type of grouping to perform, 'var' or 'beta'

        Returns:
            coefs_dict: updated dictionary of np arrays
        """

        # If coefs is not None, then update coefs
        if coefs_dict is None:
            coefs_dict = dict()

        ind_group, _ = self.sort_ind_bygroup(grouping)

        for group in ['1', '2', '12', None]:
            if group not in coefs_dict:
                coefs_dict[group] = self.coef[ind_group[group], :]
            else:
                coefs_dict[group] = np.concatenate(
                    (coefs_dict[group], self.coef[ind_group[group], :]))

        return coefs_dict

    def plot_betaweights(self, coefs_dict, fancy_color=False, save_name=None):
        """Plot beta weights.

        Args:
            coefs_dict: dict of np arrays (N, 4), coefficients
            fancy_color: bool, whether to use fancy colors
        """

        regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']

        # Plot important comparisons
        fig, axarr = plt.subplots(3, 2, figsize=(4,5))
        fs = 6

        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for i_plot in range(6):
            i, j = pairs[i_plot]
            ax = axarr[i_plot//2, i_plot%2]

            for group in [None, '12', '1', '2']:
                if group is not None:
                    if fancy_color:
                        color = self.colors[group]
                    else:
                        color = 'gray'
                else:
                    color = 'gray'
                # Find all units here that belong to group 1, 2, or 12
                # as defined in UnitAnalysis
                ax.plot(coefs_dict[group][:,i], coefs_dict[group][:,j], 'o',
                        color=color, ms=1.5, mec='white', mew=0.2)

            ax.plot([-2, 2], [0, 0], color='gray')
            ax.plot([0, 0], [-2, 2], color='gray')
            ax.set_xlabel(regr_names[i], fontsize=fs)
            ax.set_ylabel(regr_names[j], fontsize=fs)
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])

            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.locator_params(nbins=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        plt.tight_layout()

        if save:
            fig_name = 'beta_weights_sub'
            if fancy_color:
                fig_name = fig_name + '_color'
            if save_name:
                fig_name += save_name
            plt.savefig(os.path.join('figure', fig_name+'.pdf'), transparent=True)
        plt.show()

    def get_slowpoints(self):
        raise NotImplementedError()
        ####################### Find Fixed & Slow Points ######################
        if self.redefine_choice:
            ValueError('Finding slow points is invalid when choices are redefined')

        if self.lesion_units is not None:
            ValueError('Lesion units not supported yet')

        # Find Fixed points
        # Choosing sstimting points
        self.fixed_points_trans_all = dict()
        self.slow_points_trans_all  = dict()

        # Looping over rules
        for rule in self.rules:

            print(rule_name[rule])

            ######################## Find Fixed Points ########################

            # Zero-coherence network input
            params = {'stim1_locs' : [self.stim1_loc],
                      'stim2_locs' : [np.mod(self.stim1_loc+np.pi, 2*np.pi)],
                      'stim1_mod1_strengths' : [1],
                      'stim2_mod1_strengths' : [1],
                      'stim1_mod2_strengths' : [1],
                      'stim2_mod2_strengths' : [1],
                      'stim_time'    : 600}

            task        = generate_onebatch(rule, self.hparams, 'psychometric', noise_on=False, params=params)
            epoch       = task.epochs['stim1']
            input_coh0  = task.x[epoch[1]-1, 0, :]

            # Get two sstimting points from averaged activity when choice is +1 or -1
            tmp = list()
            # Looping over choice
            for ch in [-1, 1]:
                # Last time point activity for all conditions with tihs choice
                h_tmp = self.H[-1, self.Regrs[:,0]==ch, :]

                # Get index of the condition that is farthest away from origin
                ind = np.argmax(np.sum(h_tmp**2, 1))
                tmp.append(h_tmp[ind, :])
            tmp = np.array(tmp)

            # Notice H is z-scored. Now get the sstimting point in original space
            if self.z_score:
                tmp *= self.stdh
                tmp += self.meanh


            nh_orig = self.hparams['shape'][1]
            sstimt_points = np.zeros((2, nh_orig))
            print(sstimt_points.shape)
            # Re-express sstimting points in original space
            sstimt_points[:, self.ind_active] = tmp

            # Find fixed points with function find_slowpoints
            res_list = search_slowpoints(model_dir, input=input_coh0,
                                       sstimt_points=sstimt_points, find_fixedpoints=True)

            # Store fixed points in original space, and in z-scored, subsampled space
            fixed_points_raws  = list()
            fixed_points_trans = list()
            for i, res in enumerate(res_list):
                print(res.success, res.message, res.fun)

                # Original space
                fixed_points_raws.append(res.x)

                # Transformed space
                fixed_points = res.x[self.ind_active]
                if self.z_score:
                    fixed_points -= self.meanh
                    fixed_points /= self.stdh

                # Task-related axes space
                fixed_points_tran = np.dot(fixed_points, self.q)
                fixed_points_trans.append(fixed_points_tran)

            fixed_points_raws  = np.array(fixed_points_raws)
            fixed_points_trans = np.array(fixed_points_trans)
            self.fixed_points_trans_all[rule] = fixed_points_trans


            ######################## Find Slow Points ########################
            # The sstimting conditions will be equally sampled points in between two fixed points
            n_slow_points = 100 # actual points will be this minus 1
            mix_weight = np.array([np.arange(1,n_slow_points),
                                   n_slow_points-np.arange(1,n_slow_points)], dtype='float').T/n_slow_points

            # Various ways to generate sstimting points for the search

            # sstimt_points = np.dot(mix_weight, fixed_points_raws)
            sstimt_points = np.dot(mix_weight, sstimt_points)
            # sstimt_points+= np.random.randn(*sstimt_points.shape) # Randomly perturb sstimting points
            # sstimt_points *= np.random.uniform(0, 2, size=sstimt_points.shape) # Randomly perturb sstimting points
            # sstimt_points = np.random.rand(100, nh)*3

            # Search slow points with the same input but different sstimting points
            res_list = search_slowpoints(model_dir, input=input_coh0,
                                       sstimt_points=sstimt_points, find_fixedpoints=False)

            slow_points_trans = list()
            for i, res in enumerate(res_list):
                # Transformed space
                slow_points = res.x[self.ind_active]
                if self.z_score:
                    slow_points -= self.meanh
                    slow_points /= self.stdh

                # Task-related axes space
                slow_points_tran = np.dot(slow_points, self.q)
                slow_points_trans.append(slow_points_tran)

            slow_points_trans = np.array(slow_points_trans)
            self.slow_points_trans_all[rule] = slow_points_trans

    def get_regr_ind(self, choice=None, coh1=None, coh2=None, rule=None):
        """For given choice, coh1, coh2, rule, get the indices of trials."""
        ind = np.ones(self.Regrs.shape[0], dtype=bool) # initialize

        if choice is not None:
            ind *= (self.Regrs[:, 0] == choice)

        if coh1 is not None:
            ind *= (self.Regrs[:, 1] == coh1)

        if coh2 is not None:
            ind *= (self.Regrs[:, 2] == coh2)

        if rule is not None:
            j_rule = 1 if rule == 'contextdm1' else -1
            ind *= (self.Regrs[:, 3] == j_rule)

        return ind

    def plot_statespace(self, plot_slowpoints=True):
        """Plot state space analysis."""

        if plot_slowpoints:
            try:
                # Check if slow points are already computed
                _ = self.slow_points_trans_all
            except AttributeError:
                # If not, compute it now.
                self.get_slowpoints()

        ################ Pretty Plotting of State-space Results #######################
        fs = 6

        colors1 = sns.diverging_palette(10, 220, sep=1, s=99, l=30, n=6)
        colors2 = sns.diverging_palette(280, 145, sep=1, s=99, l=30, n=6)

        fig, axarr = plt.subplots(2, len(self.rules),
                                  sharex=True, sharey='row',
                                  figsize=(len(self.rules)*1, 2))
        for i_col, rule in enumerate(self.rules):
            # Different ways of separation, either by Mod1 or Mod2
            # Also different subspaces shown
            for i_row in range(2):
                ax = axarr[i_row, i_col]
                ax.axis('off')

                if i_row == 0:
                    # Separate by coherence of Mod 1
                    cohs = self.Regrs[:, 1]

                    # Show subspace (Choice, Mod1)
                    pcs = [0, 1]

                    # Color set
                    colors = colors1

                    # Rule title
                    ax.set_title(rule_name[rule], fontsize=fs, y=0.8)
                else:
                    # Separate by coherence of Mod 2
                    cohs = self.Regrs[:, 2]

                    # Show subspace (Choice, Mod2)
                    pcs = [0, 2]

                    # Color set
                    colors = colors2


                if i_col == 0:
                    anc = [self.H_tran[:,:,pcs[0]].min()+1,
                           self.H_tran[:,:,pcs[1]].max()-5]  # anchor point
                    ax.plot([anc[0], anc[0]], [anc[1]-5, anc[1]-1],
                            color='black', lw=1.0)
                    ax.plot([anc[0]+1, anc[0]+5], [anc[1], anc[1]],
                            color='black', lw=1.0)
                    ax.text(anc[0], anc[1], self.regr_names[pcs[0]],
                            fontsize=fs, va='bottom')
                    ax.text(anc[0], anc[1], self.regr_names[pcs[1]],
                            fontsize=fs, rotation=90, ha='right', va='top')

                # ind = self.get_regr_ind(rule=rule) # for batch
                # ax.plot(self.H_tran[:,ind,pcs[0]], self.H_tran[:,ind,pcs[1]],
                #     '-', color='gray', alpha=0.5, linewidth=0.5)

                # Loop over coherences to choice 1, from high to low
                for i, coh in enumerate(np.unique(cohs)[::-1]):

                    # Loop over choices
                    for choice in [1, -1]:

                        if choice == 1:
                            # Solid circles
                            kwargs = {'markerfacecolor':colors[i], 'linewidth':1}
                        else:
                            # Empty circles
                            kwargs = {'markerfacecolor':'white', 'linewidth':0.5}

                        if i_row == 0:
                            # Separate by coherence of Mod 1
                            ind = self.get_regr_ind(choice=choice, coh1=coh, rule=rule) # for batch
                        else:
                            # Separate by coherence of Mod 2
                            ind = self.get_regr_ind(choice=choice, coh2=coh, rule=rule) # for batch


                        if not np.any(ind):
                            continue

                        h_plot = self.H_tran[:, ind, :].mean(axis=1)

                        ax.plot(h_plot[:,pcs[0]], h_plot[:,pcs[1]],
                                '.-', markersize=2, color=colors[i], markeredgewidth=0.2, **kwargs)


                        if not plot_slowpoints:
                            continue

                        # Plot slow points
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
                colors = colors1
            else:
                ax = fig.add_axes([0.25,0.05,0.2,0.1])
                colors = colors2

            for i in range(6):
                kwargs = {'markerfacecolor' : colors[i], 'linewidth' : 1}
                ax.plot([i], [0], '.-', color=colors[i],
                        markersize=4, markeredgewidth=0.5, **kwargs)
            ax.axis('off')
            ax.text(2.5, 1, 'Strong Weak Strong',
                    fontsize=5, va='bottom', ha='center')
            # During looping, we use coherence to choice 1 from high to low
            ax.text(2.5, -1, 'To choice 1    To choice 2',
                    fontsize=5, va='top', ha='center')
            ax.set_xlim([-1,6])
            ax.set_ylim([-3,3])

        if save:
            figname = os.path.join(
                'figure', 'fixpoint_choicetasks_statespace.pdf')
            plt.savefig(figname, transparent=True)
        plt.show()

    def plot_units_intime(self, plot_individual=False):
        """Plot averaged unit activity in time for all groups."""
        for group in ['1', '2', '12']:
            self.plot_units_intime_bygroup(group, plot_individual)

    def plot_units_intime_bygroup(self, group, plot_individual=False):
        """Plot averaged unit activity in time for one group."""

        ind_group, ind_active_group = self.sort_ind_bygroup()

        rules = ['contextdm1', 'contextdm2']

        t_plot = np.arange(self.H.shape[0])*self.hparams['dt_new']/1000

        # Plot the group averaged activity
        fig, axarr = plt.subplots(1, 2, figsize=(2,1.0), sharey=True)
        fig.suptitle('Group {:s} average'.format(group), fontsize=7)
        for i_rule, rule in enumerate(rules):
            ax = axarr[i_rule]
            ind_trial = self.get_regr_ind(rule=rule)
            h_plot = self.H[:, ind_trial, :][:, :, ind_group[group]]
            h_plot = h_plot.mean(axis=1).mean(axis=1)
            _ = ax.plot(t_plot, h_plot, color=self.colors[group])

            for ind_unit in ind_group[group]:
                h_plot = self.H[:, ind_trial, ind_unit]
                h_plot = h_plot.mean(axis=1)
                _ = ax.plot(t_plot, h_plot, color='gray', alpha=0.1)

            if not self.z_score:
                ax.set_ylim([0, 1.5])

            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.locator_params(nbins=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join('figure',
        'choiceatt_intime_group'+group+'.pdf'), transparent=True)

        if not plot_individual:
            return

        # Plot individual units
        for ind_unit in ind_group[group]:
            fig, axarr = plt.subplots(1, 2, figsize=(3,1.2), sharey=True)
            fig.suptitle('Group {:s}, Unit {:d}'.format(group, ind_unit), fontsize=7)
            for i_rule, rule in enumerate(rules):
                ax = axarr[i_rule]
                ind_trial = self.get_regr_ind(rule=rule)
                _ = ax.plot(t_plot, self.H[:,ind_trial, ind_unit])

                ax.tick_params(axis='both', which='major', labelsize=7)
                ax.locator_params(nbins=2)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

    def plot_currents_intime(self):
        """Plot averaged currents between all groups."""
        for group_from in ['1', '2', '12']:
            for group_to in ['1', '2', '12']:
                self.plot_currents_intime_bygroup(group_from, group_to)

    def plot_currents_intime_bygroup(self, group_from, group_to):
        """Plot averaged currents from one group to another."""
        ind_group, ind_active_group = self.sort_ind_bygroup()

        rules = ['contextdm1', 'contextdm2']

        t_plot = np.arange(self.H.shape[0])*self.hparams['dt_new']/1000

        # Plot the group averaged activity
        fig, axarr = plt.subplots(1, 2, figsize=(4,2.0), sharey=True)
        fig.suptitle('From group {:s} to group {:s} average'.format(group_from, group_to), fontsize=7)
        for i_rule, rule in enumerate(rules):
            ax = axarr[i_rule]
            ind_trial = self.get_regr_ind(rule=rule)
            h_group = self.H[:, ind_trial, :][:, :, ind_group[group_from]]
            w_group = (self.w_rec[ind_group[group_to], :][:, ind_group[group_from]]).T

            current_plot = np.dot(h_group, w_group)
            # Average across trials
            current_plot = current_plot.mean(axis=1)

            # Average across units
            current_plot_mean = current_plot.mean(axis=1)
            _ = ax.plot(t_plot, current_plot_mean, color='black')

            # for i in range(len(ind_group[group_to])):
            #     _ = ax.plot(t_plot, current_plot[:,i], color='gray', alpha=0.1, linewidth=0.5)

            for i_from in ind_group[group_from]:
                for i_to in ind_group[group_to]:
                    current = self.H[:, ind_trial, i_from] * self.w_rec[i_to, i_from]
                    current = current.mean(axis=1)
                    _ = ax.plot(t_plot, current, color='gray', alpha=0.1, linewidth=0.5)

            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.locator_params(nbins=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        plt.tight_layout()

        if save:
            save_name = 'choiceatt_intime_group'+group_from+'-'+group_to
            plt.savefig(os.path.join('figure', save_name+'.pdf'), transparent=True)

        return


def _plot_performance_choicetasks(model_dir, lesion_units_list,
                                  legends=None, save_name=None):
    """Plot performance across tasks for different lesioning.

    Args:
        model_dir: str, model directory
        lesion_units_list: list of lists of units to lesion
        legends: list of str
        save_name: None or str, add to the figure name
    """
    rules_perf = ['contextdm1', 'contextdm2']

    perf_stores = list()
    for lesion_units in lesion_units_list:
        perf_stores_tmp = list()
        for rule in rules_perf:
            perf, prop1s, cohs = performance.psychometric_choicefamily_2D(
                model_dir, rule, lesion_units=lesion_units,
                n_coh=4, n_stim_loc=10)
            perf_stores_tmp.append(perf)

        perf_stores_tmp = np.array(perf_stores_tmp)
        perf_stores.append(perf_stores_tmp)

    fs = 6
    width = 0.15
    fig = plt.figure(figsize=(3,1.5))
    ax = fig.add_axes([0.17,0.35,0.8,0.4])
    for i, perf in enumerate(perf_stores):
        b0 = ax.bar(np.arange(len(rules_perf))+(i-2)*width, perf,
                    width=width, color=COLORS[i], edgecolor='none')
    ax.set_xticks(np.arange(len(rules_perf)))
    ax.set_xticklabels([rule_name[r] for r in rules_perf], rotation=25)
    ax.set_xlabel('Tasks', fontsize=fs, labelpad=3)
    ax.set_ylabel('performance', fontsize=fs)
    lg = ax.legend(legends,
                   fontsize=fs, ncol=2, bbox_to_anchor=(1,1.4),
                   labelspacing=0.2, loc=1, frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.locator_params(axis='y',nbins=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim([-0.8, len(rules_perf)-0.2])
    if save:
        fig_name = 'figure/perf_contextdm_lesion'
        if save_name is not None:
            fig_name += save_name
        plt.savefig(fig_name + '.pdf', transparent=True)
    plt.show()


def plot_performance_choicetasks(model_dir, grouping):
    """Plot performance across tasks for different lesioning.

    Args:
        model_dir: str, model directory
        grouping: str, how to group different populations
    """
    ind_group, ind_group_orig = sort_ind_bygroup(grouping=grouping,
                                                 model_dir=model_dir)
    groups = ['1', '2', '12']
    lesion_units_list = [None] + [ind_group_orig[g] for g in groups]
    legends = ['Intact']+['Lesion group {:s}'.format(l) for l in groups]
    save_name = '_bygrouping_' + grouping
    print(lesion_units_list)
    _plot_performance_choicetasks(model_dir, lesion_units_list,
                                  legends=legends, save_name=save_name)


def plot_performance_2D(model_dir, rule, lesion_units=None,
                        title=None, save_name=None, **kwargs):
    """Plot performance as function of both modality coherence."""
    perf, prop1s, cohs = performance.psychometric_choicefamily_2D(
        model_dir, rule, lesion_units=lesion_units,
        n_coh=8, n_stim_loc=20)

    performance._plot_psychometric_choicefamily_2D(
        prop1s, cohs, rule, title=title, save_name=save_name, **kwargs)


def plot_performance_2D_all(model_dir, rule):
    """Plot performance of both modality coherence for all groups."""
    ua = UnitAnalysis(model_dir)
    plot_performance_2D(model_dir, rule)
    for lesion_group in ['1', '2', '12', '1+2']:
        lesion_units = ua.group_ind_orig[lesion_group]
        title = rule_name[rule] + '\n' + ua.lesion_group_names[lesion_group]
        save_name = rule_name[rule].replace(' ', '') + \
                    '_perf2D_lesion' + str(lesion_group) + '.pdf'
        plot_performance_2D(
            model_dir, rule, lesion_units=lesion_units,
            title=title, save_name=save_name, ylabel=False, colorbar=False)


def plot_performance_lesionbyactivity(root_dir, activation, n_lesion=20):
    """Plot performance across tasks for different lesioning.

    Args:
        model_dir: str, model directory
        activation: str, activation function
        n_lesion: int, number of units to lesion
    """
    hp_target = {'activation': activation,
                 'rnn_type': 'LeakyGRU',
                 'w_rec_init': 'randortho',
                 'l1_h': 0,
                 'l1_weight': 0}
    model_dir, _ = tools.find_model(root_dir, hp_target)
    ssa = StateSpaceAnalysis(model_dir, lesion_units=None,
                             redefine_choice=True)
    hh = ssa.H_original
    hh = hh.mean(axis=(0, 1))
    ind_sort = np.argsort(hh)
    
    # ind_group, ind_group_orig = ssa.sort_ind_bygroup(grouping=grouping)
    mid = int(len(ind_sort)/2)
    lesion_units_list = [None, ind_sort[:n_lesion], ind_sort[-n_lesion:],
                         ind_sort[mid-int(n_lesion/2):mid+int(n_lesion/2)]]
    legends = ['Intact', 'Most Neg ' + str(n_lesion),
               'Most Pos ' + str(n_lesion), 'Mid ' + str(n_lesion)]
    save_name = '_byactivity_' + ssa.hparams['activation']
    _plot_performance_choicetasks(model_dir, lesion_units_list,
                                  legends=legends, save_name=save_name)


def plot_fullconnectivity(model_dir):
    """Plot connectivity of the entire matrix."""
    ua = UnitAnalysis(model_dir)
    ind_active = ua.ind_active
    h_normvar_all = ua.h_normvar_all
    # Sort data by labels and by input connectivity
    model = Model(model_dir)
    hparams = model.hparams
    with tf.Session() as sess:
        model.restore()
        w_in, w_out, w_rec = sess.run(
            [model.w_in, model.w_out, model.w_rec])
    w_in, w_rec, w_out = w_in.T, w_rec.T, w_out.T

    nx, nh, ny = hparams['n_input'], hparams['n_rnn'], hparams['n_output']
    n_ring = hparams['n_eachring']

    ind_active_new = list()
    labels = list()
    for i in range(len(ind_active)):
        ind_active_addnew = True
        if h_normvar_all[i,0]>0.85:
            labels.append(0)
        elif h_normvar_all[i,0]<0.15:
            labels.append(1)
        elif (h_normvar_all[i,0]>0.15) and (h_normvar_all[i,0]<0.85):
            # labels.append(2)

            # Further divide
            # This condition works especially for networks trained only for contextdm
            if np.max(w_in[ind_active[i],1:2*n_ring+1]) > 1.0:
                labels.append(2)
            else:
                labels.append(3)

                # if np.var(w_out[1:, ind_active[i]])>0.005:
                #     labels.append(3)
                # else:
                #     labels.append(4)
        else:
            ind_active_addnew = False

        if ind_active_addnew:
            ind_active_new.append(ind_active[i])

    labels = np.array(labels)
    ind_active = np.array(ind_active_new)

    # Preferences
    # w_in preferences
    w_in = w_in[ind_active, :]
    w_in_mod1 = w_in[:, 1:n_ring+1]
    w_in_mod2 = w_in[:, n_ring+1:2*n_ring+1]
    w_in_modboth = w_in_mod1 + w_in_mod2
    w_in_prefs = np.argmax(w_in_modboth, axis=1)
    # w_out preferences
    w_out = w_out[1:, ind_active]
    w_out_prefs = np.argmax(w_out, axis=0)

    label_sort_by_w_in = [0,1,2]
    sort_by_w_in = np.array([(label in label_sort_by_w_in) for label in labels])

    w_prefs = (1-sort_by_w_in)*w_out_prefs + sort_by_w_in*w_in_prefs

    ind_sort        = np.lexsort((w_prefs, labels)) # sort by labels then by prefs
    labels          = labels[ind_sort]
    ind_active      = ind_active[ind_sort]

    nh = len(ind_active)
    nr = n_ring
    nrule = 2
    nx = 2*nr+1+nrule
    ind = ind_active

    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        w_in, w_out, w_rec, b_rec, b_out = sess.run([
            model.w_in, model.w_out, model.w_rec, model.b_rec, model.b_out
        ])
    w_in = (w_in.T)[ind, :]
    w_rec = (w_rec.T)[ind, :][:, ind]
    w_out = (w_out.T)[:, ind]
    b_rec = b_rec[ind, np.newaxis]
    b_out = b_out[:, np.newaxis]

    l = 0.35
    l0 = (1-1.5*l)/nh

    rules = ['contextdm1', 'contextdm2']
    ind_rule = [get_rule_index(r, hparams) for r in rules]
    w_in_rule = w_in[:, ind_rule]

    plot_infos = [(w_rec              , [l               ,l          ,nh*l0    ,nh*l0]),
                  (w_in[:,[0]]        , [l-(nx+15)*l0    ,l          ,1*l0     ,nh*l0]), # Fixation input
                  (w_in[:,1:nr+1]     , [l-(nx+11)*l0    ,l          ,nr*l0    ,nh*l0]), # Mod 1 stimulus
                  (w_in[:,nr+1:2*nr+1], [l-(nx-nr+8)*l0  ,l          ,nr*l0    ,nh*l0]), # Mod 2 stimulus
                  (w_in_rule          , [l-(nx-2*nr+5)*l0,l          ,nrule*l0 ,nh*l0]), # Rule inputs
                  (w_out[[0],:]       , [l               ,l-(4)*l0   ,nh*l0    ,1*l0]),
                  (w_out[1:, :]       , [l               ,l-(ny+6)*l0,nh*l0    ,(ny-1)*l0]),
                  (b_rec              , [l+(nh+6)*l0     ,l          ,l0       ,nh*l0]),
                  (b_out              , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0       ,ny*l0])]

    cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
    fig = plt.figure(figsize=(6,6))
    for plot_info in plot_infos:
        ax = fig.add_axes(plot_info[1])
        # vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5,50,95])
        vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [2,50,98])
        vmid = 0
        _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                      vmin=vmid-(vmax-vmin)/2, vmax=vmid+(vmax-vmin)/2)

        # vabs = np.max(abs(plot_info[0]))*0.3
        # _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto', vmin=-vabs, vmax=vabs)
        ax.axis('off')

    # colors = sns.xkcd_palette(['green', 'sky blue', 'pink'])
    # colors = sns.color_palette('deep', len(np.unique(labels)))
    colors = COLORS[1:] + ['gray']
    ax1 = fig.add_axes([l     , l+nh*l0, nh*l0, 6*l0])
    ax2 = fig.add_axes([l-6*l0, l      , 6*l0 , nh*l0])
    for l in np.unique(labels):
        ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
        ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                color=colors[l])
        ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                color=colors[l])
    ax1.set_xlim([0, len(labels)])
    ax2.set_ylim([0, len(labels)])
    ax1.axis('off')
    ax2.axis('off')
    plt.savefig('figure/contextdm_connectivity.pdf', transparent=True)
    plt.show()


def plot_groupsize_TEMPDISABLED(save_type):
    HDIMs = range(150, 1000)
    group_sizes = {key : list() for key in ['1', '2', '12']}
    HDIM_plot = list()
    for HDIM in HDIMs:
        model_dir = save_type+'_'+str(HDIM)
        fname = 'data/hparams'+model_dir+'.pkl'
        if not os.path.isfile(fname):
            continue
        ua = UnitAnalysis(model_dir)
        for key in ['1', '2', '12']:
            group_sizes[key].append(len(ua.group_ind[key]))

        HDIM_plot.append(HDIM)

    fs = 6
    fig = plt.figure(figsize=(1.5,1.0))
    ax = fig.add_axes([.3, .4, .5, .5])
    for i, key in enumerate(['1', '2', '12']):
        ax.plot(HDIM_plot, group_sizes[key], label=key, color=ua.colors[key])
    ax.set_xlim([np.min(HDIM_plot)-30, np.max(HDIM_plot)+30])
    ax.set_ylim([0,100])
    ax.set_xlabel('Number of rec. units', fontsize=fs)
    ax.set_ylabel('Group size (units)', fontsize=fs)
    lg = ax.legend(title='Group',
                   fontsize=fs, ncol=1, bbox_to_anchor=(1.5,1.2),
                   loc=1, frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    plt.locator_params(nbins=3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('figure/contextdm_groupsize'+save_type+'.pdf', transparent=True)


def plot_betaweights(model_dir, grouping):
    """Plot the beta weights.

    Args:
        model_dir: the root model directory. All valid model directories under
          it would be used
    """
    sigma_rec = 0.0
    model_dirs = tools.valid_model_dirs(model_dir)

    coefs_dict = {}
    for d in model_dirs:
        ssa = StateSpaceAnalysis(d, lesion_units=None,
                                 redefine_choice=True, sigma_rec=sigma_rec)

        # Update coefficient dictionary
        coefs_dict = ssa.sort_coefs_bygroup(coefs_dict, grouping)

    # ssa.plot_betaweights(coefs_dict, fancy_color=False)
    save_name = '_groupby' + grouping
    ssa.plot_betaweights(coefs_dict, fancy_color=True, save_name=save_name)


def quick_statespace(model_dir):
    """Quick state space analysis using simply PCA."""
    rules = ['contextdm1', 'contextdm2']
    h_lastts = dict()
    model = Model(model_dir)
    hparams = model.hparams
    with tf.Session() as sess:
        model.restore()
        for rule in rules:
            # Generate a batch of trial from the test mode
            trial = generate_trials(rule, hparams, mode='test')
            feed_dict = tools.gen_feed_dict(model, trial, hparams)
            h = sess.run(model.h, feed_dict=feed_dict)
            lastt = trial.epochs['stim1'][-1]
            h_lastts[rule] = h[lastt,:,:]

    from sklearn.decomposition import PCA
    model = PCA(n_components=5)
    model.fit(np.concatenate(h_lastts.values(), axis=0))
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([.3, .3, .6, .6])
    for rule, color in zip(rules, ['red', 'blue']):
        data_trans = model.transform(h_lastts[rule])
        ax.scatter(data_trans[:, 0], data_trans[:, 1], s=1,
                   label=rule_name[rule], color=color)
    plt.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlabel('PC 1', fontsize=7)
    ax.set_ylabel('PC 2', fontsize=7)
    lg = ax.legend(fontsize=7, ncol=1, bbox_to_anchor=(1,0.3),
                   loc=1, frameon=False)
    if save:
        plt.savefig('figure/choiceatt_quickstatespace.pdf',transparent=True)


def _compute_frac_var(model_dir, **kwargs):
    """FracVar analysis for Mante et al. experimental setup."""

    ssa = StateSpaceAnalysis(model_dir, z_score=False, **kwargs)

    h = ssa.H

    h_context1 = h[:,ssa.Regrs[:,3]== 1,:]
    h_context2 = h[:,ssa.Regrs[:,3]==-1,:]

    # Last time point variance
    # h_var1 = h_context1[-1].var(axis=0)
    # h_var2 = h_context2[-1].var(axis=0)

    h_var1 = h_context1.var(axis=1).mean(axis=0)
    h_var2 = h_context2.var(axis=1).mean(axis=0)

    var_noise = 0.0
    h_var1, h_var2 = h_var1+var_noise, h_var2+var_noise

    frac_var = (h_var1-h_var2)/(h_var1+h_var2)

    return frac_var


def plot_frac_var(model_dir, **kwargs):
    """Plot the fractional variance.

        Args:
            model_dir: the root model directory. All valid model directories under
              it would be used
    """

    frac_vars = list()
    model_dirs = tools.valid_model_dirs(model_dir)

    for d in model_dirs:
        frac_var = _compute_frac_var(d, **kwargs)
        frac_vars.append(frac_var)
        tmp = np.concatenate(frac_vars)

    _ = plt.hist(tmp)


def run_all_analyses(model_dir):
    from performance import plot_trainingprogress, plot_choicefamily_varytime, psychometric_contextdm
    plot_trainingprogress(model_dir)
    for rule in ['contextdm1', 'contextdm2']:
        pass
        # plot_choicefamily_varytime(model_dir, rule)

    # psychometric_contextdm(model_dir, no_ylabel=True)

    ssa = StateSpaceAnalysis(model_dir, lesion_units=None, redefine_choice=False)
    ssa.plot_statespace(plot_slowpoints=False)

    ssa = StateSpaceAnalysis(model_dir, lesion_units=None,
                                 redefine_choice=True)
    # Update coefficient dictionary
    coefs = ssa.sort_coefs_bygroup(dict())
    ssa.plot_betaweights(coefs, fancy_color=True)

    frac_var = _compute_frac_var(model_dir, analyze_allunits=False)
    _ = plt.hist(frac_var, bins=10, range=(-1,1))

    ua = UnitAnalysis(model_dir)
    ua.plot_connectivity(conn_type='rec')
    ua.plot_connectivity(conn_type='rule')
    ua.plot_connectivity(conn_type='input')
    ua.plot_connectivity(conn_type='output')
    ua.prettyplot_hist_varprop()
    # ua.plot_performance_choicetasks()


def load_data(model_dir=None, sigma_rec=0, lesion_units=None, n_rep=1):
    """Generate model data into standard format.

    Returns:
        data: standard format, list of dict of arrays/dict
            list is over neurons
            dict is for response array and task variable dict
            response array has shape (n_trial, n_time)
    """
    if model_dir is None:
        model_dir = './mantetemp'  # TEMPORARY SETTING

    # Get rules and regressors
    rules = ['contextdm1', 'contextdm2']

    n_rule = len(rules)

    data = list()

    model = Model(model_dir, sigma_rec=sigma_rec)
    hparams = model.hparams
    with tf.Session() as sess:
        model.restore()
        if lesion_units is not None:
            model.lesion_units(sess, lesion_units)

        # Generate task parameters used
        # Target location
        stim1_loc_list = np.arange(0, 2*np.pi, 2*np.pi/12)
        for stim1_loc in stim1_loc_list:
            params, batch_size = _gen_taskparams(stim1_loc=stim1_loc, n_rep=n_rep)
            stim1_locs_tmp = np.tile(params['stim1_locs'], n_rule)

            x = list() # Network input
            y_loc = list() # Network target output location

            # Start computing the neural activity
            for i, rule in enumerate(rules):
                # Generating task information
                trial = generate_trials(rule, hparams, 'psychometric',
                                        params=params, noise_on=True)
                x.append(trial.x)
                y_loc.append(trial.y_loc)

            x = np.concatenate(x, axis=1)
            y_loc = np.concatenate(y_loc, axis=1)

            # Coherences
            stim_mod1_cohs = params['stim1_mod1_strengths'] - params[
                'stim2_mod1_strengths']
            stim_mod2_cohs = params['stim1_mod2_strengths'] - params[
                'stim2_mod2_strengths']
            stim_mod1_cohs /= stim_mod1_cohs.max()
            stim_mod2_cohs /= stim_mod2_cohs.max()

            # Get neural activity
            fetches = [model.h, model.y_hat, model.y_hat_loc]
            H, y_sample, y_sample_loc = sess.run(
                fetches, feed_dict={model.x: x})

            # Downsample in time
            dt_new = 50
            every_t = int(dt_new / hparams['dt'])
            # Only analyze the target epoch
            epoch = trial.epochs['stim1']
            H = H[epoch[0]:epoch[1], ...][int(every_t / 2)::every_t, ...]

            # Get performance and choices
            # perfs = get_perf(y_sample, y_loc)
            # y_choice is 1 for choosing stim1_loc, otherwise -1
            y_actual_choice = 2*(get_dist(y_sample_loc[-1]-stim1_loc)<np.pi/2)-1
            y_target_choice = 2*(get_dist(y_loc[-1]-stim1_loc)<np.pi/2)-1

            # Get task variables
            task_var = dict()
            task_var['targ_dir'] = y_actual_choice
            task_var['stim_dir'] = np.tile(stim_mod1_cohs, n_rule)
            task_var['stim_col2dir'] = np.tile(stim_mod2_cohs, n_rule)
            task_var['context'] = np.repeat([1, -1], batch_size)
            task_var['correct'] = (y_actual_choice == y_target_choice).astype(int)
            task_var['stim_dir_sign'] = (task_var['stim_dir']>0).astype(int)*2-1
            task_var['stim_col2dir_sign'] = (task_var['stim_col2dir']>0).astype(int)*2-1


            n_unit = H.shape[-1]
            for i_unit in range(n_unit):
                unit_dict = {
                    'rate': H[:, :, i_unit].T,  # standard format (n_trial, n_time)
                    'task_var': copy.deepcopy(task_var)
                }
                data.append(unit_dict)
    return data


if __name__ == '__main__':
# =============================================================================
#     root_dir = './data/vary_l2weight_mante'
#     hp_target = {'activation': 'softplus',
#                  'rnn_type': 'LeakyRNN',
#                  'w_rec_init': 'randortho',
#                  'l2_weight': 4*1e-4}
# =============================================================================
    root_dir = './data/mante_tanh'
    hp_target = {}
    
    model_dir, _ = tools.find_model(root_dir, hp_target, perf_min=0.8)
    
    # model_dir  = './mantetemp'
    # import variance
    # variance.compute_variance(model_dir)

    ######################### Connectivity and Lesioning ######################
    ua = UnitAnalysis(model_dir)
    # ua.plot_connectivity(conn_type='rec')
    # ua.plot_connectivity(conn_type='rule')
    # ua.plot_connectivity(conn_type='input')
    # ua.plot_connectivity(conn_type='output')
    ua.prettyplot_hist_varprop()

    # plot_performance_choicetasks(model_dir, grouping='var')
    # plot_performance_choicetasks(model_dir, grouping='beta')

    # plot_performance_2D_all(model_dir, 'contextdm1')

    # plot_fullconnectivity(model_dir)
    # plot_groupsize('allrule_weaknoise')  # disabled now
    
    # plot_performance_lesionbyactivity(root_dir, 'tanh', n_lesion=50)
    # plot_performance_lesionbyactivity(root_dir, 'softplus', n_lesion=50)

    ################### State space ###########################################
    # Plot State space
    # ssa = StateSpaceAnalysis(model_dir, lesion_units=None, redefine_choice=False)
    # ssa.plot_statespace(plot_slowpoints=False)

    # Plot beta weights
    # plot_betaweights(model_dir, grouping='var')
    # plot_betaweights(model_dir, grouping='beta')

    # Plot units in time
    # ssa = StateSpaceAnalysis(model_dir, lesion_units=None, z_score=False)
    # ssa.plot_units_intime()
    # ssa.plot_currents_intime()
    # Quick state space analysis
    # quick_statespace(model_dir)

    ################### Modified state space ##################################
    # ua = UnitAnalysis(model_dir)
    # ssa = StateSpaceAnalysis(model_dir, select_group=ua.group_ind_orig['12'])
    # ssa.plot_statespace(plot_slowpoints=False)

    ################### Frac Var ##############################################
    # plot_frac_var(model_dir, analyze_allunits=False, sigma_rec=0.5)

    # data = load_data()
    



