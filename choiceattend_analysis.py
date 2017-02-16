"""
Analysis of the choice att tasks
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

def generate_surrogate_data():
    # Generate surrogate data

    # Number of time points
    nt = 20
    t_plot = np.linspace(0, 1, nt)

    # Performance
    perfs = np.ones(batch_size)

    # Generate choice
    rel_mod = '1' if rule == CHOICEATTEND_MOD1 else '2' # relevant modality
    rel_coh = params['tar1_mod'+rel_mod+'_strengths']-params['tar2_mod'+rel_mod+'_strengths']
    y_choice = (rel_coh>0)*2-1

    # Generate underlying low-dimensional representation
    mod1_plot = np.ones((nt, batch_size)) * (params['tar1_mod1_strengths']-params['tar2_mod1_strengths'])
    mod2_plot = np.ones((nt, batch_size)) * (params['tar1_mod2_strengths']-params['tar2_mod2_strengths'])
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
    def __init__(self, save_addon, fast_eval=True):
        data_type  = 'rule'
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
        # ind_active = np.where(h_var_all.sum(axis=1) > 1e-1)[0] # TEMPORARY
        h_var_all  = h_var_all[ind_active, :]

        # Normalize by the total variance across tasks
        h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

        ind_lesions = dict()

        ind_lesions['1']  = np.where(h_normvar_all[:,0]>0.9)[0]
        ind_lesions['12'] = np.where(np.logical_and(h_normvar_all[:,0]>0.4, h_normvar_all[:,0]<0.6))[0]
        ind_lesions['2']  = np.where(h_normvar_all[:,0]<0.1)[0]

        ind_lesions_orig = {key: ind_active[val] for key, val in ind_lesions.iteritems()}

        self.save_addon         = save_addon
        self.ind_lesions        = ind_lesions
        self.ind_lesions_orig   = ind_lesions_orig
        self.fast_eval          = fast_eval
        self.h_normvar_all      = h_normvar_all
        self.rules              = rules
        self.ind_active         = ind_active
        self.colors = dict(zip([None, '1', '2', '12'],
                               sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))
        self.lesion_group_names = {None : 'intact',
                                   '1'  : 'lesion groups 1',
                                   '2'  : 'lesion groups 2',
                                   '12' : 'lesion groups 12',
                                   '1+2': 'lesion groups 1 & 2'}

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
        xlabel = 'FracVar({:s}, {:s})'.format(rule_name[rules[0]], rule_name[rules[1]])
        ax.set_xlabel(xlabel, fontsize=fs)
        ax.set_ylabel('Units', fontsize=fs)
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

    def get_performance(self, rule, lesion_group, n_coh=8, n_tar_loc=20):
        if lesion_group is None:
            lesion_units = None
        elif lesion_group == '1+2':
            lesion_units = np.concatenate((self.ind_lesions_orig['1'],self.ind_lesions_orig['2']))
        else:
            lesion_units = self.ind_lesions_orig[lesion_group]

        # Generate task parameters for choice tasks
        coh_range = 0.2
        cohs = np.linspace(-coh_range, coh_range, n_coh)

        batch_size = n_tar_loc * n_coh**2
        batch_shape = (n_tar_loc,n_coh,n_coh)
        ind_tar_loc, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

        # Looping target location
        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_mod1_cohs = cohs[ind_tar_mod1]
        tar_mod2_cohs = cohs[ind_tar_mod2]

        if rule in [CHOICE_MOD1, CHOICE_MOD2]:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_strengths' : 1 + tar_mod1_cohs, # Just use mod 1 value
                      'tar2_strengths' : 1 - tar_mod1_cohs,
                      'tar_time'    : 800
                      }
        elif rule in [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
                      'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
                      'tar1_mod2_strengths' : 1 + tar_mod2_cohs,
                      'tar2_mod2_strengths' : 1 - tar_mod2_cohs,
                      'tar_time'    : 800
                      }
        elif rule == CHOICE_INT:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_mod1_strengths' : 1 + tar_mod1_cohs,
                      'tar2_mod1_strengths' : 1 - tar_mod1_cohs,
                      'tar1_mod2_strengths' : 1 + tar_mod1_cohs, # Same as Mod 1
                      'tar2_mod2_strengths' : 1 - tar_mod1_cohs,
                      'tar_time'    : 800
                      }
        else:
            raise ValueError('Unsupported rule')

        with Run(self.save_addon, lesion_units=lesion_units, fast_eval=self.fast_eval) as R:
            task  = generate_onebatch(rule, R.config, 'psychometric', params=params)
            y_sample = R.f_y_from_x(task.x)
            y_sample_loc = R.f_y_loc(y_sample)

        perf = get_perf(y_sample, task.y_loc)
        print('Performance {:0.3f}'.format(np.mean(perf)))

        # Compute the overall performance.
        # Importantly, discard trials where no decision was made to one of the choices
        loc_cor = task.y_loc[-1] # last time point, correct locations
        loc_err = (loc_cor+np.pi)%(2*np.pi)
        choose_cor = (get_dist(y_sample_loc[-1] - loc_cor) < 0.3*np.pi).sum()
        choose_err = (get_dist(y_sample_loc[-1] - loc_err) < 0.3*np.pi).sum()
        perf = choose_cor/(choose_cor+choose_err)

        # Compute the proportion of choosing choice 1, while maintaining the batch_shape
        tar1_locs_ = np.reshape(tar1_locs, batch_shape)
        tar2_locs_ = np.reshape(tar2_locs, batch_shape)

        y_sample_loc = np.reshape(y_sample_loc[-1], batch_shape)
        choose1 = (get_dist(y_sample_loc - tar1_locs_) < 0.3*np.pi).sum(axis=0)
        choose2 = (get_dist(y_sample_loc - tar2_locs_) < 0.3*np.pi).sum(axis=0)
        prop1s = choose1/(choose1 + choose2)

        return perf, prop1s, cohs

    def plot_performance_choicetasks(self):
        # Rules for performance
        rules_perf = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]
        lesion_group_list = [None, '1', '2', '12']

        perf_stores = OrderedDict()
        for lesion_group in lesion_group_list:
            perf_stores[lesion_group] = list()

            for rule in rules_perf:
                perf, prop1s, cohs = self.get_performance(rule=rule, lesion_group=lesion_group, n_tar_loc=20)
                perf_stores[lesion_group].append(perf)

            perf_stores[lesion_group] = np.array(perf_stores[lesion_group])

        fs = 6
        width = 0.15
        fig = plt.figure(figsize=(3,1.5))
        ax = fig.add_axes([0.17,0.35,0.8,0.4])
        for i, lesion_group in enumerate(lesion_group_list):
            b0 = ax.bar(np.arange(len(rules_perf))+(i-2)*width, perf_stores[lesion_group],
                   width=width, color=self.colors[lesion_group], edgecolor='none')
        ax.set_xticks(np.arange(len(rules_perf)))
        ax.set_xticklabels([rule_name[r] for r in rules_perf], rotation=25)
        ax.set_xlabel('Tasks', fontsize=fs, labelpad=3)
        ax.set_ylabel('performance', fontsize=fs)
        lg = ax.legend(['Intact']+['Lesion group {:s}'.format(l) for l in ['1','2','12']],
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
            plt.savefig('figure/perf_choiceattend_lesion'+save_addon+'.pdf', transparent=True)
        plt.show()

    def plot_performance_2D(self, rule, lesion_group=None, **kwargs):

        perf, prop1s, cohs = self.get_performance(rule=rule, lesion_group=lesion_group)

        self._plot_performance_2D(prop1s, cohs, rule, lesion_group, **kwargs)

    def _plot_performance_2D(self, prop1s, cohs, rule, lesion_group=None, **kwargs):
        n_coh = len(cohs)

        fs = 6
        fig = plt.figure(figsize=(1.5,1.5))
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.6])
        im = ax.imshow(prop1s, cmap='BrBG', origin='lower',
                       aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax.set_xlabel('Mod 2 coh.', fontsize=fs, labelpad=-3)
        plt.xticks([0, n_coh-1], [cohs[0], cohs[-1]],
                   rotation=0, va='center', fontsize=fs)
        if 'ylabel' in kwargs and kwargs['ylabel']==False:
            plt.yticks([])
        else:
            ax.set_ylabel('Mod 1 coh.', fontsize=fs, labelpad=-3)
            plt.yticks([0, n_coh-1], [cohs[0], cohs[-1]],
                       rotation=0, va='center', fontsize=fs)
        plt.title(rule_name[rule] + '\n' + self.lesion_group_names[lesion_group], fontsize=fs)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)

        if 'colorbar' in kwargs and kwargs['colorbar']==False:
            pass
        else:
            ax = fig.add_axes([0.82, 0.2, 0.03, 0.6])
            cb = plt.colorbar(im, cax=ax, ticks=[0, 1])
            cb.outline.set_linewidth(0.5)
            cb.set_label('Prop. of choice 1', fontsize=fs, labelpad=-3)
            plt.tick_params(axis='both', which='major', labelsize=fs)

        if save:
            plt.savefig('figure/'+rule_name[rule].replace(' ','')+
                        '_perf2D_lesion'+str(lesion_group)+
                        self.save_addon+'.pdf', transparent=True)
        plt.show()

    def plot_fullconnectivity(self):
        # Plot connectivity
        ind_active = self.ind_active
        h_normvar_all = self.h_normvar_all
        # Sort data by labels and by input connectivity
        with Run(save_addon, sigma_rec=0) as R:
            w_in  = R.w_in # for later sorting
            w_out = R.w_out
            config = R.config
        nx, nh, ny = config['shape']
        n_ring = config['N_RING']

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
                # This condition works especially for networks trained only for choiceattend
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

        with Run(save_addon) as R:
            params = R.params
            w_rec  = R.w_rec[ind,:][:,ind]
            w_in   = R.w_in[ind,:]
            w_out  = R.w_out[:,ind]
            b_rec  = R.b_rec[ind, np.newaxis]
            b_out  = R.b_out[:, np.newaxis]

        l = 0.35
        l0 = (1-1.5*l)/nh

        w_in_rule = w_in[:,2*nr+1+np.array([CHOICEATTEND_MOD1,CHOICEATTEND_MOD2])]

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
        colors = [self.colors[group] for group in ['1', '2', '12']] + ['gray']
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
        plt.savefig('figure/choiceattend_connectivity'+save_addon+'.pdf', transparent=True)
        plt.show()

    def plot_inout_connectivity(self, conn_type='input'):
        # Plot connectivity
        # Sort data by labels and by input connectivity

        with Run(save_addon, sigma_rec=0) as R:
            w_in  = R.w_in # for later sorting
            w_out = R.w_out
            w_rec = R.w_rec
            config = R.config
        nx, nh, ny = config['shape']
        n_ring = config['N_RING']

        groups = ['1', '2', '12']

        ############# Plot recurrent connectivity #############################
        if conn_type == 'rec':
            w_rec_group = np.zeros((len(groups), len(groups)))
            for i1, group1 in enumerate(groups):
                for i2, group2 in enumerate(groups):
                    ind1 = self.ind_lesions_orig[group1]
                    ind2 = self.ind_lesions_orig[group2]
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


        ############# Plot input from rule ####################################
        elif conn_type == 'rule':
            rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_MOD1, CHOICE_MOD2, CHOICE_INT]


            w_stores = OrderedDict()

            for group in groups:
                w_store_tmp = list()
                ind = self.ind_lesions_orig[group]
                for rule in rules:
                    w_conn = w_in[ind, 2*n_ring+1+rule].mean(axis=0)
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
            plt.savefig('figure/conn_'+conn_type+'_choiceattend_'+save_addon+'.pdf', transparent=True)
            plt.show()

            return

        ############## Plot input from stim or output to loc ##################
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
            ind_group  = self.ind_lesions_orig[group]
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
        plt.savefig('figure/conn_'+conn_type+'_choiceattend_'+save_addon+'.pdf', transparent=True)

class StateSpaceAnalysis(object):
    def __init__(self, save_addon, lesion_units=None, **kwargs):

        # Default settings
        default_setting = {
            'save_addon'         : save_addon,
            'analyze_threerules' : False,
            'analyze_allunits'   : False,
            'redefine_choice'    : False,
            'regress_product'    : False, # regression of interaction terms
            'z_score'            : True,
            'fast_eval'          : True,
            'surrogate_data'     : False}

        # Update settings with kwargs
        setting = default_setting
        for key, val in kwargs.iteritems():
            setting[key] = val

        print('Current analysis setting:')
        for key, val in default_setting.iteritems():
            print('{:20s} : {:s}'.format(key, str(val)))


        # # If using surrogate data, create random matrix for later use
        # if setting['surrogate_data']:
        #     from scipy.stats import ortho_group
        #     with Run(save_addon, fast_eval=True) as R:
        #         w_rec  = R.w_rec
        #     nh = w_rec.shape[0]
        #     random_ortho_matrix = ortho_group.rvs(dim=nh)


        #################### Computing Neural Activity #########################
        # Get rules and regressors
        rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
        regr_names = ['Choice', 'Mod 1', 'Mod 2', 'Rule']

        n_rule = len(rules)
        n_regr = len(regr_names)

        # Generate task parameters used
        # Target location
        tar1_loc  = np.pi/2
        params, batch_size = self.gen_taskparams(tar1_loc, n_tar=6, n_rep=2)

        x     = list() # Network input
        y_loc = list() # Network target output location

        # Start computing the neural activity
        with Run(save_addon, sigma_rec=0, lesion_units=lesion_units, fast_eval=setting['fast_eval']) as R:
            config = R.config

            for i, rule in enumerate(rules):
                # Generating task information
                task  = generate_onebatch(rule, R.config, 'psychometric', params=params, noise_on=False)
                x.append(task.x)
                y_loc.append(task.y_loc)

            x     = np.concatenate(x    , axis=1)
            y_loc = np.concatenate(y_loc, axis=1)

            # Get neural activity
            H     = R.f_h(x)
            y_sample     = R.f_y(H)
            y_sample_loc = R.f_y_loc(y_sample)

        H_original = H.copy()

        # Get performance and choices
        perfs = get_perf(y_sample, y_loc)
        # y_choice is 1 for choosing tar1_loc, otherwise -1
        y_actual_choice = 2*(get_dist(y_sample_loc[-1]-tar1_loc)<np.pi/2) - 1
        y_target_choice = 2*(get_dist(y_loc[-1]       -tar1_loc)<np.pi/2) - 1

        ###################### Processing activity ##################################

        # Downsample in time
        dt_new = 50
        every_t = int(dt_new/config['dt'])
        # Only analyze the target epoch
        epoch = task.epochs['tar1']
        H = H[epoch[0]:epoch[1],...][::every_t,...]

        config['dt_new'] = dt_new

        nt, nb, nh = H.shape

        # Analyze all units or only active units
        if setting['analyze_allunits']:
            ind_active = range(nh)
        else:
            if 'ind_active' in kwargs:
                ind_active = kwargs['ind_active']
            else:
                # The way to select these units are important
                # ind_active = np.where(H.reshape((-1, nh)).var(axis=0) > 1e-4)[0]
                ind_active = np.where(H[-1].var(axis=0) > 1e-3)[0]
                # ind_active = np.where(H[2:].var(axis=1).mean(axis=0) > 1e-4)[0]

        H = H.reshape((-1, nh))

        H = H[:, ind_active]
        nh = len(ind_active) # new nh

        # Z-scoring response across time and trials (can have a strong impact on results)
        if setting['z_score']:
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
        tar_mod1_cohs = params['tar1_mod1_strengths'] - params['tar2_mod1_strengths']
        tar_mod2_cohs = params['tar1_mod2_strengths'] - params['tar2_mod2_strengths']

        # Regressors (Choice, Mod1 Cohs, Mod2 Cohs, Rule)
        Regrs = np.zeros((n_rule * batch_size, n_regr))
        Regrs[:, 0] = y_target_choice
        Regrs[:, 1] = np.tile(tar_mod1_cohs/tar_mod1_cohs.max(), n_rule)
        Regrs[:, 2] = np.tile(tar_mod2_cohs/tar_mod2_cohs.max(), n_rule)
        Regrs[:, 3] = np.repeat([1, -1], batch_size) # +1 for Att 1, -1 for Att 2

        # Get unique regressors
        Regrs_new = np.vstack({tuple(row) for row in Regrs})
        # Sort it
        ind_sort = np.lexsort(Regrs_new[:, ::-1].T)
        Regrs_new = Regrs_new[ind_sort, :]

        n_cond = Regrs_new.shape[0]

        H_new = np.zeros((nt, n_cond, nh))

        for i_cond in range(n_cond):
            regr = Regrs_new[i_cond, :]

            if setting['redefine_choice']:
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

        self.config     = config
        self.setting    = setting
        self.Regrs      = Regrs_new
        self.H          = H_new
        self.H_tran     = H_new_tran
        self.H_original = H_original
        self.regr_names = regr_names
        self.coef       = coef_maxt
        self.rules      = rules
        self.ind_active = ind_active
        self.tar1_loc   = tar1_loc
        self.q          = q
        self.lesion_units = lesion_units
        self.colors     = dict(zip([None, '1', '2', '12'],
                           sns.xkcd_palette(['orange', 'green', 'pink', 'sky blue'])))


    @staticmethod
    def gen_taskparams(tar1_loc, n_tar, n_rep):
        # Generate task parameterse for state-space analysis
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

    def sort_ind_bygroup(self):
        # Sort ind by group 1, 2, 12, and others
        # ind_group are indices for the current matrix, not original
        # ind_active_group are for original matrix

        ua = UnitAnalysis(self.setting['save_addon'])

        ind_group = dict()
        ind_active_group = dict()

        for group in ['1', '2', '12', None]:
            if group is not None:
                # Find all units here that belong to group 1, 2, or 12 as defined in UnitAnalysis
                ind_group[group] = [k for k, ind in enumerate(self.ind_active) if ind in ua.ind_lesions_orig[group]]
            else:
                ind_othergroup = np.concatenate(ua.ind_lesions_orig.values())
                ind_group[group] = [k for k, ind in enumerate(self.ind_active) if ind not in ind_othergroup]

            # Transform to original matrix indices
            ind_active_group[group] = [self.ind_active[k] for k in ind_group[group]]

        return ind_group, ind_active_group

    def sort_coefs_bygroup(self, coefs=None):
        # Sort coefs by group 1, 2, 12, and others

        if coefs is None:
            coefs = dict() # Initialize

        ind_group, _ = self.sort_ind_bygroup()

        for group in ['1', '2', '12', None]:
            if group not in coefs:
                coefs[group] = self.coef[ind_group[group], :]
            else:
                coefs[group] = np.concatenate((coefs[group], self.coef[ind_group[group], :]))

        return coefs

    def plot_betaweights(self, coefs, fancy_color=False):
        '''
        Plot beta weights
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

            for group in [None, '12', '1', '2']:
                if group is not None:
                    if fancy_color:
                        color = self.colors[group]
                    else:
                        color = 'gray'
                else:
                    color = 'gray'
                # Find all units here that belong to group 1, 2, or 12 as defined in UnitAnalysis
                ax.plot(coefs[group][:,i], coefs[group][:,j], 'o', color=color, ms=1.5, mec='white', mew=0.2)

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
            save_name = 'beta_weights_sub'
            if fancy_color:
                save_name = save_name + '_color'
            plt.savefig(os.path.join('figure',save_name+'.pdf'), transparent=True)
        plt.show()

    def get_slowpoints(self):
        ####################### Find Fixed & Slow Points ######################
        if self.setting['redefine_choice']:
            ValueError('Finding slow points is invalid when choices are redefined')

        if self.lesion_units is not None:
            ValueError('Lesion units not supported yet')

        # Find Fixed points
        # Choosing starting points
        self.fixed_points_trans_all = dict()
        self.slow_points_trans_all  = dict()

        # Looping over rules
        for rule in self.rules:

            print(rule_name[rule])

            ######################## Find Fixed Points ########################

            # Zero-coherence network input
            params = {'tar1_locs' : [self.tar1_loc],
                      'tar2_locs' : [np.mod(self.tar1_loc+np.pi, 2*np.pi)],
                      'tar1_mod1_strengths' : [1],
                      'tar2_mod1_strengths' : [1],
                      'tar1_mod2_strengths' : [1],
                      'tar2_mod2_strengths' : [1],
                      'tar_time'    : 600}

            task        = generate_onebatch(rule, self.config, 'psychometric', noise_on=False, params=params)
            epoch       = task.epochs['tar1']
            input_coh0  = task.x[epoch[1]-1, 0, :]

            # Get two starting points from averaged activity when choice is +1 or -1
            tmp = list()
            # Looping over choice
            for ch in [-1, 1]:
                # Last time point activity for all conditions with tihs choice
                h_tmp = self.H[-1, self.Regrs[:,0]==ch, :]

                # Get index of the condition that is farthest away from origin
                ind = np.argmax(np.sum(h_tmp**2, 1))
                tmp.append(h_tmp[ind, :])
            tmp = np.array(tmp)

            # Notice H is z-scored. Now get the starting point in original space
            if self.setting['z_score']:
                tmp *= self.stdh
                tmp += self.meanh


            nh_orig = self.config['shape'][1]
            start_points = np.zeros((2, nh_orig))
            print(start_points.shape)
            # Re-express starting points in original space
            start_points[:, self.ind_active] = tmp

            # Find fixed points with function find_slowpoints
            res_list = search_slowpoints(save_addon, input=input_coh0,
                                       start_points=start_points, find_fixedpoints=True)

            # Store fixed points in original space, and in z-scored, subsampled space
            fixed_points_raws  = list()
            fixed_points_trans = list()
            for i, res in enumerate(res_list):
                print(res.success, res.message, res.fun)

                # Original space
                fixed_points_raws.append(res.x)

                # Transformed space
                fixed_points = res.x[self.ind_active]
                if self.setting['z_score']:
                    fixed_points -= self.meanh
                    fixed_points /= self.stdh

                # Task-related axes space
                fixed_points_tran = np.dot(fixed_points, self.q)
                fixed_points_trans.append(fixed_points_tran)

            fixed_points_raws  = np.array(fixed_points_raws)
            fixed_points_trans = np.array(fixed_points_trans)
            self.fixed_points_trans_all[rule] = fixed_points_trans


            ######################## Find Slow Points ########################
            # The starting conditions will be equally sampled points in between two fixed points
            n_slow_points = 100 # actual points will be this minus 1
            mix_weight = np.array([np.arange(1,n_slow_points),
                                   n_slow_points-np.arange(1,n_slow_points)], dtype='float').T/n_slow_points

            # Various ways to generate starting points for the search

            # start_points = np.dot(mix_weight, fixed_points_raws)
            start_points = np.dot(mix_weight, start_points)
            # start_points+= np.random.randn(*start_points.shape) # Randomly perturb starting points
            # start_points *= np.random.uniform(0, 2, size=start_points.shape) # Randomly perturb starting points
            # start_points = np.random.rand(100, nh)*3

            # Search slow points with the same input but different starting points
            res_list = search_slowpoints(save_addon, input=input_coh0,
                                       start_points=start_points, find_fixedpoints=False)

            slow_points_trans = list()
            for i, res in enumerate(res_list):
                # Transformed space
                slow_points = res.x[self.ind_active]
                if self.setting['z_score']:
                    slow_points -= self.meanh
                    slow_points /= self.stdh

                # Task-related axes space
                slow_points_tran = np.dot(slow_points, self.q)
                slow_points_trans.append(slow_points_tran)

            slow_points_trans = np.array(slow_points_trans)
            self.slow_points_trans_all[rule] = slow_points_trans

    def get_regr_ind(self, choice=None, coh1=None, coh2=None, rule=None):
        # For given choice, coh1, coh2, rule, get the indices of trials
        ind = np.ones(self.Regrs.shape[0], dtype=bool) # initialize

        if choice is not None:
            ind *= (self.Regrs[:, 0] == choice)

        if coh1 is not None:
            ind *= (self.Regrs[:, 1] == coh1)

        if coh2 is not None:
            ind *= (self.Regrs[:, 2] == coh2)

        if rule is not None:
            j_rule = 1 if rule == CHOICEATTEND_MOD1 else -1
            ind *= (self.Regrs[:, 3] == j_rule)

        return ind

    def plot_statespace(self, plot_slowpoints=True):
        '''
        Plot state space analysis
        :param plot_slowpoints:
        :return:
        '''

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
                                  sharex=True, sharey='row', figsize=(len(self.rules)*1,2))
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
                    anc = [self.H_tran[:,:,pcs[0]].min()+1, self.H_tran[:,:,pcs[1]].max()-5] # anchor point
                    ax.plot([anc[0], anc[0]], [anc[1]-5, anc[1]-1], color='black', lw=1.0)
                    ax.plot([anc[0]+1, anc[0]+5], [anc[1], anc[1]], color='black', lw=1.0)
                    ax.text(anc[0], anc[1], self.regr_names[pcs[0]], fontsize=fs, va='bottom')
                    ax.text(anc[0], anc[1], self.regr_names[pcs[1]], fontsize=fs, rotation=90, ha='right', va='top')


                # Loop over coherences to choice 1, from high to low
                for i, coh in enumerate(np.unique(cohs)[::-1]):

                    # Loop over choices
                    for choice in [1, -1]:

                        if choice == 1:
                            # Solid circles
                            kwargs = {'markerfacecolor' : colors[i], 'linewidth' : 1}
                        else:
                            # Empty circles
                            kwargs = {'markerfacecolor' : 'white', 'linewidth' : 0.5}

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
                ax.plot([i], [0], '.-', color=colors[i], markersize=4, markeredgewidth=0.5, **kwargs)
            ax.axis('off')
            ax.text(2.5, 1, 'Strong Weak Strong', fontsize=5, va='bottom', ha='center')
            # During looping, we use coherence to choice 1 from high to low
            ax.text(2.5, -1, 'To choice 1    To choice 2', fontsize=5, va='top', ha='center')
            ax.set_xlim([-1,6])
            ax.set_ylim([-3,3])

        if save:
            plt.savefig(os.path.join('figure',
        'fixpoint_choicetasks_statespace'+self.setting['save_addon']+'.pdf'), transparent=True)
        plt.show()

    def plot_units_intime(self, plot_individual=False):
        for group in ['1', '2', '12']:
            self.plot_units_intime_bygroup(group, plot_individual)

    def plot_units_intime_bygroup(self, group, plot_individual=False):
        # Plot averaged unit activity in time

        ind_group, ind_active_group = self.sort_ind_bygroup()

        rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

        t_plot = np.arange(self.H.shape[0])*self.config['dt_new']/1000

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

            if not self.setting['z_score']:
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
        'choiceatt_intime_'+self.setting['save_addon']+'_group'+group+'.pdf'), transparent=True)

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


def plot_groupsize(save_type):
    HDIMs = range(150, 1000)
    group_sizes = {key : list() for key in ['1', '2', '12']}
    HDIM_plot = list()
    for HDIM in HDIMs:
        save_addon = save_type+'_'+str(HDIM)
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue
        ua = UnitAnalysis(save_addon)
        for key in ['1', '2', '12']:
            group_sizes[key].append(len(ua.ind_lesions[key]))

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
    plt.savefig('figure/choiceattend_groupsize'+save_type+'.pdf', transparent=True)

def plot_betaweights(save_type):
    HDIMs = range(150, 1000)
    coefs = {}

    for HDIM in HDIMs:
        save_addon = save_type+'_'+str(HDIM)
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue

        ssa = StateSpaceAnalysis(save_addon, lesion_units=None, redefine_choice=True)

        # Update coefficient dictionary
        coefs = ssa.sort_coefs_bygroup(coefs)

    ssa.plot_betaweights(coefs, fancy_color=False)
    ssa.plot_betaweights(coefs, fancy_color=True)

def quick_statespace(save_addon):
    # Quick state space analysis from mode='test'
    rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]
    h_lastts = dict()
    with Run(save_addon, sigma_rec=0, fast_eval=True) as R:
        config = R.config

        for rule in rules:
            task = generate_onebatch(rule=rule, config=config, mode='test')
            h = R.f_h(task.x)
            lastt = task.epochs['tar1'][-1]
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

######################### Connectivity and Lesioning ##########################
save_addon = 'allrule_weaknoise_400'
ua = UnitAnalysis(save_addon)
# ua.plot_inout_connectivity(conn_type='rec')
# ua.plot_inout_connectivity(conn_type='rule')
# ua.plot_inout_connectivity(conn_type='output')
# ua.prettyplot_hist_varprop()
ua.plot_performance_choicetasks()

# rule = CHOICEATTEND_MOD1
# ua.plot_performance_2D(rule=rule)
# for lesion_group in ['1', '2', '12', '1+2']:
#     ua.plot_performance_2D(rule=rule, lesion_group=lesion_group, ylabel=False, colorbar=False)

# ua.plot_fullconnectivity()

# plot_groupsize('allrule_weaknoise')

################### State space ##############################################
# Plot State space
# ssa = StateSpaceAnalysis(save_addon, lesion_units=None)
# ssa.plot_statespace(plot_slowpoints=True)

# Plot beta weights
# save_type = 'allrule_weaknoise'
# plot_betaweights(save_type)

# Plot units in time
# ssa = StateSpaceAnalysis(save_addon, lesion_units=None, z_score=False)
# ssa.plot_units_intime()

# Quick state space analysis
# quick_statespace(save_addon)