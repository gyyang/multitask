"""
Analysis of general properties of tasks
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import theano
import theano.tensor as T
from blocks.model import Model
from blocks.serialization import load_parameters
from sklearn.decomposition import PCA, RandomizedPCA
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work
from task import *
from network import MyNetwork

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'

color_rules = np.array([
            [240,163,255],[0,117,220],[153,63,0],[76,0,92],[25,25,25],[0,92,49],
            [43,206,72],[255,204,153],[128,128,128],[148,255,181],[143,124,0],
            [157,204,0],[194,0,136],[0,51,128],[255,164,5],[255,168,187],[66,102,0],
            [255,0,16],[94,241,242],[0,153,143],[224,255,102],[116,10,255],
            [153,0,0],[255,255,128],[255,255,0],[255,80,5]])/255.

class GeneralAnalysis(object):
    def __init__(self, save_addon, inh_id=None, inh_output=True):
        '''
        save_addon: add on for loading and saving
        inh_id    : Ids of units to inhibit inputs or outputs
        inh_output: If True, the inhibit the output of neurons, otherwise inhibit inputs to neurons
        '''
        ############################## Load Results ###################################
        print('Analyzing network ' + save_addon)
        # Load config
        with open('data/config'+save_addon+'.pkl','rb') as f:
            config = pickle.load(f)

        alpha_original = config['alpha']
        config['dt'] = 1.
        config['alpha'] = config['dt']/TAU # For sample running, we set the time step to be 1ms. So alpha = 1/tau
        x_dim, h_dim, y_dim = config['shape']
        mynet = MyNetwork(config)
        x = T.tensor3('x')
        y_hat = T.tensor3('y_hat')
        y_hat_loc = T.vector('y_hat_loc') # Vector of size batchsize
        c_mask = T.tensor3('c_mask')

        h = mynet.get_h(x)
        y = mynet.get_y_from_h(h)
        y_loc = mynet.popvec.apply(y[-1,:,-config['N_RING']:])
        cost, performance = mynet.get_cost_from_y(y, y_hat, y_hat_loc, c_mask)

        model = Model(cost)
        with open('data/task_multi'+config['save_addon']+'.tar', 'rb') as f:
            p_dict = load_parameters(f)

        if inh_id is not None: # No inhibition of units
            control_array = np.ones(h_dim, dtype=p_dict['/mynetwork/h_rec.W'].dtype)
            control_array[inh_id] = 0

            if inh_output: # Inhibit the unit output
                p_dict['/mynetwork/h_rec.W'] = (control_array*p_dict['/mynetwork/h_rec.W'].T).T
                p_dict['/mynetwork/h_to_o.W'] = (control_array*p_dict['/mynetwork/h_to_o.W'].T).T
            else: # Inhibit the unit input
                p_dict['/mynetwork/h_rec.W'] = control_array*p_dict['/mynetwork/h_rec.W']
                p_dict['/mynetwork/x_to_h.W'] = control_array*p_dict['/mynetwork/x_to_h.W']
                p_dict['/mynetwork/x_to_h.b'] = control_array*p_dict['/mynetwork/x_to_h.b']


        model.set_parameter_values(p_dict)

        f_h = theano.function([x],h)
        f_y = theano.function([h],y)
        f_y_loc = theano.function([y], y_loc)
        f_cost  = theano.function([x, y_hat, y_hat_loc, c_mask], [cost, performance], on_unused_input='warn')

        params = model.get_parameter_dict()
        # Notice this weight is originally used as r*W, so transpose them
        Wrec = params['/mynetwork/h_rec.W'].get_value().T
        Wout = params['/mynetwork/h_to_o.W'].get_value().T
        Win  = params['/mynetwork/x_to_h.W'].get_value().T

        if config['h_type'] == 'leaky_rec_ei':
            dimE = mynet.h_layer.dimE
            dimI = mynet.h_layer.dimI
            Wrec = (np.array([1.]*dimE+[-1.]*dimI))*abs(Wrec)

        self.config = config

        self.f_h  = f_h
        self.f_y  = f_y
        self.f_y_loc = f_y_loc
        self.f_y_loc_from_x = lambda x : f_y_loc(f_y(f_h(x)))
        self.f_cost = f_cost

        self.Wrec = Wrec
        self.Wout = Wout
        self.Win  = Win
        self.p_dict = p_dict

        self.rules = self.config['rule_cost_plot'].keys()

        self.test_ran = False

    def sample_plot(self, rule, save=True, task=None):
        N_RING = self.config['N_RING']
        if task is None:
            # tdim depends on the alpha (\Delta t/\tau)
            task = generate_onebatch(rule=rule, config=self.config, mode='sample', t_tot=1000)
        x_sample = task.x
        h_sample = self.f_h(x_sample)
        y_sample = self.f_y(h_sample)
        y_hat = task.y_hat
        # y_sample = task.y_hat

        fig = plt.figure(figsize=(2.5,5))
        ylabels = ['fix. in', 'stim. mod1', 'stim. mod2', 'units','fix. out', 'out']
        heights = np.array([0.03,0.15,0.15,0.15,0.03,0.15])+0.01
        for i in range(6):
            ax = fig.add_axes([0.15,sum(heights[i+1:]+0.02)+0.1,0.8,heights[i]])
            cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
            plt.xticks([])
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            if i == 0:
                plt.plot(x_sample[:,0,0], color=sns.xkcd_palette(['blue'])[0])
                plt.yticks([0,1],['',''],rotation='vertical')
                plt.ylim([-0.1,1.5])
                plt.title('Rule: '+rule_name[rule],fontsize=7)
            elif i == 1:
                plt.imshow(x_sample[:,0,1:1+N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
            elif i == 2:
                plt.imshow(x_sample[:,0,1+N_RING:1+2*N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                # plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
            elif i == 3:
                plt.imshow(h_sample[:,0,:].T, aspect='auto', cmap=cmap, vmin=0, vmax=3, interpolation='none',origin='lower')
                # plt.yticks([0,self.config['HDIM']-1],['1',str(self.config['HDIM'])],rotation='vertical')
                plt.yticks([])
            elif i == 4:
                plt.plot(y_hat[:,0,0],color=sns.xkcd_palette(['green'])[0])
                plt.plot(y_sample[:,0,0],color=sns.xkcd_palette(['blue'])[0])
                plt.yticks([0.05,0.8],['',''],rotation='vertical')
                plt.ylim([-0.1,1.1])
            elif i == 5:
                plt.imshow(y_sample[:,0,1:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                # plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                plt.xticks([0,500,1000])
                plt.xlabel('Time (ms)',fontsize=7)
                ax.spines["bottom"].set_visible(True)
            plt.ylabel(ylabels[i],fontsize=7)
            ax.get_yaxis().set_label_coords(-0.12,0.5)

        if save:
            plt.savefig('figure/sample_'+rule_name[rule].replace(' ','')+self.config['save_addon']+'.pdf')
        plt.show()
        return task, h_sample, y_sample

    def schematic_plot(self):
        rule = CHOICE_MOD1
        N_RING = self.config['N_RING']
        # tdim depends on the alpha (\Delta t/\tau)
        task = generate_onebatch(rule=rule, config=self.config, mode='sample', t_tot=1000)
        x_sample = task.x
        h_sample = self.f_h(x_sample)
        y_sample = self.f_y(h_sample)

        y_hat = task.y_hat

        # Plot Stimulus
        fig = plt.figure(figsize=(1.2,1.5))
        heights = np.array([0.06,0.25,0.25])
        for i in range(3):
            ax = fig.add_axes([0.2,sum(heights[i+1:]+0.1)+0.05,0.7,heights[i]])
            cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
            plt.xticks([])
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            if i == 0:
                plt.plot(x_sample[:,0,0], color=sns.xkcd_palette(['blue'])[0])
                plt.yticks([0,1],['',''],rotation='vertical')
                plt.ylim([-0.1,1.5])
                plt.title('Fixation input', fontsize=7, y=0.9)
            elif i == 1:
                plt.imshow(x_sample[:,0,1:1+N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$','',r'360$\degree$'],rotation='vertical')
                plt.title('Stimulus Mod 1', fontsize=7, y=0.9)
            elif i == 2:
                plt.imshow(x_sample[:,0,1+N_RING:1+2*N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                plt.yticks([0,(N_RING-1)/2,N_RING-1],['','',''],rotation='vertical')
                plt.title('Stimulus Mod 2', fontsize=7, y=0.9)
            ax.get_yaxis().set_label_coords(-0.12,0.5)
        plt.savefig('figure/schematic_input.pdf',transparent=True)
        plt.show()

        # Plot Rule Inputs
        fig = plt.figure(figsize=(1.2, 0.5))
        ax = fig.add_axes([0.2,0.1,0.7,0.65])
        cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
        plt.xticks([])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        X = x_sample[:,0,1+2*N_RING:]
        plt.imshow(X.T, aspect='auto', vmin=0, vmax=1, cmap=cmap, interpolation='none',origin='lower')
        plt.yticks([0,X.shape[-1]-1],['1',str(X.shape[-1])],rotation='vertical')
        plt.title('Rule inputs', fontsize=7, y=0.9)
        ax.get_yaxis().set_label_coords(-0.12,0.5)
        plt.savefig('figure/schematic_rule.pdf',transparent=True)
        plt.show()


        # Plot Units
        fig = plt.figure(figsize=(1.2, 1.0))
        ax = fig.add_axes([0.2,0.1,0.7,0.65])
        cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
        plt.xticks([])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.imshow(h_sample[:,0,:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
        plt.yticks([0,A.config['HDIM']-1],['1',str(A.config['HDIM'])],rotation='vertical')
        plt.title('Recurrent units', fontsize=7, y=0.9)
        ax.get_yaxis().set_label_coords(-0.12,0.5)
        plt.savefig('figure/schematic_units.pdf',transparent=True)
        plt.show()


        # Plot Outputs
        fig = plt.figure(figsize=(1.2,1))
        heights = np.array([0.1,0.45])+0.01
        for i in range(2):
            ax = fig.add_axes([0.2,sum(heights[i+1:]+0.15)+0.1,0.7,heights[i]])
            cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
            plt.xticks([])
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            if i == 0:
                plt.plot(y_hat[:,0,0],color=sns.xkcd_palette(['green'])[0])
                plt.plot(y_sample[:,0,0],color=sns.xkcd_palette(['blue'])[0])
                plt.yticks([0.05,0.8],['',''],rotation='vertical')
                plt.ylim([-0.1,1.1])
                plt.title('Fixation output', fontsize=7, y=0.9)

            elif i == 1:
                plt.imshow(y_sample[:,0,1:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$','',r'360$\degree$'],rotation='vertical')
                plt.xticks([])
                plt.title('Output', fontsize=7, y=0.9)

            ax.get_yaxis().set_label_coords(-0.12,0.5)

        plt.savefig('figure/schematic_outputs.pdf',transparent=True)
        plt.show()

    def plot_activation(self, comparison=False):
        h_type = self.config['h_type']
        if h_type != 'leaky_rec_ca':
            raise NotImplementedError

        if comparison:
            x_plot = np.linspace(0.2,0.5,100)
            y_plot = np.log(1+np.exp(40.*(x_plot-0.4)))/40*310.
            y_plot2 = 310*(x_plot-(125./310.))/(1-np.exp(-0.16*310*(x_plot-125./310.)))
            fig = plt.figure(figsize=(1.5,1.5))
            ax = fig.add_axes([0.25,0.25,0.7,0.7])
            ax.plot(x_plot, y_plot,'black', label='SoftPlus')
            ax.plot(x_plot, y_plot2,'blue', label='Chance-Abbott')
            plt.xlabel('input', fontsize=7, labelpad=1)
            plt.ylabel('activity', fontsize=7, labelpad=1)
            plt.locator_params(nbins=3)
            lg = ax.legend(ncol=1,bbox_to_anchor=(-0.05,1),
                           fontsize=7,loc=2,frameon=False)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.savefig('figure/activation_function_compare'+h_type+'.pdf', transparent=True)
            plt.show()

        x_plot = np.linspace(0,0.5,100)
        y_plot = np.log(1+np.exp(40.*(x_plot-0.4)))/40*310.
        fig = plt.figure(figsize=(0.15,0.15))
        ax = fig.add_axes([0.1,0.1,0.8,0.8])
        ax.plot(x_plot, y_plot,'black', label='SoftPlus')
        plt.ylim(bottom=-5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.savefig('figure/activation_function'+h_type+'.pdf', transparent=True)
        plt.show()

    def get_costs(self, rules=None):
        if rules is None:
            rules = self.rules
        costs = list()
        performances = list()
        for rule in rules:
            print('Testing rule '+rule_name[rule])
            task = generate_onebatch(rule, self.config, 'random', batch_size=1000)
            # task = generate_onebatch(rule=rule, config=self.config, mode='test')

            cost, performance = self.f_cost(task.x, task.y_hat, task.y_hat_loc[-1,:], task.c_mask)
            costs.append(cost)
            performances.append(performance)

        return np.array(costs), np.array(performances)

    def plot_trainingprogress(self, rule_plot=None, save=True):
        # Plot Training Progress


        rule_cost_plot = self.config['rule_cost_plot']
        rule_performance_plot = self.config['rule_performance_plot']
        batch_plot = self.config['batch_plot']
        batch_size = self.config['batch_size']
        traintime_plot = self.config['traintime_plot']

        fig = plt.figure(figsize=(5,3))
        d1, d2 = 0.01, 0.35
        ax1 = fig.add_axes([0.15,0.5+d1,   0.5,d2])
        ax2 = fig.add_axes([0.15,0.5-d1-d2,0.5,d2])
        lines = list()
        labels = list()

        x_plot = np.array(batch_plot)*batch_size/1000.
        if rule_plot == None:
            rule_plot = rule_cost_plot.keys()

        for i, rule in enumerate(rule_plot):
            line = ax1.plot(x_plot,np.log10(rule_cost_plot[rule]),color=color_rules[i%26])
            ax2.plot(x_plot,rule_performance_plot[rule],color=color_rules[i%26])
            lines.append(line[0])
            labels.append(rule_name[rule])

        ax1.tick_params(axis='both', which='major', labelsize=7)
        ax2.tick_params(axis='both', which='major', labelsize=7)

        ax2.set_ylim([-0.05, 1.05])
        ax2.set_xlabel('Total trials (1,000)',fontsize=7)
        ax2.set_ylabel('performance',fontsize=7)
        ax1.set_ylabel('log(cost)',fontsize=7)
        ax1.set_xticklabels([])
        ax1.set_title('Training time {:0.0f} s'.format(traintime_plot[-1]),fontsize=7)
        lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.65,0.5),
                        fontsize=7,labelspacing=0.3,loc=6)
        plt.setp(lg.get_title(),fontsize=7)
        if save:
            plt.savefig('figure/Training_Progress'+self.config['save_addon']+'.pdf', transparent=True)
        plt.show()

    def psychometric(self, rule, **kwargs):
        if rule in [CHOICE_MOD1, CHOICE_MOD2]:
            self.psychometric_choice(**kwargs)
        elif rule in [CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY]:
            self.psychometric_delaychoice(**kwargs)
        elif rule in [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]:
            self.psychometric_choiceattend(**kwargs)
        elif rule == CHOICE_INT:
            self.psychometric_choiceint(**kwargs)
        elif rule == INTREPRO:
            self.psychometric_intrepro(**kwargs)
        else:
            raise ValueError('Unsupported rule for psychometric analysis')

    def psychometric_choice(self, **kwargs):
        print('Starting standard analysis of the CHOICE task...')

        n_tar_loc = 120
        n_tar = 7
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.16
        tar1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar/(n_tar-1)
        tar2_strengths = 2 - tar1_strengths

        ydatas = list()
        tar_times = [100, 400, 800]
        for tar_time in tar_times:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_strengths' : tar1_strengths,
                      'tar2_strengths' : tar2_strengths,
                      'tar_time'    : tar_time}

            task  = generate_onebatch(CHOICE_MOD1, self.config, 'psychometric', params=params)
            y_loc_sample = self.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample, batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            ydatas.append(choose1/(choose1 + choose2))

        xdatas = [tar_str_range*(-1+2*np.arange(n_tar)/(n_tar-1))] * len(tar_times)

        self.plot_psychometric_choice(xdatas, ydatas,
                                      labels=[str(t) for t in tar_times],
                                      colors=sns.dark_palette("light blue", 3, input="xkcd"),
                                      legtitle='Stim. time (ms)',rule=CHOICE_MOD1, **kwargs)

    def psychometric_delaychoice(self, **kwargs):
        print('Starting standard analysis of the CHOICEDELAY task...')

        n_tar_loc = 120
        n_tar = 7
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.75
        tar1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar/(n_tar-1)
        tar2_strengths = 2 - tar1_strengths

        tar1_ons = 200
        tar1_offs = 400

        dtars = [200,1500,3000] # tar1 offset and tar2 onset time difference
        # dtars = [2800,3000,3200] # tar1 offset and tar2 onset time difference
        ydatas = list()
        for dtar in dtars:
            tar2_ons  = tar1_offs + dtar
            tar2_offs = tar2_ons + 200
            params = {'tar1_locs'    : tar1_locs,
                      'tar2_locs'    : tar2_locs,
                      'tar1_strengths' : tar1_strengths,
                      'tar2_strengths' : tar2_strengths,
                      'tar1_ons'     : tar1_ons,
                      'tar1_offs'    : tar1_offs,
                      'tar2_ons'     : tar2_ons,
                      'tar2_offs'    : tar2_offs,
                      }

            task  = generate_onebatch(CHOICEDELAY_MOD1, self.config, 'psychometric', params=params)
            y_loc_sample = self.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample, batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            print(choose1)
            print(choose2)
            ydatas.append(choose1/(choose1 + choose2))

        xdatas = [tar_str_range*(-1+2*np.arange(n_tar)/(n_tar-1))] * len(dtars)

        self.plot_psychometric_choice(xdatas, ydatas,
                                      labels=[str(t) for t in dtars],
                                      colors=sns.dark_palette("light blue", 3, input="xkcd"),
                                      legtitle='Delay (ms)',rule=CHOICEDELAY_MOD1, **kwargs)

    def psychometric_choiceattend(self, **kwargs):
        print('Starting standard analysis of the CHOICEATTEND task...')
        from task import get_dist

        n_tar_loc = 12 # increase repeat by increasing this
        n_tar = 7
        batch_size = n_tar_loc * n_tar**2
        batch_shape = (n_tar_loc,n_tar,n_tar)
        ind_tar_loc, ind_tar_mod1, ind_tar_mod2 = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.25
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

        task  = generate_onebatch(CHOICEATTEND_MOD1, self.config, 'psychometric', params=params)
        y_loc_sample = self.f_y_loc_from_x(task.x)
        y_loc_sample = np.reshape(y_loc_sample, batch_shape)

        tar1_locs_ = np.reshape(tar1_locs, batch_shape)
        tar2_locs_ = np.reshape(tar2_locs, batch_shape)

        choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
        choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
        prop1s = choose1/(choose1 + choose2)

        xdatas = [tar_str_range*(-1+2*np.arange(n_tar)/(n_tar-1))]*2
        ydatas = [prop1s.mean(axis=k) for k in [1,0]]

        self.plot_psychometric_choice(xdatas, ydatas,
                                      labels=['Attend', 'Ignore'],
                                      colors=sns.color_palette("Set2",2),
                                      legtitle='Modality',rule=CHOICEATTEND_MOD1, **kwargs)

    def psychometric_choiceint(self, **kwargs):
        print('Starting standard analysis of the CHOICEINT task...')

        n_tar_loc = 100 # increase repeat by increasing this
        n_tar = 7
        batch_size = n_tar_loc * n_tar
        batch_shape = (n_tar_loc,n_tar)
        ind_tar_loc, ind_tar1_strength = np.unravel_index(range(batch_size),batch_shape)

        tar1_locs = 2*np.pi*ind_tar_loc/n_tar_loc
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)

        tar_str_range = 0.25
        tar1_strengths = (1-tar_str_range/2)+tar_str_range*ind_tar1_strength/(n_tar-1)
        tar2_strengths = 2 - tar1_strengths

        xdatas = list()
        ydatas = list()
        for mod_strength in [(1,0), (0,1), (1,1)]:
            params = {'tar1_locs' : tar1_locs,
                      'tar2_locs' : tar2_locs,
                      'tar1_mod1_strengths' : tar1_strengths*mod_strength[0],
                      'tar2_mod1_strengths' : tar2_strengths*mod_strength[0],
                      'tar1_mod2_strengths' : tar1_strengths*mod_strength[1],
                      'tar2_mod2_strengths' : tar2_strengths*mod_strength[1],
                      'tar_time'    : 800}

            task  = generate_onebatch(CHOICE_INT, self.config, 'psychometric', params=params)
            y_loc_sample = self.f_y_loc_from_x(task.x)
            y_loc_sample = np.reshape(y_loc_sample, batch_shape)

            tar1_locs_ = np.reshape(tar1_locs, batch_shape)
            tar2_locs_ = np.reshape(tar2_locs, batch_shape)

            choose1 = (get_dist(y_loc_sample - tar1_locs_) < 0.3*np.pi).sum(axis=0)
            choose2 = (get_dist(y_loc_sample - tar2_locs_) < 0.3*np.pi).sum(axis=0)
            prop1s = choose1/(choose1 + choose2)

            xdatas.append(tar_str_range*(-1+2*np.arange(n_tar)/(n_tar-1)))
            ydatas.append(prop1s)

        fits = self.plot_psychometric_choice(
            xdatas,ydatas, labels=['1 only', '2 only', 'both'],
            colors=sns.color_palette("Set2",3),
            legtitle='Modality',rule=CHOICE_INT, **kwargs)
        sigmas = [fit[1] for fit in fits]
        print('Fit sigmas:')
        print(sigmas)

    def psychometric_intrepro(self, **kwargs):
        n_tar_loc = 360
        # intervals = [700]
        # intervals = [500, 600, 700, 800, 900, 1000]
        intervals = np.linspace(500, 1000, 10)
        mean_sample_intervals = list()
        for interval in intervals:
            batch_size = n_tar_loc
            tar_mod1_locs  = 2*np.pi*np.arange(n_tar_loc)/n_tar_loc

            params = {'tar_mod1_locs'  : tar_mod1_locs,
                      'interval'       : interval}

            task  = generate_onebatch(INTREPRO, self.config, 'psychometric', params=params)
            h_test = self.f_h(task.x)
            y = self.f_y(h_test)

            sample_intervals = list() # sampled interval test
            for i_batch in range(batch_size):
                try: ##TODO: Temporary solution
                    # Setting the threshold can be tricky, but doesn't impact the overall results
                    sample_interval = np.argwhere(y[:,i_batch,0]<0.3)[0]-task.epochs['tar2'][1]
                    sample_intervals.append(sample_interval)
                except IndexError:
                    # print i_batch
                    pass
            mean_sample_intervals.append(np.mean(sample_intervals))

        fig = plt.figure(figsize=(2,1.5))
        ax = fig.add_axes([0.25,0.25,0.65,0.65])
        ax.plot(intervals, mean_sample_intervals, 'o-', markersize=3.5, color=sns.xkcd_palette(['blue'])[0])
        ax.plot(intervals, intervals, color=sns.xkcd_palette(['black'])[0])
        plt.xlim([intervals[0]-50,intervals[-1]+50])
        plt.ylim([intervals[0]-50,intervals[-1]+50])
        plt.xlabel('Sample Interval (ms)',fontsize=7)
        plt.ylabel('Production interval (ms)',fontsize=7)
        plt.title('Rule '+rule_name[INTREPRO], fontsize=7)
        plt.locator_params(nbins=5)
        ax.tick_params(axis='both', which='major', labelsize=7)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/analyze_'+rule_name[INTREPRO].replace(' ','')+'_performance.pdf', transparent=True)
        plt.show()

    def plot_psychometric_choice(self, xdatas, ydatas, labels, colors, **kwargs):
        '''
        For the class of choice tasks, plot the psychometric function for a list of
        xdatas and ydatas
        '''
        fig = plt.figure(figsize=(2,1.5))
        ax = fig.add_axes([0.25,0.25,0.65,0.65])
        fits = list()
        for i in range(len(xdatas)):
            # Analyze performance of the choice tasks
            cdf_gaussian = lambda x, mu, sigma : stats.norm.cdf(x, mu, sigma)

            xdata = xdatas[i]
            ydata = ydatas[i]
            (mu,sigma), _ = curve_fit(cdf_gaussian, xdata, ydata, bounds=([-0.5,0.01],[0.5,1]))
            fits.append((mu,sigma))
            x_plot = np.linspace(xdata[0],xdata[-1],100)
            ax.plot(x_plot, cdf_gaussian(x_plot,mu,sigma), label=labels[i],
                    linewidth=1, color=colors[i])
            ax.plot(xdata, ydata, 'o', markersize=3.5, color=colors[i])

        plt.xlabel('Target 1 - Target 2',fontsize=7)
        plt.ylim([-0.05,1.05])
        plt.xlim([xdata[0]*1.1,xdata[-1]*1.1])
        plt.yticks([0,0.5,1])
        if 'no_ylabel' in kwargs and kwargs['no_ylabel']:
            plt.yticks([0,0.5,1],['','',''])
        else:
            plt.ylabel('Proportion of choice 1',fontsize=7)
        plt.title(rule_name[kwargs['rule']], fontsize=7)
        plt.locator_params(axis='x', nbins=5)
        ax.tick_params(axis='both', which='major', labelsize=7)

        if len(xdatas)>1:
            if len(kwargs['legtitle'])>10:
                bbox_to_anchor = (0.6, 1.1)
            else:
                bbox_to_anchor = (0.5, 1.1)
            leg = plt.legend(title=kwargs['legtitle'],fontsize=7,frameon=False,
                             bbox_to_anchor=bbox_to_anchor,labelspacing=0.3)
            plt.setp(leg.get_title(),fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/analyze_'+rule_name[kwargs['rule']].replace(' ','')+'_performance.pdf', transparent=True)
        plt.show()
        return fits

    def plot_singleneuron_DMCGO(self, neuron_ids):
        rule = DMCGO
        tar1_locs = np.array([0.1,0.3,0.5,0.7,0.9,
                              1.1,1.3,1.5,1.7,1.9])*np.pi # try avoid border case
        matchnogo = 0
        matchs = (1 - matchnogo)*np.ones(len(tar1_locs)) # make sure the response is Go
        tar2_locs = (tar1_locs+np.pi*(1-matchs))%(2*np.pi)
        params = {'tar1_locs' : tar1_locs,
                  'tar2_locs' : tar2_locs}
        task = generate_onebatch(rule=rule, config=self.config,
                                 mode='psychometric', params=params, add_rule=DMCGO)
        x_sample = task.x
        h_sample = self.f_h(x_sample)
        sample_cat = tar1_locs<np.pi

        t_left = task.epochs['tar1'][0]
        #t_right = task.epochs['delay1'][1]
        t_right = None
        for neuron_id in neuron_ids:
            h_plot = h_sample[t_left:t_right,:,neuron_id]
            #if h_plot.max()>0.5:
            if True:
                fig = plt.figure(figsize=(1.5,1.))
                ax = fig.add_axes([0.2,0.2,0.7,0.7])
                _ = ax.plot(h_plot[:,sample_cat==1], 'blue')
                _ = ax.plot(h_plot[:,sample_cat==0], 'red')
                ax.tick_params(axis='both', which='major', labelsize=7)
                plt.locator_params(nbins=3)
                plt.title('neuron {:d}'.format(neuron_id), fontsize=7)
                ylim_top = np.max((0.5, h_plot.max()))
                plt.ylim([0, ylim_top])
                plt.show()

    def get_dprime_DMC(self):
        rule = DMCGO
        tar1_locs = np.array([0.1,0.3,0.5,0.7,0.9,
                              1.1,1.3,1.5,1.7,1.9])*np.pi # try avoid border case
        tar2_locs = np.ones(len(tar1_locs))*0.5*np.pi
        params = {'tar1_locs' : tar1_locs,
                  'tar2_locs' : tar2_locs}
        task = generate_onebatch(rule=rule, config=self.config,
                                 mode='psychometric', params=params, add_rule=DMCGO)
        x_sample = task.x
        h_sample = self.f_h(x_sample)
        sample_cat = tar1_locs<np.pi

        t_analyze = task.epochs['tar1'][-1]
        h_analyze = h_sample[t_analyze,:,:]
        # h_analyze *= np.random.uniform(0.8,1.1,size=h_analyze.shape)
        h_analyze += np.random.randn(*h_analyze.shape)*0.1

        h_cat0 = h_analyze[sample_cat==0,:]
        h_cat1 = h_analyze[sample_cat==1,:]

        h_cat0_mean = h_cat0.mean(axis=0)
        h_cat1_mean = h_cat1.mean(axis=0)
        h_units = np.logical_or(h_cat0_mean>0.02, h_cat1_mean>0.02)

        dprimes = abs(np.mean(h_cat0,axis=0)-np.mean(h_cat1,axis=0))/np.sqrt((np.var(h_cat0,axis=0)+np.var(h_cat1,axis=0))/2)
        return dprimes

    def plot_hunits_obsolete(self, rule, ind_unit):
        plt.figure()
        _ = plt.plot(h_samples[rule][:,:,ind_unit])
        plt.xlabel('Time (ms)')
        plt.ylabel('Rec units activity')

        #plt.savefig('figure/'+rule_name[rule]+'_sampleactivity.pdf')

    def plot_weight_obsolete(self, Wrec):
        #---------------------------------------------------------------------------------
        # Display connection matrix
        #---------------------------------------------------------------------------------

        f_norm = lambda x: x/abs(x).max()
        Wrec_plot = f_norm(Wrec*(Wrec>0)) + f_norm(Wrec*(Wrec<0))
        plt.figure()
        plt.imshow(Wrec_plot, cmap='seismic_r', interpolation='nearest', aspect='equal',
                   vmin=-1, vmax=1)
        plt.savefig('figure/weight.pdf')

    def get_vars_obsolete(self, rules, norm_by='none'):
        for j, rule in enumerate(rules):
            print 'Running rule ' + rule_name[rule] + '...'
            task = generate_onebatch(rule=rule, config=self.config, mode='test',
                                     t_tot=1000)
            data = self.f_h(task.x)[:,:,:] # (Time, Batch, Units)
            data = data[range(0,data.shape[0],int(data.shape[0]/100)),:,:] # Downsample in time

            if j == 0:
                # (Rule, Time, Batch, Units)
                data_all = np.zeros((len(rules), data.shape[0], data.shape[1], data.shape[2]))
            data_all[j] = data

        # data_all = (Rule, Time, Batch, Units)
        # These are quite arbitrary now
        vars = dict()
        vars['task'] = data_all.mean(axis=1).mean(axis=1).var(axis=0)
        vars['stim'] = data_all.var(axis=2).mean(axis=0).mean(axis=0)
        vars['time'] = data_all.mean(axis=2).var(axis=1).mean(axis=0)
        vars['all']  = data_all.reshape((-1, data.shape[2])).var(axis=0)
        mean_all = data_all.mean(axis=0).mean(axis=0).mean(axis=0)


        from scipy.stats import spearmanr
        fig = plt.figure(figsize=(4,2))
        for left, xtype, ytype in zip([0.15,0.65],['stim','stim'],['task','time']):
            ax = fig.add_axes([left,0.2,0.3,0.6])
            if norm_by == 'total_var':
                x_scatter, y_scatter = vars[xtype]/vars['all'], vars[ytype]/vars['all']
            elif norm_by == 'total_mean':
                x_scatter, y_scatter = np.sqrt(vars[xtype])/mean_all, np.sqrt(vars[ytype])/mean_all
            elif norm_by == 'none':
                x_scatter, y_scatter = vars[xtype], vars[ytype]

            ax.scatter(x_scatter, y_scatter, s=1.5, color=sns.xkcd_rgb['denim blue'])
            plt.xlabel(xtype + ' std', fontsize=7)
            plt.ylabel(ytype + ' std', fontsize=7)
            plt.xlim([0-x_scatter.max()*0.1, x_scatter.max()*1.1])
            plt.ylim([0-y_scatter.max()*0.1, y_scatter.max()*1.2])
            plt.title('Normalized by '+norm_by,fontsize=7)

            rho, pval = spearmanr(x_scatter, y_scatter)
            ax.text(0.05, 0.9, r'$\rho={:0.3f}, P={:0.1e}$'.format(rho,pval), transform=ax.transAxes, fontsize=7)

            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/unitstd_normby'+norm_by+self.config['save_addon']+'.pdf')
        plt.show()

    def get_sumstat(self, rules=None, do_PCA=True):
        '''
        Get summary statistics for a list of rules or for epochs within a rule
        '''
        if rules is None:
            rules = self.rules
        if FIXATION in rules:
            rules.pop(rules.index(FIXATION))

        if len(rules)>20:
            raise ValueError('Length of rules too long for data type epoch')

        t_start = time.time()

        # Get summary statistics
        datas_byrule = OrderedDict()
        datas_byepoch = OrderedDict()

        zeromean = True
        normalize_variance = True

        for rule in rules:
            # print 'Running rule ' + rule_name[rule] + '...'
            task = generate_onebatch(rule=rule, config=self.config, mode='test')
            h_test = self.f_h(task.x) # (Time, Batch, Units)

            # Downsample in time, always keep the same number of points
            data = h_test[range(0,h_test.shape[0],int(h_test.shape[0]/100)),:,:]
            data = data.reshape((-1,data.shape[-1]))
            datas_byrule[rule] = data

            epoch_names = [i for i in task.epochs.keys() if 'fix' not in i]
            for epoch_name in epoch_names:
                epoch = task.epochs[epoch_name]
                data_epoch = h_test[epoch[0]:epoch[1],:,:]

                # Downsampling in time. Notice the kind of downsampling has big impact
                # uniformly downsample in time
                # data_epoch = data_epoch[::10,:,:]

                # take only last time points of an epoch
                data_epoch = data_epoch[-10:,:,:]

                datas_byepoch[(rule,epoch_name)] = data_epoch.reshape((-1,data_epoch.shape[-1]))

        for data_type, datas in zip(['rule', 'epoch'], [datas_byrule, datas_byepoch]):
            h_tbmeans = OrderedDict()

            print('Processing data type ' + data_type)
            # Preprocess before PCA
            data_names = datas.keys()
            for data_name in data_names:
                # averaged across time and batch, do so before normalizing
                h_tbmeans[data_name] = np.mean(datas[data_name], axis=0)

                if zeromean: # It's important this zero-mean is done before the concatenation
                    datas[data_name] = datas[data_name] - datas[data_name].mean(axis=0)
                if normalize_variance:
                    datas[data_name] = datas[data_name]/np.std(datas[data_name])

            fname = 'data/h_tbmeans_'+data_type+self.config['save_addon']+'.pkl'
            with open(fname, 'wb') as f: # Store summary statistics
                pickle.dump(h_tbmeans, f)

            if do_PCA:
                # Combine pair of data
                data_name_combs = list()
                for i in range(len(data_names)-1):
                    for j in range(i+1, len(data_names)):
                        data_name_comb = (data_names[i],data_names[j])
                        data_name_combs.append(data_name_comb)
                        # Concatenate along the batch dimension
                        datas[data_name_comb] = np.concatenate((datas[data_names[i]],
                                                                datas[data_names[j]]),axis=0)
                exp_vars = OrderedDict()
                # Do PCA
                print 'Starting PCA...'
                data_names += data_name_combs # extend list
                for data_name in data_names:
                    data = datas[data_name]
                    pca = PCA(n_components=data.shape[-1], whiten=False)
                    pca.fit(data)
                    exp_vars[data_name] = pca.explained_variance_ratio_
                    # sumstat['comps'+str(data_name)] = pca.components_ # this takes up too much space

                fname = 'data/exp_vars_'+data_type+self.config['save_addon']+'.pkl'
                with open(fname,'wb') as f:
                    pickle.dump(exp_vars, f)

            print('Time taken {:0.2f} s'.format(time.time()-t_start))

    def run_test(self, rules=None):
        self.h_tests = OrderedDict()
        self.epoch_names = OrderedDict()
        self.epochs = OrderedDict()
        if rules is None:
            rules = self.rules
            
        for rule in rules:
            task = generate_onebatch(rule=rule, config=self.config, mode='test')
            h_test = self.f_h(task.x) # (Time, Batch, Units)
            self.h_tests[rule] = h_test
            self.epoch_names[rule] = [i for i in task.epochs.keys() if 'fix' not in i]
            for epoch_name in self.epoch_names[rule]:
                epoch = task.epochs[epoch_name]
                self.epochs[(rule, epoch_name)] = epoch

        self.test_ran = True

        # For later
        self.ids = dict()
        self.counts = dict()

    def get_ids(self, condition):
        if not self.test_ran:
            self.run_test()
        # Get neuron indices for each rule/epoch that satisfy certain condition
        # Here the condition is mean>0.2, and std/mean<0.5
        ind_wheres = OrderedDict()
        for rule in self.rules:
            h_test = self.h_tests[rule]
            for epoch_name in self.epoch_names[rule]:
                epoch = self.epochs[(rule, epoch_name)]
                data_epoch = h_test[epoch[0]:epoch[1],:,:]

                d_mean = data_epoch[-1,:,:].mean(axis=0)
                d_std = data_epoch[-1,:,:].std(axis=0)

                # The condition
                if condition == 'epochselective':
                    ind = (d_mean>0.1)*(d_std/d_mean<0.5)
                elif condition == 'stimselective':
                    ind = (d_mean>0.1)*(d_std/d_mean>0.5)
                else:
                    raise ValueError('Unknown condition')

                ind_wheres[(rule,epoch_name)] = np.where(ind)[0]

        ids = OrderedDict()
        counts = OrderedDict()
        # For this epoch_type, gather all neuron_id that satisfied the conditions
        # epoch_types = ['go', 'tar']
        epoch_types = ['go1', 'tar1', 'delay1']
        for epoch_type in epoch_types:
            print('Epoch type ' + epoch_type)
            ind_where_all = np.array([], dtype=int)
            for name, ind_where in ind_wheres.iteritems():
                rule, epoch_name = name
                if epoch_type in epoch_name:
                    # print(rule_name[rule] + ' ' + epoch_name),
                    # print(ind_where)
                    ind_where_all = np.concatenate((ind_where_all, ind_where))

            id_unique, count_unique = np.unique(ind_where_all, return_counts=True) # across rules
            ind_sort = np.argsort(count_unique)[::-1]
            ids[epoch_type] = id_unique[ind_sort]
            counts[epoch_type] = count_unique[ind_sort]

            print('Neuron ids:'),
            print(ids[epoch_type])
            rules0 = [rule for rule in rules if epoch_type in self.epoch_names[rule]]
            print('Total eligible tasks: {:d}'.format(len(rules0)))
            print('Tasks satisfied conditions'),
            print((counts[epoch_type]))

            ind_tasknonselective = counts[epoch_type]>5 ##TODO: setting the threshold 5 is kind of arbitrary
            ids[epoch_type] = ids[epoch_type][ind_tasknonselective]
            counts[epoch_type] = counts[epoch_type][ind_tasknonselective]

        self.ids[condition] = ids
        self.counts[condition] = counts

    def get_subconn(self, plot_fig=False):
        ids = self.ids
        counts = self.counts
        # Get the sub-connectivity
        epoch_types = ['go1', 'tar1']
        epoch_types += ['all']
        ids['all'] = range(self.Wrec.shape[0])

        self.w_sub = OrderedDict()
        self.w_in  = OrderedDict()
        self.w_out = OrderedDict()
        for epoch_type0 in epoch_types:
            for epoch_type1 in epoch_types:
                # Connection weight from type 0 to 1
                self.w_sub[(epoch_type0, epoch_type1)] = \
                    self.Wrec[ids[epoch_type1],:][:,ids[epoch_type0]].mean()

            self.w_out[epoch_type0] = self.Wout[:,ids[epoch_type0]]
            self.w_in[epoch_type0]  = self.Win[ids[epoch_type0],:]

        if plot_fig:
            ename = {'go1' : 'go', 'tar1' : 'fix', 'delay1' : 'delay', 'all' : 'all'}
            n = len(self.w_sub.values())
            plt.figure()
            plt.bar(np.arange(n)-0.25, self.w_sub.values(), width=0.5)
            xticklabels = [ename[e0]+' to '+ename[e1] for e0, e1 in self.w_sub.keys()]
            plt.xticks(np.arange(n), xticklabels)
            plt.ylabel('Mean connection weight')
            plt.show()

    def plot_neuron_condition(self, rules=None, save_fig=True, condition='epochselective',
                            epoch_types=None, plot_ids=None):
        # Plot neuron activity and tuning that satisfy some conditions
        if not self.test_ran:
            self.run_test()

        if condition not in self.ids:
            self.get_ids(condition=condition)
        ids = self.ids[condition]
        counts = self.counts[condition]

        if rules is None:
            rules = self.rules
        if epoch_types is None:
            epoch_types = ['go1', 'tar1']
        if plot_ids is None:
            plot_ids = [0,0]
        for epoch_type, plot_id in zip(epoch_types, plot_ids):
            # Plot units that are non-selective to tasks
            neuron_id = ids[epoch_type][plot_id]
            count = counts[epoch_type][plot_id]
            # Only use rule where the epoch_type is one of the epochs
            rules0 = [rule for rule in rules if epoch_type in self.epoch_names[rule]]

            print('Analyzing unit {:d} '.format(neuron_id))
            print('Satisfied conditions in {:d}/{:d} tasks'.format(count, len(rules0)))

            self.plot_singleneuron_intime(neuron_id, rules, epoch_type, save_fig=save_fig)
            if epoch_type == 'tar1':
                self.plot_singleneuron_tuning(neuron_id, rules, epoch_type, save_fig=save_fig)

    def plot_singleneuron_intime(self, neuron_id, rules, epoch_type=None, save_fig=False):

        h_max = 0
        for rule in rules:
            h_max = max((h_max, np.max(self.h_tests[rule][:,:,neuron_id])))

        for i, rule in enumerate(rules):
            h_test = self.h_tests[rule]
            fig = plt.figure(figsize=(1.5,1))
            ax = fig.add_axes([0.3,0.25,0.6,0.6])
            ax.set_color_cycle(sns.color_palette("husl", h_test.shape[1]))
            _ = ax.plot(h_test[:,:,neuron_id])

            if epoch_type is not None:
                epoch = self.epochs[(rule, epoch_type)]
                epoch0 = epoch[0] if epoch[0] is not None else 0
                epoch1 = epoch[1] if epoch[1] is not None else h_test.shape[0]
                ax.plot([epoch0, epoch1], [h_max*1.15]*2,
                        color='black',linewidth=1.5)
                save_name = 'figure/trace_'+rule_name[rule]+epoch_type+self.config['save_addon']+'.pdf'
            else:
                save_name = 'figure/trace_unit'+str(neuron_id)+rule_name[rule]+self.config['save_addon']+'.pdf'

            plt.xticks([0,2000])
            if i==(len(rules)-1):
                plt.xlabel('Time (ms)', fontsize=7, labelpad=-5)
            else:
                ax.set_xticklabels([])
            plt.ylabel('activitity (a.u.)', fontsize=7)
            plt.title('Unit {:d} '.format(neuron_id) + rule_name[rule], fontsize=7)
            plt.ylim(np.array([-0.1, 1.2])*h_max)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.locator_params(axis='y', nbins=4)
            if save_fig:
                plt.savefig(save_name, transparent=True)
            plt.show()

    def plot_singleneuron_tuning(self, neuron_id, rules, epoch_type=None, save_fig=False):
        h_plots = list()
        h_max = 0
        for rule in rules:
            assert rule in [INHGO, INHREMAP] # Only support these two for now
            h_test = self.h_tests[rule]
            epoch = self.epochs[(rule, epoch_type)]
            # epoch0 = epoch[0] if epoch[0] is not None else 0
            epoch1 = epoch[1] if epoch[1] is not None else h_test.shape[0]
            h_max = max((h_max, np.max(self.h_tests[rule][epoch1,:,neuron_id])))
            h_plots.append(h_test[epoch1,:,neuron_id])

        for i, rule in enumerate(rules):
            save_name = 'figure/tuning_unit'+str(neuron_id)+rule_name[rule]+epoch_type+A.config['save_addon']+'.pdf'

            fig = plt.figure(figsize=(1.5,1))
            ax = fig.add_axes([0.3,0.3,0.6,0.5])
            _ = ax.plot(h_plots[i], color=sns.xkcd_palette(['black'])[0])
            plt.ylim([0, h_max])
            n = h_test.shape[1]
            plt.xticks([0,(n-1)/2,n-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'])
            if i==(len(rules)-1):
                plt.xlabel('Stimulus', fontsize=7, labelpad=0)
            else:
                ax.set_xticklabels([])
            plt.ylabel('activitity (a.u.)', fontsize=7)
            plt.title('Unit {:d} '.format(neuron_id) + rule_name[rule] + ' ' + epoch_type, fontsize=7)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.locator_params(axis='y', nbins=4)
            if save_fig:
                plt.savefig(save_name, transparent=True)
            plt.show()

    def plot_meanstd(self, rules=None):
        if not self.test_ran:
            self.run_test()
        if rules is None:
            rules = self.rules
        d_mean_all = np.array([])
        d_std_all = np.array([])

        d_mean_all_lastt = np.array([])
        d_std_all_lastt = np.array([])

        for rule in rules:
            for epoch_name in self.epoch_names[rule]:
                epoch = self.epochs[(rule, epoch_name)]
                data_epoch = self.h_tests[rule][epoch[0]:epoch[1],:,:]

                d_mean_batch = data_epoch.mean(axis=1) # average across stimulus (batch)
                d_std_batch = data_epoch.std(axis=1)
                d_mean_all_lastt = np.concatenate((d_mean_all_lastt, d_mean_batch[-1,:]))
                d_std_all_lastt = np.concatenate((d_std_all_lastt, d_std_batch[-1,:]))

                d_mean_all = np.concatenate((d_mean_all, d_mean_batch.flatten()))
                d_std_all = np.concatenate((d_std_all, d_std_batch.flatten()))

                #
                # fig = plt.figure(figsize=(1.5,1.5))
                # ax = fig.add_axes([0.2,0.2,0.7,0.7])
                # ax.scatter(d_mean, d_std, s=5)
                # plt.xlabel('mean', fontsize=7)
                # plt.ylabel('std', fontsize=7)
                # plt.title('Rule '+rule_name[rule]+' Epoch '+epoch_name, fontsize=7)
                # ax.tick_params(axis='both', which='major', labelsize=7)
                # plt.show()

        color = sns.xkcd_palette(['black'])[0]
        fig = plt.figure(figsize=(1.5,1.5))
        ax = fig.add_axes([0.2,0.2,0.7,0.7])
        ax.scatter(d_mean_all_lastt, d_std_all_lastt, s=5, alpha=0.2, c=color, edgecolors='face', marker='.')
        ax.plot([0,d_std_all_lastt.max()*0.9],[0,d_std_all_lastt.max()*0.9],'black')
        plt.xlabel('stim mean', fontsize=7, labelpad=1)
        plt.ylabel('stim std', fontsize=7, labelpad=1)
        plt.locator_params(nbins=3)
        plt.xlim([-d_mean_all_lastt.max()*0.1,d_mean_all_lastt.max()*1.1])
        plt.ylim([-d_std_all_lastt.max()*0.1,d_std_all_lastt.max()*1.1])
        # plt.title('All rules, epochs, units', fontsize=7)
        #plt.xlim([0,3])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/meanstd_scatter'+self.config['save_addon']+'.pdf', transparent=True)
        plt.show()

        # d_mean_plot, d_std_plot = d_mean_all_lastt, d_std_all_lastt
        d_mean_plot, d_std_plot = d_mean_all, d_std_all

        color = sns.xkcd_palette(['cerulean'])[0]
        # ind = d_mean_all>0.5
        prop = 0.05
        ind = np.argsort(d_mean_plot)
        ind = ind[-int(prop*len(ind)):]
        fig = plt.figure(figsize=(1.5,1.5))
        ax = fig.add_axes([0.3,0.3,0.6,0.5])
        hist, bins_edge = np.histogram(d_std_plot[ind]/d_mean_plot[ind], range=(0,3), bins=30)
        ax.bar(bins_edge[:-1], hist/1000, width=bins_edge[1]-bins_edge[0], color=color, edgecolor='none')
        # _ = ax.hist(d_std_plot[ind]/d_mean_plot[ind], range=(0,3), bins=30, color=color, edgecolor='none')
        plt.xlabel('stimulus std/mean', fontsize=7)
        plt.ylabel('count (1,000)', fontsize=7)
        plt.xticks([0,1,2,3])
        plt.locator_params(axis='y', nbins=2)
        plt.title('Mean activity in top {:d}%'.format(int(100*prop)), fontsize=7)
        plt.xlim([0,3])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/meanvsstd_hist'+self.config['save_addon']+'.pdf', transparent=True)
        plt.show()

    def plot_input_weight_hist(self):
        N_RING = self.config['N_RING']
        Win = self.Win
        fix_input = Win[:, 0].flatten()
        stim_input = Win[:, 1:2*N_RING+1].flatten()
        rule_input = Win[:, 2*N_RING+1:].flatten()


        plot_inputs = [fix_input, stim_input, rule_input]
        colors = sns.xkcd_palette(['blue','red','black'])
        labels = ['fix', 'stim', 'rule']

        fig = plt.figure(figsize=(1.0,1.0))
        ax = fig.add_axes([0.35,0.35,0.6,0.6])
        for i in range(len(plot_inputs)):
            hist, bins_edge = np.histogram(plot_inputs[i], range=(0,3), bins=70)
            ax.plot(bins_edge[:-1], np.cumsum(hist/hist.sum()), color=colors[i], label=labels[i], linewidth=1)

        plt.xlabel('weight', fontsize=7, labelpad=0)
        plt.ylabel('cum. prob.', fontsize=7, labelpad=1)
        plt.xticks([0,3])
        plt.yticks([0,1])
        plt.xlim([-0.1,3.1])
        plt.ylim([0,1.1])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        lg = ax.legend(title='from',ncol=1,bbox_to_anchor=(1,0.8),
                        fontsize=6,labelspacing=0.2,loc=1,frameon=False)
        plt.setp(lg.get_title(),fontsize=6)
        plt.savefig('figure/input_weight_hist'+self.config['save_addon']+'.pdf', transparent=True)
        plt.show()




rules = [FIXATION, GO, INHGO, DELAYGO, CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,
         CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,
         REMAP, INHREMAP, DELAYREMAP, DELAYMATCHGO, DELAYMATCHNOGO]


def plot_subconn_fixgo(rules):
    HDIMs = range(190,208)
    #HDIMs = [200]
    N_RINGs = [16] * len(HDIMs)

    # HDIMs = [200]
    # N_RINGs = [16]

    j = 0
    w_subconns = OrderedDict()
    w_ins      = OrderedDict()
    w_outs     = OrderedDict()
    for HDIM, N_RING in zip(HDIMs, N_RINGs):
        save_addon = 'latest_'+str(HDIM)+'_'+str(N_RING)
        A = GeneralAnalysis(save_addon=save_addon)
        A.run_test(rules)
        A.get_subconn(plot_fig=False)

        for key in A.w_sub:
            if j == 0:
                w_subconns[key] = [A.w_sub[key]]
            else:
                w_subconns[key] += [A.w_sub[key]]

        for key in A.w_out:
            if j == 0:
                w_ins[key]  = A.w_in[key]
                w_outs[key] = A.w_out[key]
            else:
                w_ins[key]  = np.concatenate((w_ins[key], A.w_in[key]), axis=0)
                w_outs[key] = np.concatenate((w_outs[key], A.w_out[key]), axis=1)

        j += 1

    ################################## Plot sub-connectivity #####################
    n_network = len(HDIMs)

    w_subconn_mean = OrderedDict()
    w_subconn_se   = OrderedDict()
    for key, val in w_subconns.iteritems():
        w_subconn_mean[key] = np.mean(val)
        w_subconn_se[key]   = np.std(val)/np.sqrt(len(val))

    ename = {'go1' : 'go', 'tar1' : 'fix', 'delay1' : 'delay', 'all' : 'all'}
    color = sns.xkcd_palette(['cerulean'])[0]
    n = len(w_subconn_mean.values())
    fig = plt.figure(figsize=(3,1.5))
    ax = fig.add_axes([0.2,0.3,0.7,0.6])
    ax.bar(np.arange(n)-0.25, w_subconn_mean.values(), width=0.5,
           yerr=w_subconn_se.values(), color=color, ecolor=color, edgecolor='none')
    xticklabels = [ename[e0]+' to '+ename[e1] for e0, e1 in w_subconn_mean.keys()]
    plt.xticks(np.arange(n), xticklabels, rotation=45)
    plt.ylabel('Mean conn. weight (a.u.)', fontsize=7)
    plt.yticks([0,-0.5,-1])
    plt.ylim([np.min(w_subconn_mean.values())*1.1,0.1])
    plt.title('Average of {:d} networks'.format(n_network), fontsize=7)
    ax.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig('figure/fixgo_subconnectivity_averaged.pdf', transparent=True)
    plt.show()

    epoch_types = ['go1', 'tar1', 'all']
    W = np.zeros((3,3))
    for i0, epoch_type0 in enumerate(epoch_types):
        for i1, epoch_type1 in enumerate(epoch_types):
            W[i1,i0] = w_subconn_mean[(epoch_type0,epoch_type1)]
    vmax = np.ceil(abs(W).max()*10)/10
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([0.2,0.2,0.6,0.6])
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    im = ax.imshow(W, interpolation='nearest', cmap=cmap,
                   vmin=-vmax, vmax=vmax, origin='lower', aspect='auto')
    plt.xticks([0,1,2],[ename[e] for e in epoch_types])
    plt.yticks([0,1,2],[ename[e] for e in epoch_types])
    plt.xlabel('from', fontsize=7,labelpad=1)
    plt.ylabel('to', fontsize=7,labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=7)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, ticks=[-vmax,vmax])
    cb.set_label('mean conn. weight', fontsize=7, labelpad=-7)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.savefig('figure/fixgo_subconnectivity_averaged_matrix.pdf', transparent=True)
    plt.show()

    ###################### Plot input/output connectivity #####################
    w_ins_mean  = OrderedDict()
    w_ins_se    = OrderedDict()
    w_outs_mean = OrderedDict()
    w_outs_se   = OrderedDict()
    for key in w_ins.keys():
        w_ins_mean[key]  = np.mean(w_ins[key],axis=0)
        w_ins_se[key]    = np.std(w_ins[key],axis=0)/np.sqrt(w_ins[key].shape[0])
        w_outs_mean[key] = np.mean(w_outs[key],axis=1)
        w_outs_se[key]   = np.std(w_outs[key],axis=1)/np.sqrt(w_outs[key].shape[1])

    for j in [0, 1]:
        if j == 0:
            plot_mean, plotse, xlabel, lgtitle = w_ins_mean, w_ins_se, 'input', 'to'
        else:
            plot_mean, plotse, xlabel, lgtitle = w_outs_mean, w_outs_se, 'output', 'from'
        n = A.config['N_RING']+1
        fig = plt.figure(figsize=(2.5,1.5))
        ax = fig.add_axes([0.2,0.25,0.5,0.6])
        for key, val in plot_mean.iteritems():
            ax.errorbar(range(n), val[:n], fmt='o-', yerr=plotse[key][:n], label=ename[key], markersize=3)
        # get handles
        handles, labels = ax.get_legend_handles_labels()
        # remove the errorbars
        handles = [h[0] for h in handles]
        # use them in the legend
        lg = ax.legend(handles, labels, loc='upper left',numpoints=1, title=lgtitle,
                       ncol=1, bbox_to_anchor=(1.1,1.0), fontsize=7,labelspacing=0.3)
        plt.setp(lg.get_title(),fontsize=7)
        plt.xlim([-1,n])
        plt.xticks([0, int(n/2)], ['fix.', 'ring'])
        plt.xlabel(xlabel + ' units', fontsize=7, labelpad=2)
        plt.ylabel('Mean conn. weight (a.u.)', fontsize=7)
        plt.locator_params(axis='y', nbins=3)
        plt.title('Average of {:d} networks'.format(n_network), fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.savefig('figure/fixgo_'+xlabel+'connectivity_averaged.pdf', transparent=True)
        plt.show()

def plot_standardanalysis():
    HDIM = 200
    N_RING = 16
    save_addon = 'dmc_'+str(HDIM)+'_'+str(N_RING)
    A = GeneralAnalysis(save_addon=save_addon)
    A.plot_neuron_condition(rules=[INHGO, INHREMAP], condition='epochselective', plot_ids=[0,0])
    A.plot_meanstd()
    A.plot_neuron_condition(rules=[INHGO], condition='stimselective', epoch_types = ['tar1'], plot_ids=[7])

def test_init():
    HDIM = 200
    N_RING = 16
    config = {'h_type'      : 'leaky_rec_ca',
              'alpha'       : 0.2, # \Delta t/tau
              'dt'          : 0.2*TAU,
              'HDIM'        : HDIM,
              'N_RING'      : N_RING,
              'shape'       : (1+2*N_RING+N_RULE, HDIM, N_RING+1),
              'save_addon'  : 'chanceabbott_'+str(HDIM)+'_'+str(N_RING)}
    x = T.tensor3('x')
    y_hat = T.tensor3('y_hat')
    y_hat_loc = T.vector('y_hat_loc') # Vector of size batchsize
    c_mask = T.tensor3('c_mask')
    mynet = MyNetwork(config)
    cost, performance = mynet.cost(x, y_hat, y_hat_loc, c_mask)
    cost.name = 'cost'
    performance.name = 'performance'
    model = Model(cost)
    mynet.initialize()

    h = mynet.get_h(x)
    y = mynet.get_y_from_h(h)
    y_loc = mynet.popvec.apply(y[-1,:,-config['N_RING']:])
    cost, performance = mynet.cost(x, y_hat, y_hat_loc, c_mask)

    f_h = theano.function([x],h)
    f_y = theano.function([h],y)
    f_y_loc = theano.function([y], y_loc)
    f_perf  = theano.function([x, y_hat_loc], performance, on_unused_input='warn')

    task = generate_onebatch(rule=CHOICEATTEND_MOD2, config=config, mode='sample', t_tot=1000)

    h_sample = f_h(task.x)
    y_sample = f_y(h_sample)
    plt.plot(h_sample[:,0,:])
    plt.show()

    plt.plot(y_sample[:,0,:])
    plt.show()

def plot_finalperformance():
    # Initialization. Dictionary comprehension.
    HDIM, N_RING = 200, 16
    save_type  = 'latestei' # should be named latest, temporary now
    save_addon = save_type+'_'+str(HDIM)+'_'+str(N_RING)
    with open('data/config'+save_addon+'.pkl','rb') as f:
        config = pickle.load(f)
    rule_cost_plot = config['rule_cost_plot']
    final_cost        = {k: [] for k in rule_cost_plot}
    final_performance = {k: [] for k in rule_cost_plot}
    HDIM_plot = list()
    # Recording performance and cost for networks
    HDIMs = range(1000)
    for HDIM in HDIMs:
        save_addon = save_type+'_'+str(HDIM)+'_'+str(N_RING)
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue
        with open(fname,'rb') as f:
            config = pickle.load(f)
        rule_cost_plot = config['rule_cost_plot']
        rule_performance_plot = config['rule_performance_plot']
        for key in rule_performance_plot.keys():
            final_performance[key] += [float(rule_performance_plot[key][-1])]
            final_cost[key]        += [float(rule_cost_plot[key][-1])]
        HDIM_plot.append(HDIM)
    
    n_trial = config['batch_plot'][-1]*config['batch_size']
    x_plot = HDIM_plot
    rule_plot = None
    if rule_plot == None:
        rule_plot = rule_performance_plot.keys()
    
    fig = plt.figure(figsize=(5,3))
    d1, d2 = 0.01, 0.35
    ax1 = fig.add_axes([0.15,0.5+d1,   0.5,d2])
    ax2 = fig.add_axes([0.15,0.5-d1-d2,0.5,d2])
    lines = list()
    labels = list()
    for i, rule in enumerate(rule_plot):
        line = ax1.plot(x_plot,np.log10(final_cost[rule]),color=color_rules[i%26])
        ax2.plot(x_plot,final_performance[rule],color=color_rules[i%26])
        lines.append(line[0])
        labels.append(rule_name[rule])
    
    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)
    
    ax2.set_ylim(top=1.05)
    ax2.set_xlabel('Number of Recurrent Units',fontsize=7)
    ax2.set_ylabel('performance',fontsize=7)
    ax1.set_ylabel('log(cost)',fontsize=7)
    ax1.set_xticklabels([])
    ax1.set_title('After {:.1E} trials'.format(n_trial),fontsize=7)
    lg = fig.legend(lines, labels, title='Rule',ncol=1,bbox_to_anchor=(0.65,0.5),
                    fontsize=7,labelspacing=0.3,loc=6)
    plt.setp(lg.get_title(),fontsize=7)
    plt.savefig('figure/FinalCostPerformance_'+save_type+'.pdf', transparent=True)
    plt.show()

if __name__ == '__main__':
    save_addon = 'latestei_'+str(200)+'_'+str(16)
    # A = GeneralAnalysis(save_addon=save_addon)
    # A.plot_trainingprogress()
    #A.sample_plot(DMCGO)
    # A.get_sumstat(do_PCA=True)
    # A.schematic_plot()
    # A.plot_activation()
    # A.psychometric(CHOICE_MOD1)
    # A.psychometric(CHOICEATTEND_MOD1, no_ylabel=True)
    # A.psychometric(CHOICE_INT, no_ylabel=True)
    # A.psychometric(CHOICEDELAY_MOD1)
    
    # plot_subconn_fixgo(A.rules)
    # A.run_test()
    # A.plot_meanstd()
    # A.plot_tasknonselective(rules=[INHGO, INHREMAP], save_fig=True, plot_ids=[0,1])


    # Interval Reproduction Task
    # HDIM = 200
    # N_RING = 16
    # save_addon = 'intrepro_'+str(HDIM)+'_'+str(N_RING)
    # A = GeneralAnalysis(save_addon=save_addon)
    # _ = A.sample_plot(INTREPRO)
    # A.psychometric(INTREPRO)
    # A.plot_trainingprogress('all')


#==============================================================================
# save_addon = 'latest_'+str(200)+'_'+str(16)
# A = GeneralAnalysis(save_addon=save_addon)
# rule = DMCGO
# 
# a = 10 # Has to be 10 for now
# ind_tar1_loc, ind_tar2_loc = np.unravel_index(range(a**2),(a,a))
# 
# tar_loc_list = np.pi*np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])
# tar1_locs = tar_loc_list[ind_tar1_loc]
# tar2_locs = tar_loc_list[ind_tar2_loc]
# 
# # This should give cost, performance decoupling
# # in this case, cost is low, but performance is zero
# # tar1_locs = [1.1*np.pi]
# # tar2_locs = [1.1*np.pi]
# 
# start = time.time()
# costs = list()
# performances = list()
# 
# n_trial = 50
# for tar1_loc, tar2_loc in zip(tar1_locs, tar2_locs):
#     print(tar1_loc, tar2_loc)
#     params = {'tar1_locs' : np.array([tar1_loc]*n_trial),
#               'tar2_locs' : np.array([tar2_loc]*n_trial)}
#     task = generate_onebatch(rule=rule, config=A.config,
#                              mode='psychometric', params=params, add_rule=DMCGO)
#     x_sample = task.x
#     cost_sample, performance_sample = A.f_cost(x_sample, task.y_hat, task.y_hat_loc[-1,:], task.c_mask)
#     costs.append(cost_sample)
#     performances.append(performance_sample)
#     
# A.sample_plot(rule=rule, save=False, task=task)
# 
# costs = np.array(costs).reshape((a,a))
# performances = np.array(performances).reshape((a,a))
# 
# print(time.time()-start)
# 
# for data_plot, data_name in zip([costs, performances],['cost', 'performance']):
#     fig = plt.figure(figsize=(4,4))
#     ax = fig.add_axes([0.2,0.2,0.6,0.6])
#     im = ax.imshow(data_plot, aspect='equal', cmap='hot',
#                            interpolation='nearest',origin='lower')
#     plt.ylabel('tar1 loc', fontsize=7)
#     plt.xlabel('tar2 loc', fontsize=7)
#     plt.tick_params(axis='both', which='major', labelsize=7)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cb = plt.colorbar(im, cax=cax)
#     cb.set_label(data_name, fontsize=7, labelpad=3)
#     plt.tick_params(axis='both', which='major', labelsize=7)
#     plt.show()
# 
#==============================================================================
