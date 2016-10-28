"""
Test out the dimensionality analysis
"""

from __future__ import division

import os
import numpy as np
import pickle
from collections import OrderedDict
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work
from task import *

from general_analysis import GeneralAnalysis

colors = np.array([
            [240,163,255],[0,117,220],[153,63,0],[76,0,92],[25,25,25],[0,92,49],
            [43,206,72],[255,204,153],[128,128,128],[148,255,181],[143,124,0],
            [157,204,0],[194,0,136],[0,51,128],[255,164,5],[255,168,187],[66,102,0],
            [255,0,16],[94,241,242],[0,153,143],[224,255,102],[116,10,255],
            [153,0,0],[255,255,128],[255,255,0],[255,80,5]])/255.

class SumStatAnalysis(object):
    def __init__(self, data_type, save_addon):

        with open('data/config'+save_addon+'.pkl','rb') as f:
            config = pickle.load(f)
        self.save_addon_original = save_addon

        config['save_addon'] = data_type+config['save_addon']

        # Interpret the stored data, see general analysis for more information
        fname = 'data/h_tbmeans_'+config['save_addon']+'.pkl'
        with open(fname,'rb') as f:
            h_tbmeans = pickle.load(f)
        h_dim = h_tbmeans.values()[0].shape[0]
        self.h_tbmeans = h_tbmeans
        self.h_dim = h_dim
        self.config = config


        fname = 'data/exp_vars_'+config['save_addon']+'.pkl'
        with open(fname,'rb') as f:
            exp_vars = pickle.load(f)

        data_names = exp_vars.keys()

        sumevs = dict()
        for data_name in data_names:
            sumevs[data_name] = np.cumsum(exp_vars[data_name])
            sumevs[data_name] = np.insert(sumevs[data_name],0,0) # insert 0 in the beginning



        if data_type == 'rule':
            rules = list()
            for data_name in data_names:
                if isinstance(data_name,int):
                    rules.append(data_name)
            data_singlenames = rules
            self.rules = rules
        elif data_type == 'epoch':
            epochs = list()
            for data_name in data_names:
                # For epoch data_type, data_name is either a tuple, or a tuple of tuples
                if isinstance(data_name[0],int): # If tuple, then (rule, epoch)
                    epochs.append(data_name)
            data_singlenames = epochs
            self.epochs = epochs


        self.data_type = data_type
        self.exp_vars = exp_vars

        self.data_names = data_names
        self.sumevs = sumevs

        self.data_singlenames = data_singlenames
        self.fit_exponential()

    def fit_exponential(self, div_by_base=False):
        # Fit summed explained variance
        f = lambda x,p : 1-np.exp(-x/p[0])*p[1]
        obj_func = lambda p,x,y : np.sum((y-f(x,p))**2)

        fit_params = OrderedDict()
        dims = OrderedDict()
        for data_name in self.data_names:
            y = self.sumevs[data_name]
            x = np.arange(len(y))
            p0 = [5, 0.7]
            bounds = [(1,self.h_dim),(0,1)]
            res = minimize(obj_func, p0, args=(x,y), bounds=bounds,method='SLSQP')
            fit_params[data_name] = res.x
            dims[data_name] = res.x[0]

        if div_by_base:
            if self.data_type == 'rule':
                # Normalize everything by the dimension of FIXATION task
                base_name = FIXATION
            elif self.data_type == 'epoch':
                # Normalize everything by the dimension of fixation epoch
                base_name = (self.data_names[0][0],'fix1')

            dims0 = dims[base_name]
            for data_name in self.data_names:
                dims[data_name] /= dims0

        # if plot_fig:
        #     # Plot data with fitting
        #     colors = [plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(set(rules)))]
        #     plt.figure()
        #     for data_name in self.data_names:
        #         x_plot = np.linspace(0,len(y),100)
        #         if isinstance(data_name, int):
        #             label = rule_name[data_name]
        #             plt.plot(x,self.sumevs[data_name],'o', color=colors[data_name])
        #             plt.plot(x_plot,f(x_plot, fit_params[data_name]),
        #                      color=colors[data_name], label=label)
        #         else:
        #             label = rule_name[data_name[0]] + '+' + rule_name[data_name[1]]
        #     plt.legend(bbox_to_anchor=(1,0.5))
        #     plt.xlabel('PCs')
        #     plt.ylabel('Sum of explained variance')
        #     plt.show()

        self.dims = dims
        self.dims_method = 'exp'
        return dims

    def fit_scaling(self):
        if self.data_type == 'rule':
            # Normalize everything by the dimension of FIXATION task
            base_name = FIXATION
        elif self.data_type == 'epoch':
            # Normalize everything by the dimension of fixation epoch
            base_name = (self.data_names[0][0],'fix1')

        # Fit curves by scaling the first curve (curve for FIXATION)
        y0 = self.sumevs[base_name]
        x0 = np.arange(len(y0))

        f = interp1d(x0, y0, bounds_error=False, fill_value=1., assume_sorted=True)
        sqe = lambda alpha, y: np.sum((f(x0/alpha)-y)**2)

        dims = OrderedDict()
        for data_name in self.data_names:
            y1 = self.sumevs[data_name]
            bounds = [(0.5,10)]
            res = minimize(sqe, [1.2], args=(y1,), bounds=bounds, method='SLSQP')
            dim = res.x[0]
            dims[data_name] = dim

        return dims

    def plot_ranking(self):
        assert self.data_type == 'rule' # only work for data_type == rule now
        dim_tasks = list()
        for rule in self.rules:
            dim_tasks.append(self.dims[rule])

        ind = np.argsort(dim_tasks)[::-1]

        fig = plt.figure(figsize=(1.5,2))
        ax = fig.add_axes([0.6,0.15,0.35,0.8])
        ax.plot([dim_tasks[i] for i in ind],range(len(self.rules)), 'o-', color=sns.xkcd_palette(['cerulean'])[0], markersize=3)
        rule_ticks = [rule_name[self.rules[i]] for i in ind]
        plt.yticks(range(len(self.rules)),rule_ticks, rotation=0, ha='right', fontsize=6)
        plt.ylim([-0.5, len(self.rules)-0.5])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.xlabel('Task Dim.', fontsize=7, labelpad=1)
        plt.locator_params(axis='x',nbins=3)
        plt.savefig('figure/taskdimension_ranking'+self.config['save_addon']+'.pdf', transparent=True)
        plt.show()

    def plot_sumevs(self, plot_scaled=False, data_names_plot=None):
        '''
        Plot proportional of total explained variance
        :param dims: Optional. If provided, scale the x-axis accordingly
        :return:
        '''
        # colors = [plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(set(self.data_singlenames)))]

        if data_names_plot is None:
            data_names_plot = self.data_names

        x0 = np.arange(len(self.sumevs[self.data_names[0]]))
        fig = plt.figure(figsize=(1.5,1.2))
        ax = fig.add_axes([0.3,0.3,0.6,0.6])
        j = 0
        for data_name in data_names_plot:
            if plot_scaled:
                dim = self.dims[data_name]
            else:
                dim = 1

            if self.data_type == 'rule':
                if isinstance(data_name,int):
                    label = rule_name[data_name]
                    ax.plot(x0/dim, self.sumevs[data_name], color=colors[j],label=label)
                    j += 1
                else:
                    label = rule_name[data_name[0]] + '+' + rule_name[data_name[1]]
                    if len(data_names_plot)<5:
                        ax.plot(x0/dim, self.sumevs[data_name], color=colors[j], label=label)
                        j += 1
                    else:
                        ax.plot(x0/dim, self.sumevs[data_name], color='gray',alpha=0.1)

                title = 'Rule'

            elif self.data_type == 'epoch':
                # For epoch data_type, data_name is either a tuple, or a tuple of tuples
                if isinstance(data_name[0],int): # If tuple, then (rule, epoch)
                    title = rule_name[data_name[0]]
                    label = data_name[1] # the epoch name
                    ax.plot(x0/dim, self.sumevs[data_name], color=colors[j],label=label)
                    j += 1
                else:
                    label = data_name[0][1] + '+' + data_name[1][1]
                    ax.plot(x0/dim, self.sumevs[data_name], color='gray',alpha=0.1)



        ax.set_xlim([-1,15])
        ax.set_ylim([-0.05, 1.05])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if len(data_names_plot)<5:
            lg = plt.legend(title=title,ncol=1,bbox_to_anchor=(1.2,0.8),
                            fontsize=7,labelspacing=0.3,frameon=False)
            plt.setp(lg.get_title(),fontsize=7)

        if plot_scaled:
            xlabel = 'Dims (scaled)'
            plot_name = self.dims_method
        else:
            xlabel = 'PCs'
            plot_name = ''
        plt.xlabel(xlabel, fontsize=7, labelpad=1)
        plt.ylabel('Prop. of Total E.V.', fontsize=7)
        plt.title(plot_name, fontsize=7)
        plt.locator_params(nbins=3)
        plt.savefig('figure/sumexpvar_'+plot_name+self.config['save_addon']+'.pdf', transparent=True)
        plt.show()

    def compare_methods(self):
        # Comparing two methods to get the dimensions
        dim0s = self.fit_exponential()
        dim1s = self.fit_scaling()

        fig = plt.figure(figsize=(2,2))
        ax = fig.add_axes([0.2,0.2,0.7,0.7])
        ax.scatter(dim0s.values(), dim1s.values(), facecolor='gray')
        ax.plot([0.5,4],[0.5,4],'black')
        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.xlabel('Factor through fitting exponential',fontsize=7)
        plt.ylabel('Factor through direct scaling',fontsize=7)
        plt.savefig('figure/comparedims'+self.config['save_addon']+'.pdf')
        plt.show()

    def get_dim_ratios(self):
        dim_ratios = np.zeros((len(self.data_singlenames),
                               len(self.data_singlenames)))
        for i, sn_i in enumerate(self.data_singlenames):
            for j, sn_j in enumerate(self.data_singlenames):
                if i == j:
                    dim_ratios[i,j] = 0.5 # If the same
                else:
                    data_name = (sn_i,sn_j) if (sn_i,sn_j) in self.data_names else (sn_j,sn_i)
                    dim_ratios[i,j] = self.dims[data_name]/(self.dims[sn_j]+self.dims[sn_i])

        return dim_ratios

    def sort_areas(self, dim_ratios):
        W1 = dim_ratios*2-1 # lambda ratios should be between 0.5 and 1
        W1 = (W1>0)*W1
        # convert the redundant n*n square matrix form into a condensed nC2 array
        distArray = ssd.squareform(W1)
        Z = sch.linkage(distArray, method='average')
        ind = sch.leaves_list(Z)
        return ind

    def plot_dim_ratios(self, load_previous=False, sort_ind=True):
        '''
        Get and Plot Ratios of Dimensions
        Plot lambda_ratio = Dim 1+2/ (Dim 1 + Dim 2). Theoretically should range between 0.5 and 1
        lambda_ratio=0.5 when tasks 1 and 2 are the same
        lambda_ratio=1 when task 1 and 2 occupy orthongonal space
        :param dims: Relative dimensions of each task, should be dictionary
        :return:
        '''
        dim_ratios = self.get_dim_ratios()
        if not sort_ind:
            ind = range(dim_ratios.shape[0])
        elif not load_previous:
            ind = self.sort_areas(dim_ratios)
            fname = 'data/dim_ratios_sortind_'+self.config['save_addon']+'.pkl'
            with open(fname,'wb') as f:
                pickle.dump(ind, f)
        else:
            raise ValueError('No longer supported options')
            fname = 'data/dim_ratios_sortind_rule'+'_test120_16'+'.pkl'
            with open(fname,'rb') as f:
                ind = pickle.load(f)

        if self.data_type == 'rule':
            figsize = (3.5,3.5)
            rect = [0.25,0.25,0.6,0.6]
            tick_names = [rule_name[self.rules[i]] for i in ind]
        elif self.data_type == 'epoch':
            figsize = (5,5)
            rect = [0.2,0.2,0.7,0.7]
            tick_names = [rule_name[self.epochs[i][0]] +' '+ self.epochs[i][1] for i in ind]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
        im = ax.imshow(2-2*dim_ratios[ind,:][:,ind], aspect='equal', cmap='hot',
                       vmin=0,vmax=1.0,interpolation='nearest',origin='lower')

        if len(tick_names)<20:
            tick_fontsize = 7
        elif len(tick_names)<30:
            tick_fontsize = 6
        else:
            tick_fontsize = 5

        plt.xticks(range(len(tick_names)), tick_names,
                   rotation=90, ha='right', fontsize=tick_fontsize)
        plt.yticks(range(len(tick_names)), tick_names,
                   rotation=0, va='center', fontsize=tick_fontsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, ticks=[0,0.5,1])
        cb.set_label('Similarity', fontsize=7, labelpad=3)
        plt.tick_params(axis='both', which='major', labelsize=7)
        plt.savefig('figure/taskdimensionratio'+self.config['save_addon']+'.pdf',transparent=True)
        plt.show()

    def plot_PCA(self, save=True, i_pcs=(0,1)):
        data = np.array([self.h_tbmeans[rule] for rule in self.rules])
        pca = PCA(n_components=5,whiten=False)
        pca.fit(data)

        data1 = pca.transform(data)

        i_pc1, i_pc2 = i_pcs
        x_data = data1[:,i_pc1]
        y_data = data1[:,i_pc2]

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_axes([0.2,0.2,0.7,0.7])
        ax.scatter(x_data,y_data,s=6,c=sns.xkcd_rgb["pale red"], edgecolor='none')
        plt.xlabel('PC' + str(i_pc1+1),fontsize=7)
        plt.ylabel('PC' + str(i_pc2+1),fontsize=7)

        txts = [rule_name[rule] for rule in self.rules]
        from adjustText import adjust_text
        texts = []
        for x, y, s in zip(x_data, y_data, txts):
            texts.append(plt.text(x, y, s, size=7))
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

        ax.tick_params(axis='both', which='major', labelsize=7)
        plt.locator_params(nbins=3)
        titletxt = 'Explained Variance {:0.2f} %'.format(pca.explained_variance_ratio_[[i_pc1,i_pc2]].sum()*100)
        plt.title(titletxt,fontsize=7)
        if save:
            plt.savefig('figure/task_pca_dim'+str(i_pc1)+str(i_pc2)+self.config['save_addon']+'.pdf')
        plt.show()

    def plot_hunits_tasks(self):
        fig = plt.figure(figsize=(3.5,2))
        ax = fig.add_axes([0.15,0.2,0.5,0.7])
        ax.set_color_cycle([plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(set(self.rules)))])
        #colors = [240,163,255],[0,117,220],[153,63,0],[76,0,92],[25,25,25],[0,92,49],[43,206,72],[255,204,153],[128,128,128],[148,255,181],[143,124,0],[157,204,0],[194,0,136],[0,51,128],[255,164,5],[255,168,187],[66,102,0],[255,0,16],[94,241,242],[0,153,143],[224,255,102],[116,10,255],[153,0,0],[255,255,128],[255,255,0],[255,80,5]
        for rule in self.rules:
            ax.plot(self.h_tbmeans[rule],label=rule_name[rule])
        plt.xlim([-1, self.h_dim])
        plt.ylim([-0.1,1.1])

        ax.tick_params(axis='both', which='major', labelsize=7)
        lg = plt.legend(title='Rule',ncol=1,bbox_to_anchor=(1.7,1.1),
                        fontsize=7,labelspacing=0.3)
        plt.setp(lg.get_title(),fontsize=7)
        plt.xlabel('Recurrent Units',fontsize=7)
        plt.ylabel('Activity',fontsize=7)
        plt.savefig('figure/h_bytasks'+self.config['save_addon']+'.pdf')
        plt.show()

    def plot_hunits_tasks_grouping(self, group_by_feature):
        rules = A.h_tbmeans.keys()
        h_tbmeans = np.array(A.h_tbmeans.values())
        ind1 =[i for i, rule in enumerate(rules) if group_by_feature in rule_features[rule]]
        ind0 =[i for i, rule in enumerate(rules) if group_by_feature not in rule_features[rule]]
        h_tbmeans0 = h_tbmeans[ind0,:]
        h_tbmeans1 = h_tbmeans[ind1,:]
        idxsort = np.argsort(np.min(h_tbmeans1,axis=0)-np.max(h_tbmeans0,axis=0))

        fig = plt.figure(figsize=(3.5,2))
        ax = fig.add_axes([0.15,0.2,0.5,0.7])
        #ax.set_color_cycle([plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, len(set(rules)))])

        for rule in self.rules:
            if group_by_feature in rule_features[rule]:
                color = 'blue'
            else:
                color = 'red'
            ax.plot(self.h_tbmeans[rule][idxsort][-10:],'o-',label=rule_name[rule], color=color, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=7)
        lg = plt.legend(title='Rule',ncol=1,bbox_to_anchor=(1.7,1.1),
                        fontsize=7,labelspacing=0.3)
        plt.setp(lg.get_title(),fontsize=7)
        plt.xlabel('Recurrent Units',fontsize=7)
        plt.ylabel('Activity',fontsize=7)
        plt.savefig('figure/h_bytasks_sortby'+feature_names[group_by_feature]+self.config['save_addon']+'.pdf')
        plt.show()

    def find_remap_units(self):
        # Get all units that satisfy the following conditions:
        # The lowest activity of the Remap tasks is higher than the highest activity of the other tasks
        # The mean activity of other tasks is lower than 0.05 (kind of arbitrary, need to justify)

        group_by_feature = Remap

        rules = self.h_tbmeans.keys()
        h_tbmeans = np.array(self.h_tbmeans.values())
        h_tbmeans = (h_tbmeans>0)*h_tbmeans + 1e-7
        ind1 =[i for i, rule in enumerate(rules) if group_by_feature in rule_features[rule]]
        ind0 =[i for i, rule in enumerate(rules) if group_by_feature not in rule_features[rule]]
        h_tbmeans0 = h_tbmeans[ind0,:]
        h_tbmeans1 = h_tbmeans[ind1,:]

        inds = np.where((np.min(h_tbmeans1,axis=0)>np.max(h_tbmeans0,axis=0))*
                        (np.mean(h_tbmeans0,axis=0)<0.05)*
                        (np.mean(h_tbmeans1,axis=0)>0.05))[0]

        return inds

    def plot_unit_connection(self, inds):
        ga = GeneralAnalysis(save_addon=self.save_addon_original)

        for ind in inds:
            plt.figure(figsize=(2,1.5))
            plt.plot(ga.Win[ind,1:1+N_RING])
            plt.plot(ga.Win[ind,1+N_RING:1+2*N_RING])
            plt.plot(ga.Wout[:,ind][1:],'black')
            plt.show()
        '''
        for ind in inds:
            plt.figure(figsize=(4,3))
            plt.plot(ga.Win[ind,1+2*N_RING:],'o-')
            plt.xticks(range(20),[rule_name[i] for i in range(20)],rotation=90)
            plt.show()
        '''
        for ind in inds:
            plt.figure(figsize=(1.5,1.5))
            plt.plot(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING],'o')
            plt.show()


def analyze_remap_units():
    ###################### Plot single unit activity ##########################
    HDIM = 202
    N_RING = 16
    save_addon = 'chanceabbott_'+str(HDIM)+'_'+str(N_RING)
    A = SumStatAnalysis('rule', save_addon)
    inds_remap = A.find_remap_units()

    # A.plot_unit_connection(inds)
    ga = GeneralAnalysis(save_addon=A.save_addon_original)
    success = False
    for ind in inds_remap:
        res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING])
        if res.rvalue<-0.9 and res.pvalue<0.01:
            success = True
            break

    if success:
        rules = [INHGO, DELAYREMAP, INHREMAP]
        ga.run_test(rules=rules)
        ga.plot_singleneuron_intime(neuron_id=ind, rules=rules, save_fig=True)
    else:
        raise ValueError('Did not success to find a typical remap unit in this network')
    ########### Plot Causal Manipulation Results  #############################

    perfs = ga.get_performances()

    # inh_id = ind
    inh_id = inds_remap
    ga_inh = GeneralAnalysis(save_addon=save_addon, inh_id=inh_id, inh_output=True)
    perfs_inh = ga_inh.get_performances()

    fig = plt.figure(figsize=(2.5,2))
    ax = fig.add_axes([0.2,0.5,0.75,0.4])
    ax.plot(perfs, 'o-', markersize=3, label='intact', color=sns.xkcd_palette(['black'])[0])
    ax.plot(perfs_inh, 'o-', markersize=3, label='Inh. unit \n'+str(inh_id), color=sns.xkcd_palette(['red'])[0])
    plt.xticks(range(len(ga.rules)), [rule_name[rule] for rule in ga.rules], rotation=90)
    plt.ylabel('performance', fontsize=7)
    plt.ylim(bottom=0.0, top=1.05)
    plt.xlim([-0.5, len(perfs)-0.5])
    ax.tick_params(axis='both', which='major', labelsize=7)
    leg = plt.legend(fontsize=6,frameon=False, loc=2, numpoints=1,
                     bbox_to_anchor=(0,0.5), labelspacing=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.locator_params(axis='y', nbins=4)
    plt.savefig('figure/performance_inh'+str(inh_id)+A.config['save_addon']+'.pdf', transparent=True)
    plt.show()

    ###################### Plot single unit connection ##########################
    # Connection with ring input and output
    w_in = ga.Win[ind,1:1+N_RING]
    w_out = ga.Wout[:,ind][1:]
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.3,0.3,0.6,0.5])
    ax.plot(w_in, color=sns.xkcd_palette(['green'])[0], label='from input')
    ax.plot(w_out, color=sns.xkcd_palette(['blue'])[0], label='to output')
    plt.ylabel('conn. weight', fontsize=7, labelpad=1)
    plt.xlabel('ring', fontsize=7)
    plt.xticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'])
    plt.title('Unit {:d} '.format(ind), fontsize=7)

    wmax = max((w_in.max(),w_out.max()))
    wmin = min((w_in.min(),w_out.min()))
    plt.ylim([wmin-0.1*(wmax-wmin),wmax+0.7*(wmax-wmin)])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.locator_params(axis='y',nbins=5)
    lg = plt.legend(ncol=1,bbox_to_anchor=(1,1.1),frameon=False,
                    fontsize=7,labelspacing=0.3,loc=1)
    plt.savefig('figure/sample_remap_unit_connectivity.pdf', transparent=True)
    plt.show()


    ########### Plot histogram of correlation coefficient #####################
    slopes = list()
    rvalues = list()
    pvalues = list()
    remap_units = list()
    HDIMs = range(190,207)
    for j, HDIM in enumerate(HDIMs):
        N_RING = 16
        save_addon = 'chanceabbott_'+str(HDIM)+'_'+str(N_RING)
        A = SumStatAnalysis('rule', save_addon)
        inds_remap = A.find_remap_units()

        # A.plot_unit_connection(inds)
        ga = GeneralAnalysis(save_addon=A.save_addon_original)
        for ind in range(HDIM):
            res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+N_RING])
            slopes.append(res.slope)
            rvalues.append(res.rvalue)
            pvalues.append(res.pvalue)
            remap_units.append(ind in inds_remap)

        if j == 0:
            conn_rule_to_all = ga.Win[:, 1+2*N_RING:] # connection from rule inputs to all units
            conn_rule_to_remap = ga.Win[inds_remap, 1+2*N_RING:] # connection from rule inputs to remap units
        else:
            conn_rule_to_all = np.concatenate((conn_rule_to_all, ga.Win[:, 1+2*N_RING:]))
            conn_rule_to_remap = np.concatenate((conn_rule_to_remap, ga.Win[inds_remap, 1+2*N_RING:]))


    slopes, rvalues, pvalues, remap_units = np.array(slopes), np.array(rvalues), np.array(pvalues), np.array(remap_units)

    # plot_value, plot_range = slopes, (-4,4)
    plot_value, plot_range = rvalues, (-1,1)
    thres = 0.01 ##TODO: Find out how to set this threshold
    for i in [0,1]:
        if i == 0:
            units = remap_units
            title = 'Remap'
        else:
            units = (1-remap_units).astype(bool)
            title = 'Non-remap'

        fig = plt.figure(figsize=(1.5,1.2))
        ax = fig.add_axes([0.3,0.3,0.6,0.5])
        hist, bins_edge = np.histogram(plot_value[units], range=plot_range, bins=30)
        ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=sns.xkcd_palette(['navy blue'])[0], edgecolor='none')
        hist, bins_edge = np.histogram(plot_value[units*(pvalues<thres)], range=plot_range, bins=30)
        ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
        plt.xlabel('corr. coeff.', fontsize=7, labelpad=1)
        plt.ylabel('counts', fontsize=7)
        plt.locator_params(nbins=3)
        plt.title(title+' units ({:d} nets)'.format(len(HDIMs)) , fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig('figure/'+title+'_unit_cc.pdf', transparent=True)
        plt.show()

    ########### Plot connections averaged across networks #####################
    fig = plt.figure(figsize=(2.5,2))
    ax = fig.add_axes([0.2,0.5,0.75,0.4])
    ax.plot(conn_rule_to_all.mean(axis=0)[ga.rules], 'o-', markersize=3, label='all', color=sns.xkcd_palette(['black'])[0])
    ax.plot(conn_rule_to_remap.mean(axis=0)[ga.rules], 'o-', markersize=3, label='remap', color=sns.xkcd_palette(['red'])[0])

    plt.xticks(range(len(ga.rules)), [rule_name[rule] for rule in ga.rules], rotation=90)
    plt.ylabel('Mean conn. from rule', fontsize=7)
    plt.title('Average of {:d} networks'.format(len(HDIMs)), fontsize=7)
    #plt.ylim(bottom=0.0, top=1.05)
    plt.xlim([-0.5, len(ga.rules)-0.5])
    ax.tick_params(axis='both', which='major', labelsize=7)
    leg = plt.legend(title='to units',fontsize=7,frameon=False, loc=2, numpoints=1,
                     bbox_to_anchor=(0.3,0.6), labelspacing=0.3)
    plt.setp(leg.get_title(),fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.locator_params(axis='y', nbins=4)
    plt.savefig('figure/conn_ruleinput_to_remap.pdf', transparent=True)
    plt.show()

# HDIM, save_type = 200, 'latest'
# N_RING = 16
# save_addon = save_type+'_'+str(HDIM)+'_'+str(N_RING)
# A = SumStatAnalysis(data_type='rule', save_addon=save_addon)
# A.plot_ranking()
# A.plot_sumevs(plot_scaled=True)
# A.plot_sumevs(plot_scaled=False, data_names_plot=[GO, REMAP, (GO, REMAP)])
# A.plot_sumevs(plot_scaled=False)
# A.plot_hunits_tasks()
# # A.plot_hunits_tasks_grouping(group_by_feature=Remap)
# A.plot_PCA(i_pcs=(0,1))
# A.plot_PCA(i_pcs=(1,2))
# A.plot_dim_ratios(load_previous=False)

# A = SumStatAnalysis('epoch', save_addon)
# A.plot_dim_ratios(sort_ind=True)


def compute_varstim(HDIMs, rules, save_type):
    res = {k:np.array([]) for k in ['h_var_stim', 'h_var_time', 'h_mean', 'rule']}
    i_net = 0
    for HDIM in HDIMs:
        N_RING = 16
        save_addon = save_type+'_'+str(HDIM)+'_'+str(N_RING)
        fname = 'data/config'+save_addon+'.pkl'
        if not os.path.isfile(fname):
            continue
        ga = GeneralAnalysis(save_addon=save_addon)

        for rule in rules:
            task = generate_onebatch(rule=rule, config=ga.config,
                                     mode='test')
            x_sample = task.x
            t_start  = 200
            h_sample = ga.f_h(x_sample)[t_start:,:,:]

            h_mean     = h_sample.mean(axis=0).mean(axis=0)
            h_var_time = h_sample.mean(axis=1).var(axis=0)
            h_var_stim = h_sample.var(axis=1).mean(axis=0)

            res['h_var_stim'] = np.concatenate((res['h_var_stim'],h_var_stim))
            res['h_var_time'] = np.concatenate((res['h_var_time'],h_var_time))
            res['h_mean'] = np.concatenate((res['h_mean'],h_mean))
            res['rule'] = np.concatenate((res['rule'], np.ones(len(h_mean))*rule))

        i_net += 1

    res['h_var_stim_prop'] = res['h_var_stim']/(res['h_var_stim']+res['h_var_time'])
    return res, i_net


def plot_var_stim_prop_hist(HDIMs, rules, save_type):
    assert(len(rules)==1)
    res, i_net = compute_varstim(HDIMs, rules, save_type)

    color = sns.xkcd_palette(['cerulean'])[0]
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.35,0.3,0.55,0.45])
    hist, bins_edge = np.histogram(res['h_var_stim_prop'][res['h_mean']>0.1], bins=30)
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=color, edgecolor='none')
    plt.xlabel('Stim. var. prop. '+rule_name[rules[0]], fontsize=7)
    plt.ylabel('unit counts', fontsize=7)
    plt.xticks([0,0.5,1])
    plt.locator_params(axis='y', nbins=2)
    if save_type == 'dmc':
        plt.title('pooling {:d} nets \ntrained for DMC'.format(i_net), fontsize=7)
    else:
        plt.title('pooling {:d} nets'.format(i_net), fontsize=7)
    plt.xlim([-0.1,1.1])
    plt.ylim([0, hist.max()*1.2])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('figure/var_stim_prop_hist_'+save_type+'.pdf', transparent=True)
    plt.show()

#plot_var_stim_prop_hist(HDIMs=range(150,600), rules=[DMCGO], save_type='latest')
#plot_var_stim_prop_hist(HDIMs=range(150,600), rules=[DMCGO], save_type='dmc')

def plot_var_stim_prop_rulecomparison():
    rules = [DMCGO, DELAYGO]
    res, i_net = compute_varstim(HDIMs=[200], rules=rules, save_type='latest')
    
    color = sns.xkcd_palette(['cerulean'])[0]
    fig = plt.figure(figsize=(1.2,1.2))
    ax = fig.add_axes([0.35,0.3,0.55,0.55])
    ax.plot(res['h_var_stim_prop'][res['rule']==rules[0]],
             res['h_var_stim_prop'][res['rule']==rules[1]],
            'o', markerfacecolor=color, markeredgecolor=color,markersize=1.0,alpha=0.5)
    plt.xlabel(rule_name[rules[0]], fontsize=7)
    plt.ylabel(rule_name[rules[1]], fontsize=7)
    plt.title('Stim. var. prop.', fontsize=7)
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('figure/var_stim_prop_rulecomparison.pdf', transparent=True)
    plt.show()


def temp():
    h_varprop_stim_all = np.array([])
    #HDIMs, save_type = range(190,210), 'latest'
    HDIMs, save_type = [200], 'latest'
    #HDIMs, save_type = range(200,205), 'dmc'
    #HDIMs, save_type = [200], 'dmc'
    for HDIM in HDIMs:
        N_RING = 16
        save_addon = save_type+'_'+str(HDIM)+'_'+str(N_RING)
        ga = GeneralAnalysis(save_addon=save_addon)
        rule = DMCGO
        task = generate_onebatch(rule=rule, config=ga.config,
                                 mode='test')
        x_sample = task.x
        t_start  = 200
        h_sample = ga.f_h(x_sample)[t_start:,:,:]
    
        h_mean     = h_sample.mean(axis=0).mean(axis=0)
    
        h_var_time = h_sample.var(axis=0).mean(axis=0)
        h_var_stim = h_sample.mean(axis=0).var(axis=0)
        h_var_all  = h_sample.reshape((-1,HDIM)).var(axis=0)
    
        h_varprop_time = h_var_time/h_var_all
        h_varprop_stim = h_var_stim/h_var_all
        h_varprop_stim_all = np.concatenate((h_varprop_stim_all,h_varprop_stim))
    
    _ = plt.hist(h_varprop_stim_all, bins=30)
    plt.xlabel('Stim variance proportion', fontsize=7)
    plt.show()
    
    h_units = h_mean>0.1
    ind0 = np.arange(len(h_varprop_stim))
    h_varprop_stim = h_varprop_stim[h_units]
    ind = ind0[h_units]
    ind_sort = np.argsort(h_varprop_stim)
    ind1 = ind[ind_sort[-10:]] # value high
    ind2  = ind[ind_sort[:10]] # value low
    ga.plot_singleneuron_DMCGO(ind1)
    ga1 = GeneralAnalysis(save_addon=save_addon, inh_id=ind1, inh_output=True)
    ga2 = GeneralAnalysis(save_addon=save_addon, inh_id=ind2, inh_output=True)
    
    # ga2.sample_plot(GO)
    
    costs, performances = ga.get_costs()
    costs1, performances1 = ga1.get_costs()
    costs2, performances2 = ga2.get_costs()
    
    # value_type = 'performance'
    for value_type in ['cost', 'performance']:
        if value_type == 'cost':
            vals, vals1, vals2 = np.log(costs), np.log(costs1), np.log(costs2)
        elif value_type == 'performance':
            vals, vals1, vals2 = performances, performances1, performances2
        fig = plt.figure(figsize=(2.5,2.0))
        ax = fig.add_axes([0.2,0.5,0.75,0.4])
        ax.plot(vals, 'o-', markersize=3, label='intact', color=sns.xkcd_palette(['black'])[0])
        ax.plot(vals1, 'o-', markersize=3, label='Inh. unit dprime high', color=sns.xkcd_palette(['red'])[0])
        ax.plot(vals2,  'o-', markersize=3, label='Inh. unit dprime low', color=sns.xkcd_palette(['blue'])[0])
        plt.xticks(range(len(ga.rules)), [rule_name[r] for r in ga.rules], rotation=90)
        plt.ylabel(value_type, fontsize=7)
        #plt.ylim(bottom=0.0, top=1.05)
        plt.xlim([-0.5, len(vals)-0.5])
        ax.tick_params(axis='both', which='major', labelsize=7)
        leg = plt.legend(fontsize=6,frameon=False, loc=2, numpoints=1,
                         bbox_to_anchor=(0,0.5), labelspacing=0.3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.locator_params(axis='y', nbins=4)
        plt.savefig('figure/'+value_type+'_inh_dprime'+ga.config['save_addon']+'.pdf', transparent=True)
        plt.show()


save_type, HDIM, N_RING = 'latest', 200, 16
save_addon = save_type+'_'+str(HDIM)+'_'+str(N_RING)
ga = GeneralAnalysis(save_addon=save_addon)
ga.plot_trainingprogress()
#costs, performances = ga.get_costs(rules=[DMCGO])