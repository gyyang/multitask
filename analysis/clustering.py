"""
Clustering analysis
Analyze how units are involved in various tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf

from task import rule_name
from network import Model
import tools

# Colors used for clusters
kelly_colors = \
[np.array([ 0.94901961,  0.95294118,  0.95686275]),
 np.array([ 0.13333333,  0.13333333,  0.13333333]),
 np.array([ 0.95294118,  0.76470588,  0.        ]),
 np.array([ 0.52941176,  0.3372549 ,  0.57254902]),
 np.array([ 0.95294118,  0.51764706,  0.        ]),
 np.array([ 0.63137255,  0.79215686,  0.94509804]),
 np.array([ 0.74509804,  0.        ,  0.19607843]),
 np.array([ 0.76078431,  0.69803922,  0.50196078]),
 np.array([ 0.51764706,  0.51764706,  0.50980392]),
 np.array([ 0.        ,  0.53333333,  0.3372549 ]),
 np.array([ 0.90196078,  0.56078431,  0.6745098 ]),
 np.array([ 0.        ,  0.40392157,  0.64705882]),
 np.array([ 0.97647059,  0.57647059,  0.4745098 ]),
 np.array([ 0.37647059,  0.30588235,  0.59215686]),
 np.array([ 0.96470588,  0.65098039,  0.        ]),
 np.array([ 0.70196078,  0.26666667,  0.42352941]),
 np.array([ 0.8627451 ,  0.82745098,  0.        ]),
 np.array([ 0.53333333,  0.17647059,  0.09019608]),
 np.array([ 0.55294118,  0.71372549,  0.        ]),
 np.array([ 0.39607843,  0.27058824,  0.13333333]),
 np.array([ 0.88627451,  0.34509804,  0.13333333]),
 np.array([ 0.16862745,  0.23921569,  0.14901961])]


save = True


class Analysis(object):
    def __init__(self, model_dir, data_type, normalization_method='max'):
        hp = tools.load_hp(model_dir)

        # If not computed, use variance.py
        fname = os.path.join(model_dir, 'variance_' + data_type + '.pkl')
        res = tools.load_pickle(fname)
        h_var_all_ = res['h_var_all']
        self.keys  = res['keys']

        # First only get active units. Total variance across tasks larger than 1e-3
        # ind_active = np.where(h_var_all_.sum(axis=1) > 1e-2)[0]
        ind_active = np.where(h_var_all_.sum(axis=1) > 1e-3)[0]
        h_var_all  = h_var_all_[ind_active, :]

        # Normalize by the total variance across tasks
        if normalization_method == 'sum':
            h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T
        elif normalization_method == 'max':
            h_normvar_all = (h_var_all.T/np.max(h_var_all, axis=1)).T
        elif normalization_method == 'none':
            h_normvar_all = h_var_all
        else:
            raise NotImplementedError()

        ################################## Clustering ################################
        from sklearn import metrics
        X = h_normvar_all

        # Clustering
        from sklearn.cluster import AgglomerativeClustering, KMeans

        # Choose number of clusters that maximize silhouette score
        n_clusters = range(2, 30)
        scores = list()
        labels_list = list()
        for n_cluster in n_clusters:
            # clustering = AgglomerativeClustering(n_cluster, affinity='cosine', linkage='average')
            clustering = KMeans(n_cluster, algorithm='full', n_init=20, random_state=0)
            clustering.fit(X) # n_samples, n_features = n_units, n_rules/n_epochs
            labels = clustering.labels_ # cluster labels

            score = metrics.silhouette_score(X, labels)

            scores.append(score)
            labels_list.append(labels)

        scores = np.array(scores)

        # Heuristic elbow method
        # Choose the number of cluster when Silhouette score first falls
        # Choose the number of cluster when Silhouette score is maximum
        if data_type == 'rule':
            #i = np.where((scores[1:]-scores[:-1])<0)[0][0]
            i = np.argmax(scores)
        else:
            # The more rigorous method doesn't work well in this case
            i = n_clusters.index(10)

        labels = labels_list[i]
        n_cluster = n_clusters[i]
        print('Choosing {:d} clusters'.format(n_cluster))

        # Sort clusters by its task preference (important for consistency across nets)
        if data_type == 'rule':
            label_prefs = [np.argmax(h_normvar_all[labels==l].sum(axis=0)) for l in set(labels)]
        elif data_type == 'epoch':
            ## TODO: this may no longer work!
            label_prefs = [self.keys[np.argmax(h_normvar_all[labels==l].sum(axis=0))][0] for l in set(labels)]

        ind_label_sort = np.argsort(label_prefs)
        label_prefs = np.array(label_prefs)[ind_label_sort]
        # Relabel
        labels2 = np.zeros_like(labels)
        for i, ind in enumerate(ind_label_sort):
            labels2[labels==ind] = i
        labels = labels2

        # # Sort data by labels and by input connectivity
        # model = Model(save_name)
        # hp = model.hp
        # with tf.Session() as sess:
        #     model.restore(sess)
        #     var_list = sess.run(model.var_list)
        #
        # # Get connectivity
        # w_out = var_list[0].T
        # b_out = var_list[1]
        # w_in  = var_list[2][:n_input, :].T
        # w_rec = var_list[2][n_input:, :].T
        # b_rec = var_list[3]
        #
        # # nx, nh, ny = hp['shape']
        # nr = hp['n_eachring']
        #
        # sort_by = 'w_in'
        # if sort_by == 'w_in':
        #     w_in_mod1 = w_in[ind_active, :][:, 1:nr+1]
        #     w_in_mod2 = w_in[ind_active, :][:, nr+1:2*nr+1]
        #     w_in_modboth = w_in_mod1 + w_in_mod2
        #     w_prefs = np.argmax(w_in_modboth, axis=1)
        # elif sort_by == 'w_out':
        #     w_prefs = np.argmax(w_out[1:, ind_active], axis=0)
        #
        # ind_sort        = np.lexsort((w_prefs, labels)) # sort by labels then by prefs

        ind_sort = np.argsort(labels)

        labels          = labels[ind_sort]
        self.h_normvar_all   = h_normvar_all[ind_sort, :]
        self.ind_active      = ind_active[ind_sort]

        self.n_clusters = n_clusters
        self.scores = scores
        self.n_cluster = n_cluster

        self.h_var_all = h_var_all
        self.normalization_method = normalization_method
        self.labels = labels
        self.unique_labels = np.unique(labels)

        self.model_dir = model_dir
        self.hp = hp
        self.data_type = data_type
        self.rules = hp['rules']

    def plot_cluster_score(self, save_name=None):
        """Plot the score by the number of clusters."""
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.3, 0.3, 0.55, 0.55])
        ax.plot(self.n_clusters, self.scores, 'o-', ms=3)
        ax.set_xlabel('Number of clusters', fontsize=7)
        ax.set_ylabel('Silhouette score', fontsize=7)
        ax.set_title('Chosen number of clusters: {:d}'.format(self.n_cluster),
                     fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_ylim([0, 0.32])
        if save:
            fig_name = 'cluster_score'
            if save_name is None:
                save_name = self.hp['activation']
            fig_name = fig_name + save_name
            plt.savefig('figure/'+fig_name+'.pdf', transparent=True)
        plt.show()

    def plot_variance(self, save_name=None):
        labels = self.labels
        ######################### Plotting Variance ###################################
        # Plot Normalized Variance
        if self.data_type == 'rule':
            figsize = (3.5,2.5)
            rect = [0.25, 0.2, 0.6, 0.7]
            rect_color = [0.25, 0.15, 0.6, 0.05]
            rect_cb = [0.87, 0.2, 0.03, 0.7]
            tick_names = [rule_name[r] for r in self.rules]
            fs = 6
            labelpad = 13
        elif self.data_type == 'epoch':
            figsize = (3.5,4.5)
            rect = [0.25, 0.1, 0.6, 0.85]
            rect_color = [0.25, 0.05, 0.6, 0.05]
            rect_cb = [0.87, 0.1, 0.03, 0.85]
            tick_names = [rule_name[key[0]]+' '+key[1] for key in self.keys]
            fs = 5
            labelpad = 20
        else:
            raise ValueError

        h_plot  = self.h_normvar_all.T
        vmin, vmax = 0, 1
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = ax.imshow(h_plot, cmap='hot',
                       aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        plt.yticks(range(len(tick_names)), tick_names,
                   rotation=0, va='center', fontsize=fs)
        plt.xticks([])
        plt.title('Units', fontsize=7, y=0.97)
        plt.xlabel('Clusters', fontsize=7, labelpad=labelpad)
        ax.tick_params('both', length=0)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        if self.normalization_method == 'sum':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'max':
            clabel = 'Normalized Task Variance'
        elif self.normalization_method == 'none':
            clabel = 'Variance'

        cb.set_label(clabel, fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)
        

        # Plot color bars indicating clustering
        if True:
            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
                ax.plot(ind_l, [0,0], linewidth=4, solid_capstyle='butt',
                        color=kelly_colors[il+1])
                ax.text(np.mean(ind_l), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=kelly_colors[il+1])
            ax.set_xlim([0, len(labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

        if save:
            fig_name = ('feature_map_by' + self.data_type +
                        '_norm' + self.normalization_method)
            if save_name is not None:
                fig_name = fig_name + save_name
            plt.savefig('figure/'+fig_name+'.pdf', transparent=True)
        plt.show()

    def plot_similarity_matrix(self):
        labels = self.labels
        ######################### Plotting Similarity Matrix ##########################

        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(self.h_normvar_all) # TODO: check
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
        im = ax.imshow(similarity, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        ax.axis('off')

        ax = fig.add_axes([0.87, 0.25, 0.03, 0.6])
        cb = plt.colorbar(im, cax=ax, ticks=[0,1])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Similarity', fontsize=7, labelpad=0)
        plt.tick_params(axis='both', which='major', labelsize=7)

        ax1 = fig.add_axes([0.25, 0.85, 0.6, 0.05])
        ax2 = fig.add_axes([0.2, 0.25, 0.05, 0.6])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
            ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        if save:
            plt.savefig('figure/feature_similarity_by'+self.data_type+'.pdf', transparent=True)
        plt.show()

    def plot_2Dvisualization(self, method='TSNE'):
        labels = self.labels
        ######################## Plotting 2-D visualization of variance map ###########
        from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
        from sklearn.decomposition import PCA

        # model = LocallyLinearEmbedding()
        if method == 'PCA':
            model = PCA(n_components=2, whiten=False)
        elif method == 'MDS':
            model = MDS(metric=True, n_components=2, n_init=10, max_iter=1000)
        elif method == 'tSNE':
            model = TSNE(n_components=2, random_state=0, init='pca',
                         verbose=1, method='exact',
                         learning_rate=100, perplexity=30)
        else:
            raise NotImplementedError

        Y = model.fit_transform(self.h_normvar_all)

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0]
            ax.scatter(Y[ind_l, 0], Y[ind_l, 1], color=kelly_colors[il+1], s=10)
        ax.axis('off')
        plt.title(method, fontsize=7)
        figname = 'figure/taskvar_visual_by'+method+self.data_type+'.pdf'
        if save:
            plt.savefig(figname, transparent=True)
        plt.show()

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.scatter(Y[:,0], Y[:,1], color='black')
        ax.axis('off')

    def plot_example_unit(self):
        ######################## Plotting Variance for example unit ###################
        if self.data_type == 'rule':
            tick_names = [rule_name[r] for r in self.rules]

            ind = 2 # example unit
            fig = plt.figure(figsize=(1.2,1.0))
            ax = fig.add_axes([0.4,0.4,0.5,0.45])
            ax.plot(range(self.h_var_all.shape[1]), self.h_var_all[ind, :], 'o-', color='black', lw=1, ms=2)
            plt.xticks(range(len(tick_names)), [tick_names[0]] + ['.']*(len(tick_names)-2) + [tick_names[-1]],
                       rotation=90, fontsize=6, horizontalalignment='center')
            plt.xlabel('Task', fontsize=7, labelpad=-10)
            plt.ylabel('Task Variance', fontsize=7)
            plt.title('Unit {:d}'.format(self.ind_active[ind]), fontsize=7, y=0.85)
            plt.locator_params(axis='y', nbins=3)
            ax.tick_params(axis='both', which='major', labelsize=6, length=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if save:
                plt.savefig('figure/exampleunit_variance.pdf', transparent=True)
            plt.show()

            from analysis.standard_analysis import pretty_singleneuron_plot
            # Plot single example neuron in time
            pretty_singleneuron_plot(self.model_dir, ['fdgo'], [self.ind_active[ind]],
                                     epoch=None, save=save, ylabel_firstonly=True)
            
    def plot_connectivity_byclusters(self):
        """Plot connectivity of the model"""

        ind_active = self.ind_active

        # Sort data by labels and by input connectivity
        model = Model(self.model_dir)
        hp = model.hp
        with tf.Session() as sess:
            model.restore()
            w_in = sess.run(model.w_in).T
            w_rec = sess.run(model.w_rec).T
            w_out = sess.run(model.w_out).T
            b_rec = sess.run(model.b_rec)
            b_out = sess.run(model.b_out)

        w_rec = w_rec[ind_active, :][:, ind_active]
        w_in = w_in[ind_active, :]
        w_out = w_out[:, ind_active]
        b_rec = b_rec[ind_active]

        # nx, nh, ny = hp['shape']
        nr = hp['n_eachring']

        sort_by = 'w_in'
        if sort_by == 'w_in':
            w_in_mod1 = w_in[:, 1:nr+1]
            w_in_mod2 = w_in[:, nr+1:2*nr+1]
            w_in_modboth = w_in_mod1 + w_in_mod2
            w_prefs = np.argmax(w_in_modboth, axis=1)
        elif sort_by == 'w_out':
            w_prefs = np.argmax(w_out[1:], axis=0)

        # sort by labels then by prefs
        ind_sort = np.lexsort((w_prefs, self.labels))

        ######################### Plotting Connectivity ###############################
        nx = self.hp['n_input']
        ny = self.hp['n_output']
        nh = len(self.ind_active)
        nr = self.hp['n_eachring']
        nrule = len(self.hp['rules'])

        # Plot active units
        _w_rec  = w_rec[ind_sort,:][:,ind_sort]
        _w_in   = w_in[ind_sort,:]
        _w_out  = w_out[:,ind_sort]
        _b_rec  = b_rec[ind_sort, np.newaxis]
        _b_out  = b_out[:, np.newaxis]
        labels  = self.labels[ind_sort]

        l = 0.3
        l0 = (1-1.5*l)/nh

        plot_infos = [(_w_rec              , [l               ,l          ,nh*l0    ,nh*l0]),
                      (_w_in[:,[0]]        , [l-(nx+15)*l0    ,l          ,1*l0     ,nh*l0]), # Fixation input
                      (_w_in[:,1:nr+1]     , [l-(nx+11)*l0    ,l          ,nr*l0    ,nh*l0]), # Mod 1 stimulus
                      (_w_in[:,nr+1:2*nr+1], [l-(nx-nr+8)*l0  ,l          ,nr*l0    ,nh*l0]), # Mod 2 stimulus
                      (_w_in[:,2*nr+1:]    , [l-(nx-2*nr+5)*l0,l          ,nrule*l0 ,nh*l0]), # Rule inputs
                      (_w_out[[0],:]       , [l               ,l-(4)*l0   ,nh*l0    ,1*l0]),
                      (_w_out[1:,:]        , [l               ,l-(ny+6)*l0,nh*l0    ,(ny-1)*l0]),
                      (_b_rec              , [l+(nh+6)*l0     ,l          ,l0       ,nh*l0]),
                      (_b_out              , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0       ,ny*l0])]

        # cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
        cmap = 'coolwarm'
        fig = plt.figure(figsize=(6, 6))
        for plot_info in plot_infos:
            ax = fig.add_axes(plot_info[1])
            vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5,50,95])
            _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                          vmin=vmid-(vmax-vmin)/2, vmax=vmid+(vmax-vmin)/2)
            ax.axis('off')

        ax1 = fig.add_axes([l     , l+nh*l0, nh*l0, 6*l0])
        ax2 = fig.add_axes([l-6*l0, l      , 6*l0 , nh*l0])
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
            ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
                    color=kelly_colors[il+1])
        ax1.set_xlim([0, len(labels)])
        ax2.set_ylim([0, len(labels)])
        ax1.axis('off')
        ax2.axis('off')
        if save:
            plt.savefig('figure/connectivity_by'+self.data_type+'.pdf', transparent=True)
        plt.show()

    def lesions(self):
        labels = self.labels

        from network import get_perf
        from task import generate_trials

        # The first will be the intact network
        lesion_units_list = [None]
        for il, l in enumerate(self.unique_labels):
            ind_l = np.where(labels == l)[0]
            # In original indices
            lesion_units_list += [self.ind_active[ind_l]]

        perfs_store_list = list()
        perfs_changes = list()
        cost_store_list = list()
        cost_changes = list()

        for i, lesion_units in enumerate(lesion_units_list):
            model = Model(self.model_dir)
            hp = model.hp
            with tf.Session() as sess:
                model.restore()
                model.lesion_units(sess, lesion_units)

                perfs_store = list()
                cost_store = list()
                for rule in self.rules:
                    n_rep = 16
                    batch_size_test = 256
                    batch_size_test_rep = int(batch_size_test / n_rep)
                    clsq_tmp = list()
                    perf_tmp = list()
                    for i_rep in range(n_rep):
                        trial = generate_trials(rule, hp, 'random',
                                                batch_size=batch_size_test_rep)
                        feed_dict = tools.gen_feed_dict(model, trial, hp)
                        y_hat_test, c_lsq = sess.run(
                            [model.y_hat, model.cost_lsq], feed_dict=feed_dict)

                        # Cost is first summed over time, and averaged across batch and units
                        # We did the averaging over time through c_mask

                        # IMPORTANT CHANGES: take overall mean
                        perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
                        clsq_tmp.append(c_lsq)
                        perf_tmp.append(perf_test)

                    perfs_store.append(np.mean(perf_tmp))
                    cost_store.append(np.mean(clsq_tmp))

            perfs_store = np.array(perfs_store)
            cost_store = np.array(cost_store)

            perfs_store_list.append(perfs_store)
            cost_store_list.append(cost_store)

            if i > 0:
                perfs_changes.append(perfs_store - perfs_store_list[0])
                cost_changes.append(cost_store - cost_store_list[0])

        perfs_changes = np.array(perfs_changes)
        cost_changes = np.array(cost_changes)

        return perfs_changes, cost_changes

    def plot_lesions(self):
        """Lesion individual cluster and show performance."""

        perfs_changes, cost_changes = self.lesions()

        cb_labels = ['Performance change after lesioning',
                     'Cost change after lesioning']
        vmins = [-0.5, -0.5]
        vmaxs = [+0.5, +0.5]
        ticks = [[-0.5,0.5], [-0.5, 0.5]]
        changes_plot = [perfs_changes, cost_changes]

        fs = 6
        figsize = (2.5,2.5)
        rect = [0.3, 0.2, 0.5, 0.7]
        rect_cb = [0.82, 0.2, 0.03, 0.7]
        rect_color = [0.3, 0.15, 0.5, 0.05]
        for i in range(2):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(rect)
            im = ax.imshow(changes_plot[i].T, cmap='coolwarm', aspect='auto',
                           interpolation='nearest', vmin=vmins[i], vmax=vmaxs[i])

            tick_names = [rule_name[r] for r in self.rules]
            _ = plt.yticks(range(len(tick_names)), tick_names,
                       rotation=0, va='center', fontsize=fs)
            plt.xticks([])
            plt.xlabel('Clusters', fontsize=7, labelpad=13)
            ax.tick_params('both', length=0)
            for loc in ['bottom','top','left','right']:
                ax.spines[loc].set_visible(False)

            ax = fig.add_axes(rect_cb)
            cb = plt.colorbar(im, cax=ax, ticks=ticks[i])
            cb.outline.set_linewidth(0.5)
            cb.set_label(cb_labels[i], fontsize=7, labelpad=-10)
            plt.tick_params(axis='both', which='major', labelsize=7)

            ax = fig.add_axes(rect_color)
            for il, l in enumerate(self.unique_labels):
                ax.plot([il, il+1], [0,0], linewidth=4, solid_capstyle='butt',
                        color=kelly_colors[il+1])
                ax.text(np.mean(il+0.5), -0.5, str(il+1), fontsize=6,
                        ha='center', va='top', color=kelly_colors[il+1])
            ax.set_xlim([0, len(self.unique_labels)])
            ax.set_ylim([-1, 1])
            ax.axis('off')

            if save:
                plt.savefig('figure/lesion_cluster_by'+self.data_type+'_{:d}.pdf'.format(i), transparent=True)

if __name__ == '__main__':
    root_dir = './data/train_all'
    model_dir = root_dir + '/1'
    # CA = Analysis(model_dir, data_type='rule')
