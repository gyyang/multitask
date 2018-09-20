"""
Analysis of anti units
"""

from __future__ import division

import numpy as np
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from network import Model, get_perf
from task import get_dist, generate_trials
import tools

save = True


class Analysis(object):
    """Analyze the Anti tasks."""
    def __init__(self, model_dir):
        self.model_dir = model_dir

        # Run model
        model = Model(model_dir)
        self.hp = model.hp
        with tf.Session() as sess:
            model.restore()
            for name in ['w_in', 'w_rec', 'w_out', 'b_rec', 'b_out']:
                setattr(self, name, sess.run(getattr(model, name)))
                if 'w_' in name:
                    setattr(self, name, getattr(self, name).T)

        data_type  = 'rule'
        with open(model_dir + '/variance_'+data_type+'.pkl', 'rb') as f:
            res = pickle.load(f)
        h_var_all = res['h_var_all']
        self.rules = res['keys']

        # First only get active units. Total variance across tasks larger than 1e-3
        ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
        # ind_active = np.where(h_var_all.sum(axis=1) > 0.)[0]
        h_var_all = h_var_all[ind_active, :]

        # Normalize by the total variance across tasks
        h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

        ########################## Get Anti Units ####################################
        # Directly search
        # This will be a stricter subset of the anti modules found in clustering results
        self.rules_anti = np.array(['fdanti', 'reactanti', 'delayanti'])
        self.rules_nonanti = np.array([r for r in self.rules if r not in self.rules_anti])

        # Rule index used only for the rules
        self.ind_rules_anti = [self.rules.index(r) for r in self.rules_anti]
        self.ind_rules_nonanti = [self.rules.index(r) for r in self.rules_nonanti]

        self.h_normvar_all_anti = h_normvar_all[:, self.ind_rules_anti].sum(axis=1)
        self.h_normvar_all_nonanti = h_normvar_all[:, self.ind_rules_nonanti].sum(axis=1)

        # plt.figure()
        # _ = plt.hist(h_normvar_all_anti, bins=50)
        # plt.xlabel('Proportion of variance in anti tasks')
        # plt.show()

        ind_anti = np.where(self.h_normvar_all_anti>0.5)[0]
        ind_nonanti = np.where(self.h_normvar_all_anti<=0.5)[0]
        self.ind_anti_orig = ind_active[ind_anti] # Indices of anti units in the original matrix
        self.ind_nonanti_orig = ind_active[ind_nonanti]

        # Use clustering results (tend to be loose)
        # label_anti    = np.where(label_prefs==FDANTI)[0][0]
        # ind_anti      = np.where(labels==label_anti)[0]
        # ind_anti_orig = ind_orig[ind_anti] # Indices of anti units in the original matrix

    def plot_example_unit(self):
        """Plot activity of an example unit."""
        from standard_analysis import pretty_singleneuron_plot
        pretty_singleneuron_plot(
            self.model_dir, ['fdanti', 'fdgo'], self.ind_anti_orig[2],
            save=save, ylabel_firstonly = True)

    def plot_inout_connections(self):
        """Plot the input and output connections."""
        n_eachring = self.hp['n_eachring']
        w_in, w_out = self.w_in, self.w_out

        w_in_ = (w_in[:, 1:n_eachring+1]+w_in[:, 1+n_eachring:2*n_eachring+1])/2.
        w_out_ = w_out[1:, :].T

        for ind_group, unit_type in zip([self.ind_anti_orig, self.ind_nonanti_orig],
                                        ['Anti units', 'Non-Anti Units']):
            # ind_group = ind_anti_orig
            n_group    = len(ind_group)
            w_in_group = np.zeros((n_group, n_eachring))
            w_out_group = np.zeros((n_group, n_eachring))

            ind_pref_ins = list()
            ind_pref_outs = list()

            for i, ind in enumerate(ind_group):
                tmp_in           = w_in_[ind, :]
                tmp_out           = w_out_[ind, :]

                # Get preferred input and output directions
                ind_pref_in       = np.argmax(tmp_in)
                ind_pref_out      = np.argmax(tmp_out)

                ind_pref_ins.append(ind_pref_in)
                ind_pref_outs.append(ind_pref_out)

                # Sort by preferred input direction
                w_in_group[i, :] = np.roll(tmp_in, int(n_eachring/2)-ind_pref_in)
                w_out_group[i, :] = np.roll(tmp_out, int(n_eachring/2)-ind_pref_in)

            w_in_ave = w_in_group.mean(axis=0)
            w_out_ave = w_out_group.mean(axis=0)

            fs = 6
            fig = plt.figure(figsize=(1.5, 1.0))
            ax = fig.add_axes([.3, .3, .6, .6])
            ax.plot(w_in_ave, color='black', label='In')
            ax.plot(w_out_ave, color='red', label='Out')
            ax.set_xticks([int(n_eachring/2)])
            ax.set_xticklabels(['Preferred input dir.'])
            # ax.set_xlabel(xlabel, fontsize=fs, labelpad=3)
            ax.set_ylabel('Conn. weight', fontsize=fs)
            lg = ax.legend(fontsize=fs, bbox_to_anchor=(1.1,1.1),
                           labelspacing=0.2, loc=1, frameon=False)
            # plt.setp(lg.get_title(),fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.set_title(unit_type, fontsize=fs, y=0.9)
            plt.locator_params(axis='y',nbins=3)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if save:
                plt.savefig('figure/conn_'+unit_type+'.pdf', transparent=True)

    def plot_rule_connections(self):
        """Plot connectivity from the rule input units"""

        # Rule index for the connectivity
        from task import get_rule_index
        indconn_rules_anti     = [get_rule_index(r, self.hp) for r in self.rules_anti]
        indconn_rules_nonanti  = [get_rule_index(r, self.hp) for r in self.rules_nonanti]

        for ind, unit_type in zip([self.ind_anti_orig, self.ind_nonanti_orig],
                                  ['Anti units', 'Non-Anti units']):
            b1 = self.w_in[:, indconn_rules_anti][ind, :].flatten()
            b2 = self.w_in[:, indconn_rules_nonanti][ind, :].flatten()

            fs = 6
            fig = plt.figure(figsize=(1.5,1.2))
            ax = fig.add_axes([0.3,0.3,0.6,0.4])
            ax.boxplot([b1, b2], showfliers=False)
            ax.set_xticklabels(['Anti', 'Non-Anti'])
            ax.set_xlabel('Input from rule units', fontsize=fs, labelpad=3)
            ax.set_ylabel('Conn. weight', fontsize=fs)
            ax.set_title('To '+unit_type, fontsize=fs, y=0.9)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            plt.locator_params(axis='y',nbins=2)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if save:
                plt.savefig('figure/connweightrule_'+unit_type+'.pdf', transparent=True)
            plt.show()

    def plot_rec_connections(self):
        """Plot connectivity between recurrent units"""

        n_eachring = self.hp['n_eachring']
        w_in, w_rec, w_out = self.w_in, self.w_rec, self.w_out

        w_in_ = (w_in[:, 1:n_eachring+1]+w_in[:, 1+n_eachring:2*n_eachring+1])/2.
        w_out_ = w_out[1:, :].T

        inds = [self.ind_nonanti_orig, self.ind_anti_orig]
        names = ['Non-Anti', 'Anti']

        i_pairs = [(0,0), (0,1), (1,0), (1,1)]

        pref_diffs_list = list()
        w_recs_list = list()

        w_rec_bygroup = np.zeros((2,2))

        for i_pair in i_pairs:
            ind1, ind2 = inds[i_pair[0]], inds[i_pair[1]]
            # For each neuron get the preference based on input weight
            # sort by weights
            w_sortby = w_in_
            # w_sortby = w_out_
            prefs1 = np.argmax(w_sortby[ind1, :], axis=1)*2.*np.pi/n_eachring
            prefs2 = np.argmax(w_sortby[ind2, :], axis=1)*2.*np.pi/n_eachring

            # Compute the pairwise distance based on preference
            # Then get the connection weight between pairs
            pref_diffs = list()
            w_recs = list()
            for i, ind_i in enumerate(ind1):
                for j, ind_j in enumerate(ind2):
                    if ind_j == ind_i:
                        # Excluding self connections, which tend to be positive
                        continue
                    pref_diffs.append(get_dist(prefs1[i]-prefs2[j]))
                    # pref_diffs.append(prefs1[i]-prefs2[j])
                    w_recs.append(w_rec[ind_j, ind_i])
            pref_diffs, w_recs = np.array(pref_diffs), np.array(w_recs)
            pref_diffs_list.append(pref_diffs)
            w_recs_list.append(w_recs)

            w_rec_bygroup[i_pair[1], i_pair[0]] = np.mean(w_recs[pref_diffs<np.pi/6.])


        fs = 6
        vmax = np.ceil(np.max(w_rec_bygroup)*100)/100.
        vmin = np.floor(np.min(w_rec_bygroup)*100)/100.
        fig = plt.figure(figsize=(1.5,1.5))
        ax = fig.add_axes([0.2, 0.1, 0.5, 0.5])
        im = ax.imshow(w_rec_bygroup, cmap='coolwarm',
                       aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        plt.yticks(range(len(names)), names,
                   rotation=90, va='center', fontsize=fs)
        plt.xticks(range(len(names)), names,
                   rotation=0, ha='center', fontsize=fs)
        ax.tick_params('both', length=0)
        ax.set_xlabel('From', fontsize=fs, labelpad=2)
        ax.set_ylabel('To', fontsize=fs, labelpad=2)
        for loc in ['bottom','top','left','right']:
            ax.spines[loc].set_visible(False)

        ax = fig.add_axes([0.72, 0.1, 0.03, 0.5])
        cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Rec. weight', fontsize=fs, labelpad=-7)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        if save:
            plt.savefig('figure/conn_anti_task_recurrent.pdf', transparent=True)

    def plot_rec_connections_temp(self):
        raise NotImplementedError()
        # Continue to plot the recurrent connections by difference in input weight
        fs = 6
        fig = plt.figure(figsize=(3, 3.0))
        ax = fig.add_axes([.3, .3, .6, .6])

        for i, i_pair in enumerate(i_pairs):
            bin_means, bin_edges, binnumber = binned_statistic(
                pref_diffs_list[i], w_recs_list[i], bins=6, statistic='mean')

            ax.plot(bin_means, label=names[i_pair[0]]+' to '+names[i_pair[1]])
            ax.set_xticks([int(n_eachring/2)])
            ax.set_xticklabels(['preferred input loc.'])
            # ax.set_xlabel(xlabel, fontsize=fs, labelpad=3)
            ax.set_ylabel('conn. weight', fontsize=fs)
            lg = ax.legend(fontsize=fs, bbox_to_anchor=(1.1,1.1),
                           labelspacing=0.2, loc=1, frameon=False)
            # plt.setp(lg.get_title(),fontsize=fs)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            # ax.set_title(names[i_pair[0]]+' to '+names[i_pair[1]], fontsize=fs, y=0.9)
            plt.locator_params(axis='y',nbins=3)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            # if save:
            #     plt.savefig('figure/conn_'+unit_type+'_'+save_addon+'.pdf', transparent=True)


    def plot_lesions(self):
        """Plot results of lesioning."""

        n_hidden = self.hp['n_rnn']
        # Randomly select a group to lesion. The group has the same size as the anti group
        ind_randomselect = np.arange(n_hidden)
        np.random.shuffle(ind_randomselect)
        ind_randomselect = ind_randomselect[:len(self.ind_anti_orig)]

        # Performance for the anti and non-anti tasks
        perfs_nonanti = list()
        perfs_anti = list()

        # Units to be lesioned
        lesion_units_list = [None, self.ind_anti_orig, ind_randomselect]
        names = ['Control', 'Anti units all lesioned', 'Random group lesioned']

        for lesion_units in lesion_units_list:
            model = Model(self.model_dir)
            hp = model.hp
            with tf.Session() as sess:
                model.restore()
                model.lesion_units(sess, lesion_units)

                perfs_store = list()
                for rule in self.rules:
                    trial = generate_trials(rule, hp, mode='test')
                    feed_dict = tools.gen_feed_dict(model, trial, hp)
                    y_hat_test = sess.run(model.y_hat, feed_dict=feed_dict)
                    perf = np.mean(get_perf(y_hat_test, trial.y_loc))
                    perfs_store.append(perf)

            perfs_store = np.array(perfs_store)

            perfs_nonanti.append(np.mean(perfs_store[self.ind_rules_nonanti]))
            perfs_anti.append(np.mean(perfs_store[self.ind_rules_anti]))


        fs = 6
        width = 0.2
        n_bars = len(lesion_units_list)

        fig = plt.figure(figsize=(1.5,1.2))
        ax = fig.add_axes([0.3,0.3,0.6,0.3])

        bars = list()
        colors = ['orange', 'green', 'violet']
        for i in range(n_bars):
            b = ax.bar(np.arange(2)+(1-n_bars)*width/2+width*i,
                       [perfs_nonanti[i], perfs_anti[i]], width=width,
                       color='xkcd:'+colors[i], edgecolor='none')
            bars.append(b)

        ax.plot(np.arange(n_bars)*width+(1-n_bars)*width/2  , perfs_nonanti, '.-', color='gray', lw=0.75, ms=3)
        ax.plot(np.arange(n_bars)*width+(1-n_bars)*width/2+1, perfs_anti   , '.-', color='gray', lw=0.75, ms=3)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(['Non-Anti', 'Anti'])
        ax.set_xlabel('Tasks', fontsize=fs, labelpad=3)
        ax.set_ylabel('Performance', fontsize=fs)
        lg = ax.legend(bars, names,
                       fontsize=fs, ncol=1, bbox_to_anchor=(1,2.2),
                       labelspacing=0.2, loc=1, frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(axis='y',nbins=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #    ax.set_xlim([-0.5, 1.5])
        if save:
            plt.savefig('figure/perflesion_anti_units.pdf', transparent=True)
        plt.show()




# # A.plot_unit_connection(inds)
# ga = GeneralAnalysis(save_addon=A.save_addon_original)
# success = False
# for ind in inds_anti:
#     res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+n_eachring])
#     if res.rvalue<-0.9 and res.pvalue<0.01:
#         success = True
#         break
#
# if success:
#     rules = [FDGO, DELAYANTI, FDANTI]
#     ga.run_test(rules=rules)
#     ga.plot_singleneuron_intime(neuron_id=ind, rules=rules, save_fig=True)
# else:
#     raise ValueError('Did not success to find a typical anti unit in this network')
#
#

# ########### Plot histogram of correlation coefficient #####################
# slopes = list()
# rvalues = list()
# pvalues = list()
# anti_units = list()
# HDIMs = range(190,207)
# for j, HDIM in enumerate(HDIMs):
#     n_eachring = 16
#     save_addon = 'chanceabbott_'+str(HDIM)+'_'+str(n_eachring)
#     A = SumStatAnalysis('rule', save_addon)
#     inds_anti = A.find_anti_units()
#
#     # A.plot_unit_connection(inds)
#     ga = GeneralAnalysis(save_addon=A.save_addon_original)
#     for ind in range(HDIM):
#         res = linregress(ga.Wout[:,ind][1:],ga.Win[ind,1:1+n_eachring])
#         slopes.append(res.slope)
#         rvalues.append(res.rvalue)
#         pvalues.append(res.pvalue)
#         anti_units.append(ind in inds_anti)
#
#     if j == 0:
#         conn_rule_to_all = ga.Win[:, 1+2*n_eachring:] # connection from rule inputs to all units
#         conn_rule_to_anti = ga.Win[inds_anti, 1+2*n_eachring:] # connection from rule inputs to anti units
#     else:
#         conn_rule_to_all = np.concatenate((conn_rule_to_all, ga.Win[:, 1+2*n_eachring:]))
#         conn_rule_to_anti = np.concatenate((conn_rule_to_anti, ga.Win[inds_anti, 1+2*n_eachring:]))
#
#
# slopes, rvalues, pvalues, anti_units = np.array(slopes), np.array(rvalues), np.array(pvalues), np.array(anti_units)
#
# # plot_value, plot_range = slopes, (-4,4)
# plot_value, plot_range = rvalues, (-1,1)
# thres = 0.01 ##TODO: Find out how to set this threshold
# for i in [0,1]:
#     if i == 0:
#         units = anti_units
#         title = 'Anti'
#     else:
#         units = (1-anti_units).astype(bool)
#         title = 'Non-anti'
#
#     fig = plt.figure(figsize=(1.5,1.2))
#     ax = fig.add_axes([0.3,0.3,0.6,0.5])
#     hist, bins_edge = np.histogram(plot_value[units], range=plot_range, bins=30)
#     ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=sns.xkcd_palette(['navy blue'])[0], edgecolor='none')
#     hist, bins_edge = np.histogram(plot_value[units*(pvalues<thres)], range=plot_range, bins=30)
#     ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
#     plt.xlabel('corr. coeff.', fontsize=7, labelpad=1)
#     plt.ylabel('counts', fontsize=7)
#     plt.locator_params(nbins=3)
#     plt.title(title+' units ({:d} nets)'.format(len(HDIMs)) , fontsize=7)
#     ax.tick_params(axis='both', which='major', labelsize=7)
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     plt.savefig('figure/'+title+'_unit_cc.pdf', transparent=True)
#     plt.show()
#
# ########### Plot connections averaged across networks #####################
# fig = plt.figure(figsize=(2.5,2))
# ax = fig.add_axes([0.2,0.5,0.75,0.4])
# ax.plot(conn_rule_to_all.mean(axis=0)[ga.rules], 'o-', markersize=3, label='all', color=sns.xkcd_palette(['black'])[0])
# ax.plot(conn_rule_to_anti.mean(axis=0)[ga.rules], 'o-', markersize=3, label='anti', color=sns.xkcd_palette(['red'])[0])
#
# plt.xticks(range(len(ga.rules)), [rule_name[rule] for rule in ga.rules], rotation=90)
# plt.ylabel('Mean conn. from rule', fontsize=7)
# plt.title('Average of {:d} networks'.format(len(HDIMs)), fontsize=7)
# #plt.ylim(bottom=0.0, top=1.05)
# plt.xlim([-0.5, len(ga.rules)-0.5])
# ax.tick_params(axis='both', which='major', labelsize=7)
# leg = plt.legend(title='to units',fontsize=7,frameon=False, loc=2, numpoints=1,
#                  bbox_to_anchor=(0.3,0.6), labelspacing=0.3)
# plt.setp(leg.get_title(),fontsize=7)
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# plt.locator_params(axis='y', nbins=4)
# plt.savefig('figure/conn_ruleinput_to_anti.pdf', transparent=True)
# plt.show()
