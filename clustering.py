"""
Clustering analysis
Analyze how units are involved in various tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from task import *
from run import Run, plot_singleneuron_intime

save = True

normalization_method = 'max'
# normalization_method = 'sum'

########################## Running the network ################################
save_addon = 'allrule_softplus_0_256paper'
# data_type = 'rule'
data_type = 'epoch'

rules = range(N_RULE)

# rules = [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2]

# If not computed, use variance.py
fname = 'data/variance'+data_type+save_addon
with open(fname+'.pkl','rb') as f:
    res = pickle.load(f)
h_var_all_= res['h_var_all']
keys      = res['keys']

# First only get active units. Total variance across tasks larger than 1e-3
ind_active = np.where(h_var_all_.sum(axis=1) > 1e-2)[0]
h_var_all  = h_var_all_[ind_active, :]

# Normalize by the total variance across tasks
if normalization_method == 'sum':
    h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T
elif normalization_method == 'max':
    h_normvar_all = (h_var_all.T/np.max(h_var_all, axis=1)).T
else:
    raise ValueError()

################################## Clustering ################################
if data_type == 'rule':
    n_cluster = 15
    # n_cluster = 3
elif data_type == 'epoch':
    n_cluster = 15
else:
    raise ValueError

# Clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_cluster, affinity='cosine', linkage='average')
ac.fit(h_var_all) # n_samples, n_features = n_units, n_rules/n_epochs
labels = ac.labels_ # cluster labels

# Sort clusters by its task preference (important for consistency across nets)
if data_type == 'rule':
    label_prefs = [rules[np.argmax(h_normvar_all[labels==l].sum(axis=0))] for l in set(labels)]
elif data_type == 'epoch':
    label_prefs = [keys[np.argmax(h_normvar_all[labels==l].sum(axis=0))][0] for l in set(labels)]

ind_label_sort = np.argsort(label_prefs)
label_prefs = np.array(label_prefs)[ind_label_sort]
# Relabel
labels2 = np.zeros_like(labels)
for i, ind in enumerate(ind_label_sort):
    labels2[labels==ind] = i
labels = labels2

# Sort data by labels and by input connectivity
with Run(save_addon, sigma_rec=0) as R:
    w_in  = R.w_in # for later sorting
    w_out = R.w_out
    config = R.config
nx, nh, ny = config['shape']
n_ring = config['N_RING']

sort_by = 'w_in'
if sort_by == 'w_in':
    w_in = w_in[ind_active, :]
    w_in_mod1 = w_in[:, 1:n_ring+1]
    w_in_mod2 = w_in[:, n_ring+1:2*n_ring+1]
    w_in_modboth = w_in_mod1 + w_in_mod2
    w_prefs = np.argmax(w_in_modboth, axis=1)
elif sort_by == 'w_out':
    w_out = w_out[1:, ind_active]
    w_prefs = np.argmax(w_out, axis=0)

ind_sort        = np.lexsort((w_prefs, labels)) # sort by labels then by prefs
labels          = labels[ind_sort]
h_normvar_all   = h_normvar_all[ind_sort, :]
ind_active      = ind_active[ind_sort]

# # Save results
# result = {'labels'          : labels,
#           'label_prefs'     : label_prefs,
#           'h_normvar_all'   : h_normvar_all,
#           'ind_active'      : ind_active,
#           'rules'           : rules,
#           'data_type'       : data_type}
#
# with open('data/clustering'+save_addon+'.pkl','wb') as f:
#     pickle.dump(result, f)

######################### Plotting Variance ###################################
# Plot Normalized Variance
if data_type == 'rule':
    figsize = (3.5,2.5)
    rect = [0.25, 0.2, 0.6, 0.7]
    rect_color = [0.25, 0.15, 0.6, 0.05]
    rect_cb = [0.87, 0.2, 0.03, 0.7]
    tick_names = [rule_name[r] for r in rules]
    fs = 6
elif data_type == 'epoch':
    figsize = (3.5,4.5)
    rect = [0.25, 0.1, 0.6, 0.85]
    rect_color = [0.25, 0.05, 0.6, 0.05]
    rect_cb = [0.87, 0.1, 0.03, 0.85]
    tick_names = [rule_name[key[0]]+' '+key[1] for key in keys]
    fs = 5
else:
    raise ValueError

h_plot  = h_normvar_all.T
vmin, vmax = 0, 1
fig = plt.figure(figsize=figsize)
ax = fig.add_axes(rect)
im = ax.imshow(h_plot, cmap='hot',
               aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

plt.yticks(range(len(tick_names)), tick_names,
           rotation=0, va='center', fontsize=fs)
plt.xticks([])
ax.tick_params('both', length=0)
for loc in ['bottom','top','left','right']:
    ax.spines[loc].set_visible(False)
ax = fig.add_axes(rect_cb)
cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
cb.outline.set_linewidth(0.5)
if normalization_method == 'sum':
    clabel = 'Sum-normalized Variance'
elif normalization_method == 'max':
    clabel = 'Max-normalized Variance'

cb.set_label(clabel, fontsize=7, labelpad=0)
plt.tick_params(axis='both', which='major', labelsize=7)

# Plot color bars indicating clustering
if True:
    ax = fig.add_axes(rect_color)
    for l in range(n_cluster):
        ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
        ax.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
                color=sns.color_palette('deep', n_cluster)[l])
    ax.set_xlim([0, len(labels)])
    ax.axis('off')

if save:
    plt.savefig('figure/feature_map_by'+data_type+'_norm'+normalization_method+'.pdf', transparent=True)
plt.show()

######################### Plotting Similarity Matrix ##########################

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(h_normvar_all) # TODO: check
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
for l in range(n_cluster):
    ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
    ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
            color=sns.color_palette('deep', n_cluster)[l])
    ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
            color=sns.color_palette('deep', n_cluster)[l])
ax1.set_xlim([0, len(labels)])
ax2.set_ylim([0, len(labels)])
ax1.axis('off')
ax2.axis('off')
if save:
    plt.savefig('figure/feature_similarity_by'+data_type+'.pdf', transparent=True)
plt.show()


######################## Plotting Variance for example unit ###################
if data_type == 'rule':
    ind = 2 # example unit
    fig = plt.figure(figsize=(1.2,1.0))
    ax = fig.add_axes([0.4,0.4,0.5,0.45])
    ax.plot(range(h_var_all.shape[1]), h_var_all_[ind_active[ind], :], 'o-', color='black', lw=1, ms=2)
    plt.xticks(range(len(tick_names)), [tick_names[0]] + ['.']*(len(tick_names)-2) + [tick_names[-1]],
               rotation=90, fontsize=6, horizontalalignment='center')
    plt.xlabel('Task', fontsize=7, labelpad=-10)
    plt.ylabel('Task Variance', fontsize=7)
    plt.title('Unit {:d}'.format(ind_active[ind]), fontsize=7, y=0.85)
    plt.locator_params(axis='y', nbins=3)
    ax.tick_params(axis='both', which='major', labelsize=6, length=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save:
        plt.savefig('figure/exampleunit_variance.pdf', transparent=True)
    plt.show()

    # Plot single example neuron in time
    plot_singleneuron_intime(save_addon, [ind_active[ind]], [FDGO],
                                 epoch=None, save=True, ylabel_firstonly=True)

######################### Plotting Connectivity ###############################
nh = len(ind_active)
nr = n_ring
nrule = (nx-2*nr-1)
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

plot_infos = [(w_rec              , [l               ,l          ,nh*l0    ,nh*l0]),
              (w_in[:,[0]]        , [l-(nx+15)*l0    ,l          ,1*l0     ,nh*l0]), # Fixation input
              (w_in[:,1:nr+1]     , [l-(nx+11)*l0    ,l          ,nr*l0    ,nh*l0]), # Mod 1 stimulus
              (w_in[:,nr+1:2*nr+1], [l-(nx-nr+8)*l0  ,l          ,nr*l0    ,nh*l0]), # Mod 2 stimulus
              (w_in[:,2*nr+1:]    , [l-(nx-2*nr+5)*l0,l          ,nrule*l0 ,nh*l0]), # Rule inputs
              (w_out[[0],:]       , [l               ,l-(4)*l0   ,nh*l0    ,1*l0]),
              (w_out[1:,:]        , [l               ,l-(ny+6)*l0,nh*l0    ,(ny-1)*l0]),
              (b_rec              , [l+(nh+6)*l0     ,l          ,l0       ,nh*l0]),
              (b_out              , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0       ,ny*l0])]

cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
fig = plt.figure(figsize=(6,6))
for plot_info in plot_infos:
    ax = fig.add_axes(plot_info[1])
    vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5,50,95])
    _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                  vmin=vmid-(vmax-vmin)/2, vmax=vmid+(vmax-vmin)/2)
    ax.axis('off')

ax1 = fig.add_axes([l     , l+nh*l0, nh*l0, 6*l0])
ax2 = fig.add_axes([l-6*l0, l      , 6*l0 , nh*l0])
for l in range(n_cluster):
    ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
    ax1.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
            color=sns.color_palette('deep', n_cluster)[l])
    ax2.plot([0,0], len(labels)-ind_l, linewidth=2, solid_capstyle='butt',
            color=sns.color_palette('deep', n_cluster)[l])
ax1.set_xlim([0, len(labels)])
ax2.set_ylim([0, len(labels)])
ax1.axis('off')
ax2.axis('off')
if save:
    plt.savefig('figure/connectivity_by'+data_type+'.pdf', transparent=True)
plt.show()