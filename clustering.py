"""
Clustering analysis
"""

from __future__ import division

import os
import numpy as np
import pickle
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from task import *
from run import Run

########################## Running the network ################################
save_addon = 'tf_latest_200'
data_type = 'epoch'

rules = [GO, INHGO, DELAYGO,\
    CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,\
    REMAP, INHREMAP, DELAYREMAP,\
    DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO]

n_rules = len(rules)

h_all_byrule  = OrderedDict()
h_all_byepoch = OrderedDict()
with Run(save_addon) as R:
    config = R.config
    for rule in rules:
        task = generate_onebatch(rule=rule, config=config, mode='test')
        h_all_byrule[rule] = R.f_h(task.x)
        for e_name, e_time in task.epochs.iteritems():
            if 'fix' not in e_name:
                h_all_byepoch[(rule, e_name)] = h_all_byrule[rule][e_time[0]:e_time[1],:,:]

# Reorder h_all_byepoch by epoch-first
keys = h_all_byepoch.keys()
ind_key_sort = np.lexsort(zip(*keys))
h_all_byepoch = OrderedDict([(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

nx, nh, ny = config['shape']
n_ring = config['N_RING']

if data_type == 'rule':
    h_all = h_all_byrule
    t_start = 500 # Important: Ignore the initial transition
    n_cluster = 12
elif data_type == 'epoch':
    h_all = h_all_byepoch
    t_start = None
    n_cluster = 15
else:
    raise ValueError

h_var_all = np.zeros((nh, len(h_all.keys())))
for i, val in enumerate(h_all.values()):
    h_var_all[:, i] = val[t_start:].reshape((-1, nh)).var(axis=0)

# Plot total variance distribution
fig = plt.figure(figsize=(1.5,1.2))
ax = fig.add_axes([0.3,0.3,0.6,0.5])
hist, bins_edge = np.histogram(np.log10(h_var_all.sum(axis=1)), bins=30)
ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0],
       color=sns.xkcd_palette(['cerulean'])[0], edgecolor='none')
plt.xlabel(r'$log_{10}$ total variance', fontsize=7, labelpad=1)
plt.ylabel('counts', fontsize=7)
plt.locator_params(nbins=3)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.savefig('figure/hist_totalvar.pdf', transparent=True)
plt.show()

# First only get active units. Total variance across tasks larger than 1e-3
ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
ind_orig   = np.arange(nh)
h_var_all  = h_var_all[ind_active, :]
ind_orig   = ind_orig[ind_active]

# Normalize by the total variance across tasks
h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

################################## Clustering ################################
# Clustering
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_cluster, affinity='cosine', linkage='average')
ac.fit(h_var_all) # n_samples, n_features = n_units, n_rules/n_epochs
labels = ac.labels_ # cluster labels

# Sort clusters by its task preference (important for consistency across nets)
label_prefs = [(np.argmax(h_normvar_all[labels==l].sum(axis=0))) for l in set(labels)]
ind_label_sort = np.argsort(label_prefs)
# Relabel
labels2 = np.zeros_like(labels)
for i, ind in enumerate(ind_label_sort):
    labels2[labels==ind] = i
labels = labels2

# Sort data by labels
ind_sort = np.argsort(labels)
labels = labels[ind_sort]
h_normvar_all = h_normvar_all[ind_sort, :]
ind_orig = ind_orig[ind_sort]

######################### Plotting Variance ###################################
# Plot Normalized Variance
if data_type == 'rule':
    figsize = (3.5,2.5)
    tick_names = [rule_name[r] for r in rules]
elif data_type == 'epoch':
    figsize = (3.5,4.5)
    tick_names = [rule_name[key[0]]+' '+key[1] for key in h_all.keys()]
else:
    raise ValueError

vmin, vmax = 0, 0.5
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0.25, 0.2, 0.6, 0.7])
im = ax.imshow(h_normvar_all.T, cmap='hot',
               aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)

plt.yticks(range(len(tick_names)), tick_names,
           rotation=0, va='center', fontsize=6)
plt.xticks([])
ax.tick_params('both', length=0)
for loc in ['bottom','top','left','right']:
    ax.spines[loc].set_visible(False)
ax = fig.add_axes([0.87, 0.2, 0.03, 0.7])
cb = plt.colorbar(im, cax=ax, ticks=[vmin,vmax])
cb.outline.set_linewidth(0.5)
cb.set_label('Normalized Variance', fontsize=7, labelpad=0)
plt.tick_params(axis='both', which='major', labelsize=7)

ax = fig.add_axes([0.25, 0.15, 0.6, 0.05])
for l in range(n_cluster):
    ind_l = np.where(labels==l)[0][[0, -1]]+np.array([0,1])
    ax.plot(ind_l, [0,0], linewidth=2, solid_capstyle='butt',
            color=sns.color_palette('deep', n_cluster)[l])
ax.set_xlim([0, len(labels)])
ax.axis('off')

plt.savefig('figure/feature_map_by'+data_type+'.pdf', transparent=True)
plt.show()

# Plot similarity matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(h_normvar_all) # TODO: check
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
im = ax.imshow(similarity, cmap='hot', interpolation='none', vmin=0, vmax=1)
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
plt.savefig('figure/feature_similarity_by'+data_type+'.pdf', transparent=True)
plt.show()

######################### Plotting Connectivity ###############################
with Run(save_addon) as R:
    params = R.params
    w_rec  = R.w_rec
    w_in   = R.w_in
    w_out  = R.w_out
    b_rec  = R.b_rec
    b_out  = R.b_out

nh = len(ind_orig)
nr = n_ring
l = 0.25
l0 = (1-1.5*l)/nh
plot_infos = [(w_rec[ind_orig,:][:,ind_orig], [l               ,l          ,nh*l0       ,nh*l0]),
              (w_in[ind_orig,:nr]           , [l-(nx+12)*l0    ,l          ,nr*l0       ,nh*l0]),
              (w_in[ind_orig,nr:2*nr]       , [l-(nx-nr+9)*l0  ,l          ,nr*l0       ,nh*l0]),
              (w_in[ind_orig,2*nr:]         , [l-(nx-2*nr+6)*l0,l          ,(nx-2*nr)*l0,nh*l0]),
              (w_out[:, ind_orig]           , [l               ,l-(ny+6)*l0,nh*l0       ,ny*l0]),
              (b_rec[ind_orig, np.newaxis]  , [l+(nh+6)*l0     ,l          ,l0          ,nh*l0]),
              (b_out[:, np.newaxis]         , [l+(nh+6)*l0     ,l-(ny+6)*l0,l0          ,ny*l0])]

cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
fig = plt.figure(figsize=(4,4))
for plot_info in plot_infos:
    ax = fig.add_axes(plot_info[1])
    vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5,50,95])
    _ = ax.imshow(plot_info[0], interpolation='none', cmap=cmap, aspect='auto',
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
plt.savefig('figure/connectivity_by'+data_type+'.pdf', transparent=True)
plt.show()



#==============================================================================
# with Run(save_addon) as R:
#     config = R.config
#     rule = DMCGO
#     task = generate_onebatch(rule=rule, config=config, mode='test')
#     h0 = R.f_h(task.x)
#     y0 = R.f_y(h0)
#==============================================================================
