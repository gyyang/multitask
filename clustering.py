"""
Clustering analysis
"""

from __future__ import division

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn.apionly as sns

from task import *
from run import Run
from general_analysis import GeneralAnalysis

########################## Running the network ################################
save_addon = 'tf_300_16'

rules = [GO, INHGO, DELAYGO,\
    CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,\
    REMAP, INHREMAP, DELAYREMAP,\
    DELAYMATCHGO, DELAYMATCHNOGO, DMCGO]
# rules += [DMCNOGO]

n_rules = len(rules)

h_all = dict()

with GeneralAnalysis(save_addon) as R:
# with Run(save_addon) as R:
    config = R.config
    for rule in rules:
        task = generate_onebatch(rule=rule, config=config, mode='test')
        h_all[rule] = R.f_h(task.x)


# with Run(save_addon) as R:
#     params = R.params
#     w_rec  = R.w_rec
#     w_in   = R.w_in

n_x, n_h, n_y = config['shape']

h_var_all = np.zeros((n_h, n_rules))
for i, rule in enumerate(rules):
    h_var_all[:, i] = h_all[rule].reshape((-1, n_h)).var(axis=0)

# First only get active units. Total variance across tasks larger than 1e-3
ind_active = np.where(h_var_all.sum(axis=1) > 1e-3)[0]
h_var_all  = h_var_all[ind_active, :]

# Normalize by the total variance across tasks
h_normvar_all = (h_var_all.T/np.sum(h_var_all, axis=1)).T

################################## Clustering ################################
# Clustering
n_cluster = 10
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

################################## Plotting ###################################
# Plot Normalized Variance
fig = plt.figure(figsize=(3.5,2.5))
ax = fig.add_axes([0.25, 0.2, 0.6, 0.7])
im = ax.imshow(h_normvar_all[ind_sort, :].T, cmap='hot',
               aspect='auto', interpolation='none', vmin=0, vmax=1)
tick_names = [rule_name[r] for r in rules]
plt.yticks(range(len(tick_names)), tick_names,
           rotation=0, va='center', fontsize=6)
plt.xticks([])
ax.tick_params('both', length=0)
for loc in ['bottom','top','left','right']:
    ax.spines[loc].set_visible(False)
ax = fig.add_axes([0.87, 0.2, 0.03, 0.7])
cb = plt.colorbar(im, cax=ax, ticks=[0,1])
cb.outline.set_linewidth(0.5)
cb.set_label('Normalized Variance', fontsize=7, labelpad=0)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.savefig('figure/feature_map.pdf', transparent=True)
plt.show()