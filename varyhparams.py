"""Analyze the results after varying hyperparameters."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt

import tools
import variance
import clustering

DATAPATH = os.path.join(os.getcwd(), 'data', 'debug')

# Get all the subdirectories
train_dirs = [os.path.join(DATAPATH, d) for d in os.listdir(DATAPATH)]
train_dirs = [d for d in train_dirs if os.path.isdir(d)]

# Compute the number of clusters
n_clusters = list()
hparams_list = list()
for train_dir in train_dirs:
    hparams = tools.load_hparams(train_dir)
    # variance.compute_variance(train_dir)
    analysis = clustering.Analysis(train_dir, 'rule')
    n_clusters.append(analysis.n_cluster)
    hparams_list.append(hparams)

plt.figure()
plt.plot(n_clusters)
