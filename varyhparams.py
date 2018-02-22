"""Analyze the results after varying hyperparameters."""

from __future__ import division

import os
#import numpy as np
import matplotlib.pyplot as plt

import tools
#import variance
import clustering

DATAPATH = os.path.join(os.getcwd(), 'data', 'debug')

# Get all the subdirectories
train_dirs = [os.path.join(DATAPATH, d) for d in os.listdir(DATAPATH)]
train_dirs = [d for d in train_dirs if os.path.isdir(d)]

from os import listdir
#check if training is completed 
from os.path import isfile, join
train_dirs = [d for d in train_dirs if 'variance_rule.pkl' in [f for f in listdir(d) if isfile(join(d, f))]]#maddy added


# Compute the number of clusters
n_clusters = list()
scores = list()

hparams_list = list()
for train_dir in train_dirs:
    hparams = tools.load_hparams(train_dir)
    #check if performance exceeds target
    log = tools.load_log(train_dir) 
    #if log['perf_avg'][-1] > hparams['target_perf']: 
    if log['perf_min'][-1] > hparams['target_perf']:         
    
        # variance.compute_variance(train_dir)
        analysis = clustering.Analysis(train_dir, 'rule')
        n_clusters.append(analysis.n_cluster)
        scores.append(analysis.scores)
        analysis.plot_cluster_score()
        hparams_list.append(hparams)
    
        #analysis.plot_example_unit()
        analysis.plot_variance()
        analysis.plot_2Dvisualization()


plt.figure()
plt.plot(n_clusters)
#min score 90% -- softplus ---- pick cluster with max sil score. 

