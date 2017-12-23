"""Analyze the results after varying hyperparameters."""

from __future__ import division

import os

import tools

import variance

DATAPATH = os.path.join(os.getcwd(), 'data', 'debug')

train_dirs = [os.path.join(DATAPATH, d) for d in os.listdir(DATAPATH)]

for train_dir in train_dirs:
    hparams = tools.load_hparams(train_dir)
    variance.compute_variance(train_dir)
    # summary = do_some_analysis(train_dir)