#! /usr/bin/env python
"""
Launching jobs on the NYU cluster
"""
from __future__ import absolute_import

import os
import argparse
import subprocess
import numpy as np

import tools

parser = argparse.ArgumentParser()
parser.add_argument('run')
args = parser.parse_args()

sbatchpath = './sbatch/'
scratchpath = '/scratch/mj98-share/multitask-master_testmaddy/'

if args.run == 'all':
    for seed in range(0, 20):
        jobname = 'train_all_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_all('''+\
              train_arg+''')"'''

        jobfile = tools.write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

if args.run == 'all_varyhp': 
    for seed in range(0, 64):
        jobname = 'train_varyhp_{:d}'.format(seed)
        train_arg = '{:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_vary_hparams('''+\
              train_arg+''')"'''

        jobfile = tools.write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

if args.run == 'mante':
    for seed in range(0, 20):
        jobname = 'train_mante_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_mante('''+\
              train_arg+''')"'''

        jobfile = tools.write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

# Current continual learning version
if args.run == 'cont':
    s = 1
    n_unit = 256
    for seed in range(0,20):
        for condition in [0, 1, 2, 3]:
            if condition == 0:
                c, ksi, seq = 0.1, 0.01, True
            elif condition == 1:
                c, ksi, seq = 1.0, 0.01, True
            elif condition == 2:
                c, ksi, seq = 0, 0, True
            elif condition == 3:
                c, ksi, seq = 0, 0, False

            jobname = 'cont{:d}_{:d}'.format(seed, condition)
            train_arg = 'c={:0.6f}, ksi={:0.6f}, seed={:d}'.format(c, ksi, seed)
            train_arg+= r", save_name='"+'{:d}_{:d}cont'.format(seed, condition)+r"'"
            train_arg+= ',seq='+str(seq)

            cmd     = r'''python -c "import paper as p;p.cont_train('''+train_arg+''')"'''

            jobfile = tools.write_jobfile(cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
            subprocess.call(['sbatch', jobfile])

# Grid search
if args.run == 'grid':
    raise NotImplementedError()
    s = 1
    n_unit = 256
    for seed in range(5):
        for i_c, c_intsyn in enumerate([0.1, 1.0]):
            for i_ksi, ksi_intsyn in enumerate([0.01, 0.1, 1.0]):
                jobname = 'grid{:d}_{:d}_{:d}'.format(seed, i_ksi, i_c)
                train_arg = 'c={:0.6f}, ksi={:0.6f}, seed={:d}'.format(
                    c_intsyn, ksi_intsyn, seed)
                train_arg+= r", save_name='"+'{:d}_{:d}_{:d}grid'.format(seed, i_ksi, i_c)+r"'"

                cmd     = r'''python -c "import paper as p;p.cont_train('''+train_arg+''')"'''

                jobfile = tools.write_jobfile(cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
                subprocess.call(['sbatch', jobfile])
