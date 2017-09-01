#! /usr/bin/env python
"""
Sample script for performing common tasks.

If you use this script to run jobs on a cluster don't forget to change the `queue`
argument in `write_jobfile`. Of course, you may have to modify the function
itself depending on your setup.

"""
from __future__ import absolute_import

import os
import subprocess
import numpy as np
import tools


#=========================================================================================
# Submit a job
#=========================================================================================

sbatchpath = './sbatch/'
scratchpath = '/scratch/gy441/multitask/'


# Current continual learning version
if True:
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


if True:
    # nunits = [128, 256, 512]
    nunits = [256]
    s_list = [0]
    for seed in range(0,20):
        for nunit in nunits:
            jobname = 'job_{:d}_{:d}'.format(seed, nunit)
            train_arg = 'nunit={:d}'.format(nunit)
            train_arg += ',seed={:d}'.format(seed)
            # train_arg += ',incomplete_set=True'

            cmd     = r'''python -c "import paper as p;p.main_train('''+train_arg+''')"'''

            jobfile = tools.write_jobfile(cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
            subprocess.call(['sbatch', jobfile])


# Grid search
if False:
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
