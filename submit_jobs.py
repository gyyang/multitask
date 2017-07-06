#! /usr/bin/env python
"""
Sample script for performing common tasks.

If you use this script to run jobs on a cluster don't forget to change the `queue`
argument in `write_jobfile`. Of course, you may have to modify the function
itself depending on your setup.

"""
from __future__ import absolute_import

import os
import errno
import subprocess
import numpy as np

def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                  nodes=1, ppn=1, gpus=0, mem=16, nhours=24):
    """
    Create a job file.

    Parameters
    ----------

    cmd : str
          Command to execute.

    jobname : str
              Name of the job.

    sbatchpath : str
              Directory to store SBATCH file in.

    scratchpath : str
                  Directory to store output files in.

    nodes : int, optional
            Number of compute nodes.

    ppn : int, optional
          Number of cores per node.

    gpus : int, optional
           Number of GPU cores.

    mem : int, optional
          Amount, in GB, of memory.

    ndays : int, optional
            Running time, in days.

    queue : str, optional
            Queue name.

    Returns
    -------

    jobfile : str
              Path to the job file.

    """

    mkdir_p(sbatchpath)
    jobfile = os.path.join(sbatchpath, jobname + '.s')
    logname = os.path.join('log', jobname)

    if gpus == 0:
        with open(jobfile, 'w') as f:
            f.write(
                '#! /bin/bash\n'
                + '\n'
                + '#SBATCH --nodes={}\n'.format(nodes)
                + '#SBATCH --ntasks=1\n'
                + '#SBATCH --cpus-per-task={}\n'.format(ppn)
                + '#SBATCH --mem={}GB\n'.format(mem)
                + '#SBATCH --time={}:00:00\n'.format(nhours)
                + '#SBATCH --job-name={}\n'.format(jobname[0:16])
                + '#SBATCH --output={}log/{}.o\n'.format(scratchpath, jobname[0:16])
                + '\n'
                + 'cd {}\n'.format(scratchpath)
                + 'pwd > {}.log\n'.format(logname)
                + 'date >> {}.log\n'.format(logname)
                + 'which python >> {}.log\n'.format(logname)
                + '{} >> {}.log 2>&1\n'.format(cmd, logname)
                + '\n'
                + 'exit 0;\n'
                )
    else:
        with open(jobfile, 'w') as f:
            f.write(
                '#! /bin/bash\n'
                + '\n'
                + '#SBATCH --nodes={}\n'.format(nodes)
                + '#SBATCH --ntasks=1\n'
                + '#SBATCH --cpus-per-task={}\n'.format(ppn)
                + '#SBATCH --mem={}GB\n'.format(mem)
                + '#SBATCH --partition=gpu\n'
                + '#SBATCH --gres=gpu:1\n'
                + '#SBATCH --time={}:00:00\n'.format(nhours)
                + '#SBATCH --job-name={}\n'.format(jobname[0:24])
                + '#SBATCH --output={}log/{}.o\n'.format(scratchpath, jobname[0:24])
                + '\n'
                + 'cd {}\n'.format(scratchpath)
                + 'pwd > {}.log\n'.format(logname)
                + 'date >> {}.log\n'.format(logname)
                + 'which python >> {}.log\n'.format(logname)
                + '{} >> {}.log 2>&1\n'.format(cmd, logname)
                + '\n'
                + 'exit 0;\n'
                )
    return jobfile

#=========================================================================================
# Submit a job
#=========================================================================================

if True:
    # nunits = range(20,301,20)[::-1]
    # nunits = [128, 256]
    nunits = [128, 256, 512]
    s_list = [0]
    for seed in range(20):
        for nunit in nunits:
            for s in s_list:
                jobname = 'job_{:d}_{:d}'.format(seed, nunit)
                train_arg = 'HDIM={:d}, s={:d}, seed={:d}'.format(nunit, s, seed)
                train_arg+= r", save_addon='"+'{:d}_{:d}paper'.format(seed, nunit)+r"'"
                cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

                sbatchpath = './sbatch/'
                scratchpath = '/scratch/gy441/multitask/'

                jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                                 ppn=1, gpus=0)
                subprocess.call(['sbatch', jobfile])

# Current continual learning version
if False:
    s = 1
    n_unit = 256
    for seed in range(10):
        for i_e, early_stop in enumerate([1.05]):
            for i_c, c_intsyn in enumerate([0, 0.1, 1.0]):
                if c_intsyn>0:
                    ksi_intsyns = [0.01, 0.1, 1.0]
                else:
                    ksi_intsyns = [0.001]
                for i_ksi, ksi_intsyn in enumerate(ksi_intsyns):
                    jobname = 'j{:d}_{:d}_{:d}_{:d}'.format(seed, i_e, i_ksi, i_c)
                    train_arg = 'HDIM={:d}, c_intsyn={:0.6f}, ksi_intsyn={:0.6f}, s={:d}, seed={:d}, early_stop={:0.4f}'.format(
                        n_unit, c_intsyn, ksi_intsyn, s, seed, early_stop)
                    train_arg+= r", save_addon='"+'{:d}_{:d}_{:d}_{:d}intsynrelu'.format(seed, i_e, i_ksi, i_c)+r"'"
                    cmd     = r'''python -c "import train as t;t.train_cont('''+train_arg+''')"'''

                    sbatchpath = './sbatch/'
                    scratchpath = '/scratch/gy441/multitask/'

                    jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                                     ppn=1, gpus=0)
                    subprocess.call(['sbatch', jobfile])

# Use GPUs
if False:
    nunits = [256]
    s_list = [0]
    for seed in range(2):
        for nunit in nunits:
            for s in s_list:
                jobname = 'jobgpu_{:d}_{:d}'.format(seed, nunit)
                train_arg = 'HDIM={:d}, s={:d}, seed={:d}'.format(nunit, s, seed)
                train_arg+= r", save_addon='"+'{:d}_{:d}gpu'.format(seed, nunit)+r"'"
                cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

                sbatchpath = './sbatch/'
                scratchpath = '/scratch/gy441/multitask/'

                jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                                 ppn=1, gpus=1, mem=12, nhours=6)
                subprocess.call(['sbatch', jobfile])

# Vary learning rate
if False:
    lrs = np.logspace(-2,-4,30)
    for i, lr in enumerate(lrs):
        jobname = 'job_lr{:d}'.format(i)
        train_arg = 'learning_rate={:0.7f}'.format(lr)
        train_arg+= r",save_addon='"+'lr{:d}'.format(i)+r"'"
        cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

        sbatchpath = './sbatch/'
        scratchpath = '/scratch/gy441/multitask/'

        jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                         ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])


# Vary beta_anchor
if False:
    beta_anchors = np.concatenate(([0], np.logspace(-3, -1, 30)))
    s_list = [17, 21]
    for i_beta, beta_anchor in enumerate(beta_anchors):
        for s in s_list:
            jobname = 'job_anchor{:d}_{:d}'.format(i_beta, s)
            train_arg = 's={:d}, beta_anchor={:0.6f}'.format(s, beta_anchor)
            train_arg+= r", save_addon='"+'{:d}wednight'.format(i_beta)+r"'"
            cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

            sbatchpath = './sbatch/'
            scratchpath = '/scratch/gy441/multitask/'

            jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                             ppn=1, gpus=0)
            subprocess.call(['sbatch', jobfile])

# Vary seed & beta_anchor
if False:
    seeds = range(20)
    beta_anchors = [0, 0.02]
    s = 21
    for seed in seeds:
        for i_beta, beta_anchor in enumerate(beta_anchors):
            jobname = 'job_anchor{:d}_{:d}_{:d}'.format(i_beta, s, seed)
            train_arg = 's={:d}, beta_anchor={:0.6f}, seed={:d}'.format(s, beta_anchor, seed)
            train_arg+= r", save_addon='"+'{:d}_anchor{:d}'.format(seed, i_beta)+r"'"
            cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

            sbatchpath = './sbatch/'
            scratchpath = '/scratch/gy441/multitask/'

            jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                             ppn=1, gpus=0)
            subprocess.call(['sbatch', jobfile])

# Vary l_ewc
if False:
    l_ewcs = np.concatenate(([0], np.logspace(13, 14.5, 50)))
    for i_l_ewc, l_ewc in enumerate(l_ewcs):
        jobname = 'job_ewc{:d}'.format(i_l_ewc)
        train_arg = 'l_ewc={:0.6f}'.format(l_ewc)
        train_arg+= r", save_addon='"+'{:d}friday'.format(i_l_ewc)+r"'"
        cmd     = r'''python -c "import train as t;t.train_cont('''+train_arg+''')"'''

        sbatchpath = './sbatch/'
        scratchpath = '/scratch/gy441/multitask/'

        jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                         ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

# Vary c_intsyn
if False:
    s = 1
    early_stop = 0.85
    # n_unit = 300
    for seed in range(5):
        for n_unit in [300, 400]:
            for i_c, c_intsyn in enumerate([0, 0.1, 1.0]):
                if c_intsyn>0:
                    ksi_intsyns = [0.01, 0.1, 1.0]
                else:
                    ksi_intsyns = [0.001]
                for i_ksi, ksi_intsyn in enumerate(ksi_intsyns):
                    jobname = 'j{:d}_{:d}_{:d}'.format(seed, n_unit, i_ksi, i_c)
                    train_arg = 'HDIM={:d}, c_intsyn={:0.6f}, ksi_intsyn={:0.6f}, s={:d}, seed={:d}, early_stop={:0.4f}'.format(n_unit, c_intsyn, ksi_intsyn, s, seed, early_stop)
                    train_arg+= r", save_addon='"+'{:d}_{:d}_{:d}_{:d}intsynthu'.format(seed, n_unit, i_ksi, i_c)+r"'"
                    cmd     = r'''python -c "import train as t;t.train_cont('''+train_arg+''')"'''

                    sbatchpath = './sbatch/'
                    scratchpath = '/scratch/gy441/multitask/'

                    jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                                     ppn=1, gpus=0)
                    subprocess.call(['sbatch', jobfile])


# Train OIC & DMC
if False:
    nunits = range(20,301,10)[::-1]
    s = 12
    training_iters=500000
    for nunit in nunits:
        jobname = 'job_{:d}_{:d}'.format(s, nunit)
        train_arg = 'HDIM={:d}, s={:d}, training_iters={:d}'.format(nunit, s, training_iters)
        train_arg+= r", save_addon='"+'{:d}'.format(nunit)+r"'"
        cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

        sbatchpath = './sbatch/'
        scratchpath = '/scratch/gy441/multitask/'

        jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                         ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])


# Dimensionality analysis
if False:
    for analyze_data in ['True', 'False']:
        jobname = 'job_dim_data'+analyze_data
        train_arg = 'analyze_data='+analyze_data
        cmd     = r'''python -c "import dimensionality as d;d.get_dimension_varyusedunit('''+train_arg+''')"'''

        sbatchpath = './sbatch/'
        scratchpath = '/scratch/gy441/multitask/'

        jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                                         ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])
