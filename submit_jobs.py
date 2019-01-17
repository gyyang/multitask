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
scratchpath = '/scratch/gy441/multitask/'


def write_jobfile(cmd, jobname, sbatchpath, scratchpath,
                  nodes=1, ppn=1, gpus=0, mem=16, nhours=18):
    """
    Create a job file.

    Args:
        cmd : str, Command to execute.
        jobname : str, Name of the job.
        sbatchpath : str, Directory to store SBATCH file in.
        scratchpath : str, Directory to store output files in.
        nodes : int, optional, Number of compute nodes.
        ppn : int, optional, Number of cores per node.
        gpus : int, optional, Number of GPU cores.
        mem : int, optional, Amount, in GB, of memory.
        ndays : int, optional, Running time, in days.
        queue : str, optional, Queue name.

    Returns:
        jobfile : str, Path to the job file.
    """

    tools.mkdir_p(sbatchpath)
    jobfile = os.path.join(sbatchpath, jobname + '.s')
    logname = os.path.join('log', jobname)

    if gpus == 0:
        with open(jobfile, 'w') as f:
            f.write(
                '#! /bin/bash\n'
                + '\n'
                + '#SBATCH --nodes={}\n'.format(nodes)
                #+ '#SBATCH --ntasks=1\n'
                + '#SBATCH --ntasks-per-node=1\n'
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
                + '#SBATCH --ntasks-per-node=1\n'
                + '#SBATCH --cpus-per-task={}\n'.format(ppn)
                + '#SBATCH --mem={}GB\n'.format(mem)
                + '#SBATCH --partition=xwang_gpu\n'
                + '#SBATCH --gres=gpu:1\n'
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
    return jobfile


if args.run == 'all':
    for seed in range(0, 40):
        jobname = 'train_all_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_all('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'analysis_all':
    for seed in range(0, 40):
        jobname = 'analysis_all_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_all_analysis('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'tanhgru':
    for seed in range(0, 20):
        jobname = 'tanhgru_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_all_tanhgru('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mixrule':
    for seed in range(0, 20):
        jobname = 'mr_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_all_mixrule('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mixrule_softplus':
    for seed in range(0, 20):
        jobname = 'mrsp_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_all_mixrule_softplus('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'all_varyhp':
    for i in range(0, 20):
        jobname = 'train_varyhp_{:d}'.format(i)
        train_arg = '{:d}'.format(i)
        cmd = r'''python -c "import experiment as e;e.train_vary_hp('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'seq':
    for i in range(0, 40):
        jobname = 'seq_{:d}'.format(i)
        train_arg = '{:d}'.format(i)
        cmd = r'''python -c "import experiment as e;e.train_seq('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'seq_varyhp':
    for i in range(0, 72):
        jobname = 'seq_varyhp_{:d}'.format(i)
        train_arg = '{:d}'.format(i)
        cmd = r'''python -c "import experiment as e;e.train_vary_hp_seq('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mante':
    for seed in range(0, 20):
        jobname = 'train_mante_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.train_mante('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mante_tanh':
    for seed in range(0, 50):
        jobname = 'mantetanh_{:d}'.format(seed)
        train_arg = 'seed={:d}'.format(seed)
        cmd = r'''python -c "import experiment as e;e.mante_tanh('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mante_vary_l2init':
    for i in range(0, 300):
        jobname = 'mante_vary_l2init_{:d}'.format(i)
        train_arg = '{:d}'.format(i)
        cmd = r'''python -c "import experiment as e;e.vary_l2_init_mante('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mante_vary_l2weight':
    for i in range(0, 300):
        jobname = 'mante_vary_l2weight_{:d}'.format(i)
        train_arg = '{:d}'.format(i)
        cmd = r'''python -c "import experiment as e;e.vary_l2_weight_mante('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'mante_vary_pweighttrain':
    for i in range(200, 260):
        jobname = 'mante_vary_pweighttrain_{:d}'.format(i)
        train_arg = '{:d}'.format(i)
        cmd = r'''python -c "import experiment as e;e.vary_p_weight_train_mante('''+\
              train_arg+''')"'''

        jobfile = write_jobfile(
            cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
        subprocess.call(['sbatch', jobfile])

elif args.run == 'pretrain':
    for seed in range(0, 20):
        for setup in range(2):
            jobname = 'pt_{:d}_{:d}'.format(setup, seed)
            train_arg = 'setup={:d},seed={:d}'.format(setup, seed)
            cmd = r'''python -c "import experiment as e;e.pretrain('''+\
                  train_arg+''')"'''

            jobfile = write_jobfile(
                cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
            subprocess.call(['sbatch', jobfile])

elif args.run == 'posttrain':
    for seed in range(0, 20):
        for pretrain_setup in range(2):
            for posttrain_setup in range(2):
                for trainables in range(2):
                    jobname = 'pt{:d}{:d}{:d}{:d}'.format(
                        pretrain_setup, posttrain_setup, trainables, seed)
                    train_arg = '{:d}, {:d}, {:d}, {:d}'.format(
                        pretrain_setup, posttrain_setup, trainables, seed)
                    cmd = r'''python -c "import experiment as e;e.posttrain('''+\
                          train_arg+''')"'''

                    jobfile = write_jobfile(
                        cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
                    subprocess.call(['sbatch', jobfile])

# Grid search
elif args.run == 'grid':
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

                jobfile = write_jobfile(cmd, jobname, sbatchpath, scratchpath, ppn=1, gpus=0)
                subprocess.call(['sbatch', jobfile])

else:
    raise ValueError('Unknow argument run ' + str(args.run))
