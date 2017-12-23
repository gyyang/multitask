"""Utility functions."""

import os
import errno
import fnmatch
import pickle
import json


def valid_save_names(save_pattern):
    """Get valid save_names given a save pattern"""

    # Get all model names that match patterns (strip off .ckpt.meta at the end)
    return [f[:-10] for f in os.listdir('data/') if fnmatch.fnmatch(f, save_pattern+'.ckpt.meta')]


def load_log(save_name):
    """Load the log file of model save_name"""
    fname = os.path.join('data','log_'+save_name+'.pkl')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'rb') as f:
        log = pickle.load(f)
    return log


def save_log(log, save_name):
    """Save the log file of model save_name"""
    with open(os.path.join('data', 'log_'+save_name+'.pkl'), 'wb') as f:
        pickle.dump(log, f)


def load_hparams(save_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(save_dir, 'hparams.json')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'rb') as f:
        hparams = json.load(f)
    return hparams


def save_hparams(hparams, save_dir):
    """Save the hyper-parameter file of model save_name"""
    with open(os.path.join(save_dir, 'hparams.json'), 'wb') as f:
        json.dump(hparams, f)


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
                  nodes=1, ppn=1, gpus=0, mem=16, nhours=12):
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
                + '#SBATCH --partition=gpu\n'
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