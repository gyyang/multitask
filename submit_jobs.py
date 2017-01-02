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


def write_jobfile(cmd, jobname, pbspath, scratchpath,
                  nodes=1, ppn=1, gpus=0, mem=12, nhours=24):
    """
    Create a job file.

    Parameters
    ----------

    cmd : str
          Command to execute.

    jobname : str
              Name of the job.

    pbspath : str
              Directory to store PBS file in.

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
    if gpus > 0:
        gpus = ':gpus={}:titan'.format(gpus)
    else:
        gpus = ''

    if ppn > 1:
        threads = '#PBS -v OMP_NUM_THREADS={}\n'.format(ppn)
    else:
        threads = ''

    mkdir_p(pbspath)
    jobfile = os.path.join(pbspath, jobname + '.pbs')
    logname = os.path.join('log', jobname)
    with open(jobfile, 'w') as f:
        f.write(
            '#! /bin/bash\n'
            + '\n'
            + '#PBS -l nodes={}:ppn={}{}\n'.format(nodes, ppn, gpus)
            + '#PBS -l mem={}GB\n'.format(mem)
            + '#PBS -l walltime={}:00:00\n'.format(nhours)
            + '#PBS -l feature=ivybridge\n' # Have to run it on Ivy Bridge CPUs
            + '#PBS -q s48\n'
            + '#PBS -N {}\n'.format(jobname[0:16])
            + '#PBS -e {}log/${{PBS_JOBNAME}}.e${{PBS_JOBID}}\n'.format(scratchpath)
            + '#PBS -o {}log/${{PBS_JOBNAME}}.o${{PBS_JOBID}}\n'.format(scratchpath)
            + threads
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
    nunits = range(20,501,20)[::-1]
    s_list = [1, 2, 4, 5, 7, 8]
    for nunit in nunits:
        for s in s_list:
            jobname = 'job_{:d}_{:d}'.format(s, nunit)
            train_arg = 'HDIM={:d}, s={:d}'.format(nunit, s)
            train_arg+= r",save_addon='"+'{:d}'.format(nunit)+r"'"
            cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

            pbspath = './pbs/'
            scratchpath = '/scratch/gy441/multitask/'

            jobfile = write_jobfile(cmd, jobname, pbspath, scratchpath,
                                             ppn=1, gpus=0)
            subprocess.call(['qsub', jobfile])

if False:
    lrs = np.logspace(-2,-4,30)
    for i, lr in enumerate(lrs):
        jobname = 'job_lr{:d}'.format(i)
        train_arg = 'learning_rate={:0.7f}'.format(lr)
        train_arg+= r",save_addon='"+'lr{:d}'.format(i)+r"'"
        cmd     = r'''python -c "import train as t;t.train('''+train_arg+''')"'''

        pbspath = './pbs/'
        scratchpath = '/scratch/gy441/multitask/'

        jobfile = write_jobfile(cmd, jobname, pbspath, scratchpath,
                                         ppn=1, gpus=0)
        subprocess.call(['qsub', jobfile])



