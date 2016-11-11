"""
Find fixed points of a RNN
"""

from __future__ import division

import os
import numpy as np
import pickle
import time
import copy
from collections import OrderedDict
import scipy.stats as stats
from scipy.optimize import curve_fit, minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work
from task import *
from run import Run
from network import get_perf

save_addon = 'tf_latest_300'
rule = CHOICEATTEND_MOD1

with Run(save_addon) as R:
    w_rec = R.w_rec
    w_in  = R.w_in
    b_rec = R.b_rec
    config = R.config

N_RING = config['N_RING']
_, nh, _ = config['shape']
# Add the constant rule input to the baseline
b_rec = b_rec + w_in[:, 2*N_RING+1+rule]

f = lambda x : np.log(1+np.exp(x)) # nonlinearity
dxdt = lambda x : -x + f(np.dot(w_rec,x)+b_rec) # dx/dt
g = lambda x : np.mean(dxdt(x)**2)

x = np.zeros(nh)

print(g(x))
