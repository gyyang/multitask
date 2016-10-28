# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:43:32 2016

@author: guangyuyang
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#==============================================================================
# N_RING = 32
# 
# def get_dist(original_dist): # Get the distance in periodic boundary conditions
#     return np.minimum(abs(original_dist),360-abs(original_dist))
# 
# 
# for y_hat_loc in range(0,360,10):
# 
#     pref = np.arange(0,360,360./N_RING) # preferences
#     dist = get_dist(y_hat_loc-pref) # periodic boundary
#     dist /= 20
#     y_hat = 0.8*np.exp(-dist**2/2)
#     y_hat /= y_hat.sum()
# 
#     y_hat += np.random.randn(N_RING)*0.0
#
#     vec_cos = np.cos(pref/360*2*np.pi)
#     vec_sin = np.sin(pref/360*2*np.pi)
#     temp_cos = np.sum(y_hat*vec_cos)/y_hat.sum()
#     temp_sin = np.sum(y_hat*vec_sin)/y_hat.sum()
# 
#     temp = np.arctan2(temp_sin,temp_cos)/2/np.pi*360
#     temp = np.mod(temp, 360)
# 
#     print(y_hat_loc),
#     print(temp),
#     print(get_dist(y_hat_loc-temp))
#==============================================================================


import theano
import theano.tensor as T
import pickle
from blocks.model import Model
from blocks.serialization import load_parameters
from task import *
from network import MyNetwork

# Load config
save_addon = '_temp40_8'
with open('data/config'+save_addon+'.pkl','rb') as f:
    config = pickle.load(f)

alpha_original = config['alpha']
config['dt'] = 1.
config['alpha'] = 1./TAU # For sample running, we set the time step to be 1ms. So alpha = 1/tau
x_dim, h_dim, y_dim = config['shape']
mynet = MyNetwork(config)

x = T.tensor3('x')
y_hat = T.tensor3('y_hat')
y_hat_loc = T.matrix('y_hat_loc')
c_mask = T.tensor3('c_mask')
mynet = MyNetwork(config)
cost, performance = mynet.cost(x, y_hat, y_hat_loc, c_mask)
#cost = mynet.cost_temp(x, y_hat, c_mask)
cost.name = 'cost'
model = Model(cost)

y = mynet.get_y(x)
y_loc = mynet.popvec.apply(y)

with open('data/task_multi'+config['save_addon']+'.tar', 'rb') as f:
    model.set_parameter_values(load_parameters(f))

h = mynet.get_h(x)
f_h = theano.function([x],h)
f_y = theano.function([x],y)
f_y_loc = theano.function([x],y_loc)
f_cost = theano.function([x, y_hat, y_hat_loc, c_mask], cost, on_unused_input='warn')
f_perf = theano.function([x, y_hat, y_hat_loc, c_mask], performance, on_unused_input='warn')

# f_cost = theano.function([x, y_hat, c_mask], cost, on_unused_input='warn')

rule = FIXATION
N_RING = config['N_RING']
# tdim depends on the alpha (\Delta t/\tau)
task = generate_onebatch(rule=rule, config=config,
                         mode='test', t_tot=1000)
x_sample = task.x
y_sample = f_y(x_sample)
h_sample = f_h(x_sample)
y_loc_sample = f_y_loc(x_sample)
cost_sample = f_cost(x_sample, task.y_hat, task.y_hat_loc, task.c_mask)
perf_sample = f_perf(x_sample, task.y_hat, task.y_hat_loc, task.c_mask)


plt.plot(y_loc_sample[:,10])
plt.plot(task.y_hat_loc[:,10])
plt.show()


# pref = np.arange(0, 2*np.pi, 2*np.pi/N_RING) # preferences
# vec_cos = theano.shared(np.cos(pref))
# vec_sin = theano.shared(np.sin(pref))
#
# y_sumunit = y.sum(axis=2)
# temp_cos = T.sum(y*vec_cos, axis=2)/y_sumunit
# temp_sin = T.sum(y*vec_sin, axis=2)/y_sumunit
#
# y_loc0 = T.arctan2(temp_sin, temp_cos)
# y_loc = T.mod(y_loc0, theano.shared(2*np.pi))
#
# y_hat_loc = T.matrix('y_hat_loc')
# thres = theano.shared(0.2*np.pi)
# temp = T.any((abs(y_loc-y_hat_loc)>thres)*c_mask[:,:,0], axis=0).mean()
# f_temp = theano.function([x, y_hat_loc, c_mask],temp, on_unused_input='ignore')
#
# temp_sample = f_temp(x_sample, task.y_hat_loc, task.c_mask)
#
#
