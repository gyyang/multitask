"""
2016/06/03 Restart, with Blocks

Main training loop and network structure
"""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import theano
import theano.tensor as T

from fuel.streams import DataStream
from fuel.datasets import IterableDataset
from blocks.model import Model
from blocks.algorithms import GradientDescent, Adam, CompositeRule, StepClipping
from blocks.graph import ComputationGraph
from blocks.extensions import FinishAfter, Printing, ProgressBar, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT, BIAS, INITIAL_STATE
from blocks.main_loop import MainLoop

from task import *
from network import MyNetwork


def train(HDIM=200, N_RING=16, save_main_loop=False):
    config = {'h_type'      : 'leaky_rec_ei',
              'alpha'       : 0.2, # \Delta t/tau
              'dt'          : 0.2*TAU,
              'HDIM'        : HDIM,
              'N_RING'      : N_RING,
              'shape'       : (1+2*N_RING+N_RULE, HDIM, N_RING+1),
              'save_addon'  : 'latestei_'+str(HDIM)+'_'+str(N_RING)}

    ###############################   Setup  ##################################
    # Training parameters
    batch_size  = 10 # Number of examples in a single batch.
    num_batches = 2000 # Number of batches in the training dataset.
    num_epochs  = 50 # Number of epochs to do.

    # Rules to train and the proportion of them
    rules = [FIXATION, GO, INHGO, DELAYGO, CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,
             CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,
             REMAP, INHREMAP, DELAYREMAP, DELAYMATCHGO, DELAYMATCHNOGO, DMCGO]
    rule_set = rules[:]
    rules += [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2] * 4
    rules += [DELAYMATCHGO, DELAYMATCHNOGO] * 4
    rules += [DMCGO] * 4

    # rules = [DMCGO]
    # rule_set = rules[:]

    # Generating the dataset
    dataset_train = IterableDataset(generate_data(config, batch_size, num_batches, rules))
    dataset_test = IterableDataset(generate_data(config, batch_size, 1000, rules))

    stream_train = DataStream(dataset=dataset_train)
    stream_test = DataStream(dataset=dataset_test)

    # Generating the network
    x = T.tensor3('x')
    y_hat = T.tensor3('y_hat')
    y_hat_loc = T.vector('y_hat_loc') # Vector of size batchsize
    c_mask = T.tensor3('c_mask')
    mynet = MyNetwork(config)
    cost, performance = mynet.cost(x, y_hat, y_hat_loc, c_mask)
    cost.name = 'cost'
    performance.name = 'performance'
    model = Model(cost)

    mynet.initialize()

    # Setting up the learning
    step_rule = CompositeRule([StepClipping(1.),
                               Adam(learning_rate=0.002, beta1=0.1,
                                    beta2=0.001, epsilon=1e-8)])
    
    cg = ComputationGraph(cost)
    trained_parameters = VariableFilter(roles=[WEIGHT, BIAS])(cg.parameters) # don't train initial state
    # trained_parameters = VariableFilter(roles=[WEIGHT, BIAS, INITIAL_STATE])(cg.parameters) # train initial state
    algorithm = GradientDescent(cost=cost,
                                parameters=trained_parameters,
                                step_rule=step_rule, on_unused_sources='warn')

    # Monitor the progress
    test_monitor = DataStreamMonitoring(variables=[cost, performance],
                                        data_stream=stream_test, prefix="test")
    train_monitor = TrainingDataMonitoring(variables=[cost, performance], prefix="train",
                                           after_epoch=True)

    extensions = [test_monitor, train_monitor, FinishAfter(after_n_epochs=num_epochs),
                  Printing(), ProgressBar(), Timing(),
                  Checkpoint('data/task_multi'+config['save_addon']+'.tar',
                             every_n_epochs=10, save_separately=["model", "log"],
                             save_main_loop=save_main_loop)]

    # Rule specific monitors
    dataset = dict()
    stream = dict()
    monitor = dict()
    for rule in rule_set:
        dataset[rule] = IterableDataset(generate_data(config, batch_size, 200, [rule]))
        stream[rule]  = DataStream(dataset=dataset[rule])
        monitor[rule] = DataStreamMonitoring(variables=[cost, performance],
                                             data_stream=stream[rule],
                                             prefix=rule_name[rule])
    extensions += monitor.values()

    ###############################   Training  ###############################

    main_loop = MainLoop(model=model, algorithm=algorithm, data_stream=stream_train,
                         extensions=extensions)
    main_loop.run()

    ########################## Get Training Progress #########################
    log = main_loop.log
    batch_plot = list() # batch
    traintime_plot = list() # training time
    cost_plot = list() # cost
    rule_cost_plot = dict() # cost of individual rule
    rule_performance_plot = dict()
    for rule in rule_set:
        rule_cost_plot[rule] = []
        rule_performance_plot[rule] = []
    for i in log:
        if (len(log[i].keys())>0) and ('test_cost' in log[i].keys()):
            # prefix test + monitored variable name (cost.name)
            if i == 0:
                traintime_plot.append(0)
            else:
                traintime_plot.append(log[i]['time_train_total'])
            cost_plot.append(log[i]['test_cost'])
            batch_plot.append(i)
            for rule in rule_set:
                rule_cost_plot[rule].append(log[i][rule_name[rule]+'_cost'])
                rule_performance_plot[rule].append(log[i][rule_name[rule]+'_performance'])

    ########################## Save Configuration #############################

    config['rule_cost_plot'] = rule_cost_plot
    config['rule_performance_plot'] = rule_performance_plot
    config['batch_plot'] = batch_plot
    config['traintime_plot'] = traintime_plot
    config['batch_size'] = batch_size
    with open('data/config'+config['save_addon']+'.pkl','wb') as f:
        pickle.dump(config,f)

if __name__ == '__main__':
    # Debug now
    train(HDIM=200, N_RING=16)
    pass