"""
Analysis of general properties of tasks
"""

from __future__ import division

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.client.session import Session

from task import *
from network import LeakyRNNCell, popvec

fast_eval = True

class Run(Session):
    '''
    Run the trained network
    '''
    def __init__(self, save_addon, sigma_rec=None, lesion_units=None, fast_eval=False):
        '''
        save_addon: add on for loading and saving
        inh_id    : Ids of units to inhibit inputs or outputs
        inh_output: If True, the inhibit the output of neurons, otherwise inhibit inputs to neurons
        '''

        # Reset tensorflow graphs
        tf.reset_default_graph() # must be in the beginning

        # Initialize tensorflow session
        super(Run, self).__init__() # initialize Session()

        print('Analyzing network ' + save_addon)

        # Load config
        with open(os.path.join('data','config'+save_addon+'.pkl'),'rb') as f:
            config = pickle.load(f)


        # Evaluate at a bigger time step
        if fast_eval:
            config['dt']    = 10
            print('Currently using fast evaluation')
        else:
            config['dt']    = 1

        config['alpha'] = config['dt']/TAU

        if sigma_rec is not None:
            print('Overwrite original sigma_rec with {:0.3f}'.format(sigma_rec))
            config['sigma_rec'] = sigma_rec

        # Network Parameters
        n_input, n_hidden, n_output = config['shape']

        # tf Graph input
        x = tf.placeholder("float", [None, None, n_input]) # time * batch * n_input

        # Define weights
        w_out = tf.Variable(tf.random_normal([n_hidden, n_output]))
        b_out = tf.Variable(tf.random_normal([n_output]))

        # Initial state
        h_init = tf.Variable(tf.zeros([1, n_hidden]))
        h_init_bc = tf.tile(h_init, [tf.shape(x)[1], 1]) # broadcast to size (batch, n_h)

        # Recurrent activity
        cell = LeakyRNNCell(n_hidden, config['alpha'], sigma_rec=config['sigma_rec'])
        h, states = rnn.dynamic_rnn(cell, x, initial_state=tf.abs(h_init_bc),
                                    dtype=tf.float32, time_major=True) # time_major is important


        # Output
        y_hat = tf.sigmoid(tf.matmul(tf.reshape(h, (-1, n_hidden)), w_out) + b_out)

        init = tf.initialize_all_variables()
        self.run(init)
        # Restore variable
        saver = tf.train.Saver()
        saver.restore(self, os.path.join('data', config['save_addon']+'.ckpt'))

        if lesion_units is not None:
            try:
                _ = iter(lesion_units)
                lesion_units = np.array(lesion_units)
            except TypeError:
                lesion_units = np.array([lesion_units])
            print('Lesioning Units:')
            print(lesion_units)

            # Temporary solution before better ways to get recurrent connections
            w_rec = self.run(tf.trainable_variables()[3])
            assert w_rec.shape==(n_input+n_hidden, n_hidden) # has to be the recurrent connection
            w_rec[n_input+lesion_units, :] = 0 # Set output projections from these units to zero
            lesion_w_rec = tf.trainable_variables()[3].assign(w_rec)
            self.run(lesion_w_rec)

            w_out = self.run(tf.trainable_variables()[0])
            assert w_out.shape==(n_hidden, n_output) # has to be the recurrent connection
            w_out[lesion_units, :] = 0 # Set output projections from these units to zero
            lesion_w_out = tf.trainable_variables()[0].assign(w_out)
            self.run(lesion_w_out)

        self.f_h = lambda x0 : self.run(h, feed_dict={x : x0})
        self.f_y = lambda h0 : self.run(y_hat, feed_dict={h : h0}).reshape(
            (h0.shape[0],h0.shape[1],n_output))
        self.f_y_from_x = lambda x0 : self.f_y(self.f_h(x0))
        self.f_y_loc = lambda y0 : popvec(y0[...,1:])
        self.f_y_loc_from_x = lambda x0 : self.f_y_loc(self.f_y(self.f_h(x0)))
        self.f_cost  = lambda y0, y_hat0, c_mask0: np.mean(np.sum((c_mask0*(y_hat0-y0))**2),axis=0)

        # Notice this weight is originally used as r*W, so transpose them
        self.params = self.run(tf.trainable_variables())
        self.w_out = self.params[0].T
        self.b_out = self.params[1]
        self.h_init= abs(self.params[2][0,:])
        self.w_rec = self.params[3][-n_hidden:, :].T
        self.w_in  = self.params[3][:n_input, :].T
        self.b_rec = self.params[4]

        self.config = config

        self.test_ran = False

def sample_plot(save_addon, rule, save=False, plot_ylabel=False):

    fs = 7

    with Run(save_addon) as R:
        config = R.config
        task = generate_onebatch(rule=rule, config=config, mode='sample', t_tot=2000)
        x_sample = task.x
        h_sample = R.f_h(x_sample)
        y_sample = R.f_y(h_sample)

        params = R.params
        w_rec = R.w_rec
        w_in  = R.w_in

    assert config['num_ring'] == 2

    y_sample = y_sample.reshape((-1,1,config['shape'][2]))

    y = task.y

    N_RING = config['N_RING']

    fig = plt.figure(figsize=(1.3,2))
    ylabels = ['fix. in', 'stim. mod1', 'stim. mod2','fix. out', 'out']
    heights = np.array([0.03,0.2,0.2,0.03,0.2])+0.01
    for i in range(5):
        ax = fig.add_axes([0.15,sum(heights[i+1:]+0.02)+0.1,0.8,heights[i]])
        cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
        plt.xticks([])
        ax.tick_params(axis='both', which='major', labelsize=fs, width=0.5, length=2, pad=3)

        if plot_ylabel:
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        else:
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('none')

        if i == 0:
            plt.plot(x_sample[:,0,0], color=sns.xkcd_palette(['blue'])[0])
            if plot_ylabel:
                plt.yticks([0,1],['',''],rotation='vertical')
            plt.ylim([-0.1,1.5])
            plt.title(rule_name[rule],fontsize=fs)
        elif i == 1:
            plt.imshow(x_sample[:,0,1:1+N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            if plot_ylabel:
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
        elif i == 2:
            plt.imshow(x_sample[:,0,1+N_RING:1+2*N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            # plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
            if plot_ylabel:
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
        elif i == 3:
            plt.plot(y[:,0,0],color=sns.xkcd_palette(['green'])[0])
            plt.plot(y_sample[:,0,0],color=sns.xkcd_palette(['blue'])[0])
            if plot_ylabel:
                plt.yticks([0.05,0.8],['',''],rotation='vertical')
            plt.ylim([-0.1,1.1])
        elif i == 4:
            plt.imshow(y_sample[:,0,1:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            # plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
            if plot_ylabel:
                plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
            plt.xticks([0,2000], ['0', '2'])
            plt.xlabel('Time (s)',fontsize=fs, labelpad=-3)
            ax.spines["bottom"].set_visible(True)

        if plot_ylabel:
           plt.ylabel(ylabels[i],fontsize=fs)
        else:
            plt.yticks([])
        ax.get_yaxis().set_label_coords(-0.12,0.5)

    if save:
        plt.savefig('figure/sample_'+rule_name[rule].replace(' ','')+'.pdf', transparent=True)
    plt.show()


    _ = plt.plot(h_sample[:,0,:20])
    plt.show()

def schematic_plot(save_addon):
    fontsize = 6

    rule = CHOICE_MOD1

    with Run(save_addon) as R:
        config = R.config
        task = generate_onebatch(rule=rule, config=config, mode='sample', t_tot=1000)
        x_sample = task.x
        h_sample = R.f_h(x_sample)
        y_sample = R.f_y(h_sample)


    N_RING = config['N_RING']

    # Plot Stimulus
    fig = plt.figure(figsize=(1.0,1.2))
    heights = np.array([0.06,0.25,0.25])
    for i in range(3):
        ax = fig.add_axes([0.2,sum(heights[i+1:]+0.1)+0.05,0.7,heights[i]])
        cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
        plt.xticks([])

        # Fixed style for these plots
        ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if i == 0:
            plt.plot(x_sample[:,0,0], color=sns.xkcd_palette(['blue'])[0])
            plt.yticks([0,1],['',''],rotation='vertical')
            plt.ylim([-0.1,1.5])
            plt.title('Fixation input', fontsize=fontsize, y=0.9)
        elif i == 1:
            plt.imshow(x_sample[:,0,1:1+N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$','',r'360$\degree$'],rotation='vertical')
            plt.title('Stimulus Mod 1', fontsize=fontsize, y=0.9)
        elif i == 2:
            plt.imshow(x_sample[:,0,1+N_RING:1+2*N_RING].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            plt.yticks([0,(N_RING-1)/2,N_RING-1],['','',''],rotation='vertical')
            plt.title('Stimulus Mod 2', fontsize=fontsize, y=0.9)
        ax.get_yaxis().set_label_coords(-0.12,0.5)
    plt.savefig('figure/schematic_input.pdf',transparent=True)
    plt.show()

    # Plot Rule Inputs
    fig = plt.figure(figsize=(1.0, 0.5))
    ax = fig.add_axes([0.2,0.3,0.7,0.45])
    cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
    X = x_sample[:,0,1+2*N_RING:]
    plt.imshow(X.T, aspect='auto', vmin=0, vmax=1, cmap=cmap, interpolation='none',origin='lower')

    plt.xticks([0, 1000])
    ax.set_xlabel('Time (ms)', fontsize=fontsize, labelpad=-5)

    # Fixed style for these plots
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.yticks([0,X.shape[-1]-1],['1',str(X.shape[-1])],rotation='vertical')
    plt.title('Rule inputs', fontsize=fontsize, y=0.9)
    ax.get_yaxis().set_label_coords(-0.12,0.5)

    plt.savefig('figure/schematic_rule.pdf',transparent=True)
    plt.show()


    # Plot Units
    fig = plt.figure(figsize=(1.0, 0.8))
    ax = fig.add_axes([0.2,0.1,0.7,0.75])
    cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
    plt.xticks([])
    # Fixed style for these plots
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.imshow(h_sample[:,0,:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
    plt.yticks([0,config['HDIM']-1],['1',str(config['HDIM'])],rotation='vertical')
    plt.title('Recurrent units', fontsize=fontsize, y=0.95)
    ax.get_yaxis().set_label_coords(-0.12,0.5)
    plt.savefig('figure/schematic_units.pdf',transparent=True)
    plt.show()


    # Plot Outputs
    fig = plt.figure(figsize=(1.0,0.8))
    heights = np.array([0.1,0.45])+0.01
    for i in range(2):
        ax = fig.add_axes([0.2,sum(heights[i+1:]+0.15)+0.1,0.7,heights[i]])
        cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
        plt.xticks([])

        # Fixed style for these plots
        ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if i == 0:
            # plt.plot(task.y[:,0,0],color=sns.xkcd_palette(['green'])[0])
            plt.plot(y_sample[:,0,0],color=sns.xkcd_palette(['blue'])[0])
            plt.yticks([0.05,0.8],['',''],rotation='vertical')
            plt.ylim([-0.1,1.1])
            plt.title('Fixation output', fontsize=fontsize, y=0.9)

        elif i == 1:
            plt.imshow(y_sample[:,0,1:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            plt.yticks([0,(N_RING-1)/2,N_RING-1],[r'0$\degree$','',r'360$\degree$'],rotation='vertical')
            plt.xticks([])
            plt.title('Output', fontsize=fontsize, y=0.9)

        ax.get_yaxis().set_label_coords(-0.12,0.5)

    plt.savefig('figure/schematic_outputs.pdf',transparent=True)
    plt.show()

def plot_singleneuron_intime(save_addon, neurons, rules,
                             epoch=None, save=False, ylabel_firstonly=True):
    '''

    :param save_addon:
    :param neurons: indices of neurons to plot
    :param rules: rules to plot
    :param epoch: epoch to plot
    :param save: save figure?
    :param ylabel_firstonly: if True, only plot ylabel for the first rule in rules
    :return:
    '''
    try:
        _ = iter(rules)
    except TypeError:
        rules = [rules]

    try:
        _ = iter(neurons)
    except TypeError:
        neurons = [neurons]

    h_tests = dict()
    with Run(save_addon, sigma_rec=0.0, fast_eval=fast_eval) as R:
        config = R.config
        t_start = int(500/config['dt'])

        for rule in rules:
            task = generate_onebatch(rule=rule, config=config, mode='test')
            h_tests[rule] = R.f_h(task.x) # (Time, Batch, Units)

    for neuron in neurons:
        h_max = np.max([h_tests[r][t_start:,:,neuron].max() for r in rules])
        for j, rule in enumerate(rules):
            fs = 6
            fig = plt.figure(figsize=(1.0,0.8))
            ax = fig.add_axes([0.35,0.25,0.55,0.55])
            ax.set_color_cycle(sns.color_palette("husl", h_tests[rule].shape[1]))
            _ = ax.plot(np.arange(h_tests[rule][t_start:].shape[0])*config['dt']/1000,
                        h_tests[rule][t_start:,:,neuron], lw=0.5)

            if epoch is not None:
                e0, e1 = task.epochs[epoch]
                e0 = e0 if e0 is not None else 0
                e1 = e1 if e1 is not None else h_tests[rule].shape[0]
                ax.plot([e0, e1], [h_max*1.15]*2,
                        color='black',linewidth=1.5)
                save_name = 'figure/trace_'+rule_name[rule]+epoch+save_addon+'.pdf'
            else:
                save_name = 'figure/trace_unit'+str(neuron)+rule_name[rule]+save_addon+'.pdf'

            plt.ylim(np.array([-0.1, 1.2])*h_max)
            plt.xticks([0,2])
            plt.xlabel('Time (s)', fontsize=fs, labelpad=-5)
            plt.locator_params(axis='y', nbins=4)
            if j>0 and ylabel_firstonly:
                ax.set_yticklabels([])
            else:
                plt.ylabel('activitity (a.u.)', fontsize=fs)
            plt.title('Unit {:d} '.format(neuron) + rule_name[rule], fontsize=5)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if save:
                plt.savefig(save_name, transparent=True)
            plt.show()


if __name__ == "__main__":
    # schematic_plot(save_addon='allrule_weaknoise_400')
    rules = [GO, INHGO, DELAYGO,\
        CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
        CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
        REMAP, INHREMAP, DELAYREMAP,\
        DMSGO, DMSNOGO, DMCGO, DMCNOGO]
    for rule in rules:
        pass
        # sample_plot(save_addon='allrule_weaknoise_400', rule=rule, save=True)

    # plot_singleneuron_intime('allrule_weaknoise_360', [80], [CHOICEATTEND_MOD1, CHOICEATTEND_MOD2],
    #                          epoch=None, save=False, ylabel_firstonly=True)
    pass

