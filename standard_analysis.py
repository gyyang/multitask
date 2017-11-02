"""Standard analyses that can be performed on any task"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from task import generate_trials, rule_name
from network import Model


def easy_activity_plot(save_name, rule):
    '''
    A simple plot of neural activity from one task
    :param save_name:
    :param rule:
    :param save:
    :return:
    '''
    model = Model(save_name)
    config = model.config

    with tf.Session() as sess:
        model.restore(sess)

        trial = generate_trials(rule=rule, config=config, mode='test')
        # All matrices have shape (n_time, n_condition, n_neuron)
        h = model.get_h(trial.x)
        y = model.get_y_from_h(h)

    # Take only the one example trial
    i_trial = 0

    for activity, title in zip([trial.x, h, y], ['input', 'recurrent', 'output']):
        plt.figure()
        plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot',
                   interpolation='none', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()


def easy_connectivity_plot(save_name):
    '''
    A simple plot of network connectivity
    '''
    model = Model(save_name)
    with tf.Session() as sess:
        model.restore(sess)
        # get all connection weights and biases as tensorflow variables
        var_list = model.var_list
        # evaluate the parameters after training
        params = [sess.run(var) for var in var_list]
        # get name of each variable
        names  = [var.name for var in var_list]

    # Plot weights
    for param, name in zip(params, names):
        if len(param.shape) != 2:
            continue

        plt.figure()
        plt.imshow(param.T, aspect='auto', cmap='bwr',
                   interpolation='none', origin='lower') # notice the transpose
        plt.title(name)
        plt.colorbar()
        plt.xlabel('From')
        plt.ylabel('To')
        plt.show()


def pretty_inputoutput_plot(save_name, rules, save=False, plot_ylabel=False):
    '''
    Plot the input and output activity for a sample trial from one task
    :param save_name:
    :param rule:
    :param save:
    :param plot_ylabel:
    :return:
    '''
    if isinstance(rules, str):
        rules = [rules]

    import seaborn.apionly as sns
    fs = 7

    model = Model(save_name)
    config = model.config

    with tf.Session() as sess:
        model.restore(sess)

        for rule in rules:
            trial = generate_trials(rule=rule, config=config, mode='sample')
            x, y = trial.x, trial.y
            h = model.get_h(x)
            y_hat = model.get_y_from_h(h)

            t_plot = np.arange(x.shape[0])*config['dt']/1000

            assert config['num_ring'] == 2

            n_eachring = config['n_eachring']

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
                    plt.plot(t_plot, x[:,0,0], color=sns.xkcd_palette(['blue'])[0])
                    if plot_ylabel:
                        plt.yticks([0,1],['',''],rotation='vertical')
                    plt.ylim([-0.1,1.5])
                    plt.title(rule_name[rule],fontsize=fs)
                elif i == 1:
                    plt.imshow(x[:,0,1:1+n_eachring].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                    if plot_ylabel:
                        plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                elif i == 2:
                    plt.imshow(x[:,0,1+n_eachring:1+2*n_eachring].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                    # plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                    if plot_ylabel:
                        plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                elif i == 3:
                    plt.plot(t_plot, y[:,0,0],color=sns.xkcd_palette(['green'])[0])
                    plt.plot(t_plot, y_hat[:,0,0],color=sns.xkcd_palette(['blue'])[0])
                    if plot_ylabel:
                        plt.yticks([0.05,0.8],['',''],rotation='vertical')
                    plt.ylim([-0.1,1.1])
                elif i == 4:
                    plt.imshow(y_hat[:,0,1:].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
                    # plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                    if plot_ylabel:
                        plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$',r'180$\degree$',r'360$\degree$'],rotation='vertical')
                    plt.xticks([0,y_hat.shape[0]], ['0', '2'])
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

            # plt.figure()
            # _ = plt.plot(h_sample[:,0,:20])
            # plt.show()
            #
            # plt.figure()
            # _ = plt.plot(y_sample[:,0,:])
            # plt.show()


def pretty_singleneuron_plot(save_name, rules, neurons,
                             epoch=None, save=False, ylabel_firstonly=True,
                             trace_only=False, plot_stim_avg=False):
    '''
    Plot the activity of a single neuron in time across many trials

    :param save_name:
    :param neurons: indices of neurons to plot
    :param rules: rules to plot
    :param epoch: epoch to plot
    :param save: save figure?
    :param ylabel_firstonly: if True, only plot ylabel for the first rule in rules
    :return:
    '''
    if isinstance(rules, str):
        rules = [rules]

    try:
        _ = iter(neurons)
    except TypeError:
        neurons = [neurons]

    h_tests = dict()
    model = Model(save_name)
    config = model.config
    with tf.Session() as sess:
        model.restore(sess)

        t_start = int(500/config['dt'])

        for rule in rules:
            # Generate a batch of trial from the test mode
            trial = generate_trials(rule=rule, config=config, mode='test')
            h_tests[rule] = model.get_h(trial.x) # (Time, Batch, Units)

    for neuron in neurons:
        h_max = np.max([h_tests[r][t_start:,:,neuron].max() for r in rules])
        for j, rule in enumerate(rules):
            fs = 6
            fig = plt.figure(figsize=(1.0,0.8))
            ax = fig.add_axes([0.35,0.25,0.55,0.55])
            # ax.set_color_cycle(sns.color_palette("husl", h_tests[rule].shape[1]))
            t_plot = np.arange(h_tests[rule][t_start:].shape[0])*config['dt']/1000
            _ = ax.plot(t_plot,
                        h_tests[rule][t_start:,:,neuron], lw=0.5, color='gray')

            if plot_stim_avg:
                # Plot stimulus averaged trace
                _ = ax.plot(np.arange(h_tests[rule][t_start:].shape[0])*config['dt']/1000,
                        h_tests[rule][t_start:,:,neuron].mean(axis=1), lw=1, color='black')

            if epoch is not None:
                e0, e1 = trial.epochs[epoch]
                e0 = e0 if e0 is not None else 0
                e1 = e1 if e1 is not None else h_tests[rule].shape[0]
                ax.plot([e0, e1], [h_max*1.15]*2,
                        color='black',linewidth=1.5)
                figname = 'figure/trace_'+rule_name[rule]+epoch+save_name+'.pdf'
            else:
                figname = 'figure/trace_unit'+str(neuron)+rule_name[rule]+save_name+'.pdf'

            plt.ylim(np.array([-0.1, 1.2])*h_max)
            plt.xticks([0,np.floor(np.max(t_plot)+0.01)])
            plt.xlabel('Time (s)', fontsize=fs, labelpad=-5)
            plt.locator_params(axis='y', nbins=4)
            if j>0 and ylabel_firstonly:
                ax.set_yticklabels([])
            else:
                plt.ylabel('Activitity (a.u.)', fontsize=fs, labelpad=2)
            plt.title('Unit {:d} '.format(neuron) + rule_name[rule], fontsize=5)
            ax.tick_params(axis='both', which='major', labelsize=fs)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            if trace_only:
                ax.spines["left"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.xaxis.set_ticks_position('none')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('')

            if save:
                plt.savefig(figname, transparent=True)
            plt.show()

def schematic_plot(save_name):
    import seaborn.apionly as sns
    fontsize = 6

    rule = 'dm1'

    model = Model(save_name, dt=1)
    config = model.config

    with tf.Session() as sess:
        model.restore(sess)
        trial = generate_trials(rule=rule, config=config, mode='sample', t_tot=1000)
        x_sample = trial.x
        h_sample = model.get_h(x_sample)
        y_sample = model.get_y_from_h(h_sample)


    n_eachring = config['n_eachring']
    n_input, n_hidden, n_output = config['shape']

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
            plt.imshow(x_sample[:,0,1:1+n_eachring].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$','',r'360$\degree$'],rotation='vertical')
            plt.title('Stimulus mod 1', fontsize=fontsize, y=0.9)
        elif i == 2:
            plt.imshow(x_sample[:,0,1+n_eachring:1+2*n_eachring].T, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='none',origin='lower')
            plt.yticks([0,(n_eachring-1)/2,n_eachring-1],['','',''],rotation='vertical')
            plt.title('Stimulus mod 2', fontsize=fontsize, y=0.9)
        ax.get_yaxis().set_label_coords(-0.12,0.5)
    plt.savefig('figure/schematic_input.pdf',transparent=True)
    plt.show()

    # Plot Rule Inputs
    fig = plt.figure(figsize=(1.0, 0.5))
    ax = fig.add_axes([0.2,0.3,0.7,0.45])
    cmap = sns.cubehelix_palette(light=1, as_cmap=True, rot=0)
    X = x_sample[:,0,1+2*n_eachring:]
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
    plt.yticks([0,n_hidden-1],['1',str(n_hidden)],rotation='vertical')
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
            plt.yticks([0,(n_eachring-1)/2,n_eachring-1],[r'0$\degree$','',r'360$\degree$'],rotation='vertical')
            plt.xticks([])
            plt.title('Response', fontsize=fontsize, y=0.9)

        ax.get_yaxis().set_label_coords(-0.12,0.5)

    plt.savefig('figure/schematic_outputs.pdf',transparent=True)
    plt.show()


if __name__ == "__main__":
    save_name = '0_256migrate'

    # Rules to analyze
    # rule = 'dm1'
    rule = ['dmsgo','dmsnogo','dmcgo','dmcnogo']

    # Easy activity plot, see this function to begin your analysis
    # easy_activity_plot(save_name, rule)

    # Easy connectivity plot
    # easy_connectivity_plot(save_name)

    # Plot sample activity
    # pretty_inputoutput_plot(save_name, rule, save=True)

    # Plot a single in time
    # pretty_singleneuron_plot(save_name, rule, [0], epoch=None, save=False,
    #                          trace_only=True, plot_stim_avg=True)

    # Plot schematic
    schematic_plot(save_name)
