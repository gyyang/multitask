"""Analyze the results after varying hyperparameters."""

from __future__ import division

from collections import defaultdict
from collections import OrderedDict
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tools
from analysis import variance
from analysis import clustering
from analysis import standard_analysis

mpl.rcParams.update({'font.size': 7})


FIGPATH = os.path.join(os.getcwd(), 'figure')

HP_NAME = {'activation': 'Activation Fun.',
           'rnn_type': 'Network type',
           'w_rec_init': 'Initialization',
           'l1_h': 'L1 rate',
           'l1_weight': 'L1 weight',
           'l2_weight_init': 'L2 weight anchor',
           'target_perf': 'Target perf.'}

#maddy added check tanh fig 4
#root_dir = './data/debug/8' #0, 33 './data/train_all'
"""
variance.compute_variance(root_dir)
variance.plot_hist_varprop_selection(root_dir)
variance.plot_hist_varprop_all(root_dir)
analysis = clustering.Analysis(root_dir, 'rule')
analysis.plot_variance()
"""
"""
standard_analysis.easy_connectivity_plot(root_dir)
rule = 'contextdm1'
standard_analysis.easy_activity_plot(root_dir, rule)
print "easy_connectivity_plot"+root_dir
"""

def compute_n_cluster(model_dirs):
    for model_dir in model_dirs:
        print(model_dir)
        log = tools.load_log(model_dir)
        hp = tools.load_hp(model_dir)
        try:
            analysis = clustering.Analysis(model_dir, 'rule')

            log['n_cluster'] = analysis.n_cluster
            log['model_dir'] = model_dir
            tools.save_log(log)
        except IOError:
            # Training never finished
            assert log['perf_min'][-1] <= hp['target_perf']

        # analysis.plot_example_unit()
        # analysis.plot_variance()
        # analysis.plot_2Dvisualization()

    print("done")


def plot_histogram():
    initdict = defaultdict(list)
    initdictother = defaultdict(list)
    initdictotherother = defaultdict(list)

    for model_dir in model_dirs:
        hp = tools.load_hp(model_dir)
        #check if performance exceeds target
        log = tools.load_log(model_dir)
        #if log['perf_avg'][-1] > hp['target_perf']: 
        if log['perf_min'][-1] > hp['target_perf']:         
            print('no. of clusters', log['n_cluster'])
            n_clusters.append(log['n_cluster'])
            hp_list.append(hp)
            
            initdict[hp['w_rec_init']].append(log['n_cluster'])
            initdict[hp['activation']].append(log['n_cluster'])
    
            #initdict[hp['rnn_type']].append(log['n_cluster']) 
            if hp['activation'] != 'tanh':
                initdict[hp['rnn_type']].append(log['n_cluster']) 
                initdictother[hp['rnn_type']+hp['activation']].append(log['n_cluster']) 
                initdictotherother[hp['rnn_type']+hp['activation']+hp['w_rec_init']].append(log['n_cluster']) 
    
            if hp['l1_h'] == 0:
                initdict['l1_h_0'].append(log['n_cluster'])   
            else: #hp['l1_h'] == 1e-3 or 1e-4 or 1e-5:  
                keyvalstr = 'l1_h_1emin'+str(int(abs(np.log10(hp['l1_h']))))
                initdict[keyvalstr].append(log['n_cluster'])   
                
            if hp['l1_weight'] == 0:            
                initdict['l1_weight_0'].append(log['n_cluster'])  
            else: #hp['l1_h'] == 1e-3 or 1e-4 or 1e-5:    
                keyvalstr = 'l1_weight_1emin'+str(int(abs(np.log10(hp['l1_weight']))))
                initdict[keyvalstr].append(log['n_cluster'])   
                
            #initdict[hp['l1_weight']].append(log['n_cluster'])      
            
    # Check no of clusters under various conditions.
    f, axarr = plt.subplots(7, 1, figsize=(3,12), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_' not in key:
            title = (key + ' ' + str(len(initdict[key])) +
                     ' mean: '+str(round(np.mean(initdict[key]),2)))
            axarr[u].set_title(title)
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_96nets.png')
    # plt.savefig('./figure/histforcases__pt9_192nets.pdf')
    # plt.savefig('./figure/histforcases___leakygrunotanh_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3,8), sharex=True)
    u = 0
    for key in initdictother.keys():
        if 'l1_' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictother[key]))+ ' mean: '+str(round(np.mean(initdictother[key]),2)) )
            axarr[u].hist(initdictother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases__leakyrnngrurelusoftplus_pt9_192nets.pdf')


    f, axarr = plt.subplots(4, 1, figsize=(3,6), sharex=True)
    u = 0
    for key in initdictotherother.keys():
        if 'l1_' not in key and 'diag' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictotherother[key]))+ ' mean: '+str(round(np.mean(initdictotherother[key]),2)) )
            axarr[u].hist(initdictotherother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_randortho_notanh_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3,6),sharex=True)
    u = 0
    for key in initdictotherother.keys():
        if 'l1_' not in key and 'randortho' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictotherother[key]))+ ' mean: '+str(round(np.mean(initdictotherother[key]),2)) )
            axarr[u].hist(initdictotherother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_diag_notanh_pt9_192nets.pdf')


    #regu--
    f, axarr = plt.subplots(4, 1,figsize=(3,8),sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_h_' in key:
            axarr[u].set_title(key + ' ' + str(len(initdict[key]))+ ' mean: '+str(round(np.mean(initdict[key]),2)) )
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    #plt.savefig('./figure/noofclusters_pt9_l1_h_192nets.pdf')

    f, axarr = plt.subplots(4, 1,figsize=(3,8),sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_weight_' in key:
            axarr[u].set_title(key + ' ' + str(len(initdict[key])) + ' mean: '+str(round(np.mean(initdict[key]),2)) )
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    #plt.savefig('./figure/noofclusters_pt9_l1_weight_192nets.pdf')


def get_n_clusters(root_dir):
    model_dirs = tools.valid_model_dirs(root_dir)
    hp_list = list()
    n_clusters = list()
    for i, model_dir in enumerate(model_dirs):
        if i % 50 == 0:
            print('Analyzing model {:d}/{:d}'.format(i, len(model_dirs)))
        hp = tools.load_hp(model_dir)
        log = tools.load_log(model_dir)
        # check if performance exceeds target
        if log['perf_min'][-1] > hp['target_perf']:
            n_clusters.append(log['n_cluster'])
            hp_list.append(hp)
    return n_clusters, hp_list


def _get_hp_ranges():
    """Get ranges of hp."""
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus', 'relu', 'retanh', 'tanh']
    hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU']
    hp_ranges['w_rec_init'] = ['diag', 'randortho']
    hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l2_h'] = [0, 1e-4]
    hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]
    return hp_ranges


def plot_n_clusters(n_clusters, hp_list):
    """Plot the number of clusters.
    
    Args:
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    hp_ranges = _get_hp_ranges()

    # The hp to show
    hp_plots = hp_ranges.keys()

    # Sort by number of clusters
    ind_sort = np.argsort(n_clusters)[::-1]
    n_clusters_sorted = [n_clusters[i] for i in ind_sort]
    hp_list_sorted = [hp_list[i] for i in ind_sort]

    # Fill a matrix with the index of hp
    hp_visualize = np.zeros([len(hp_plots), len(n_clusters)])
    for i, hp in enumerate(hp_list_sorted):
        for j, hp_plot in enumerate(hp_plots):
            ind = hp_ranges[hp_plot].index(hp[hp_plot])
            ind /= len(hp_ranges[hp_plot]) - 1.
            hp_visualize[j, i] = ind

    # Plot results
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.3, 0.6, 0.65, 0.3])
    ax.plot(n_clusters_sorted, '-')
    ax.set_xlim([0, len(n_clusters) - 1])
    ax.set_xticks([0, len(n_clusters) - 1])
    ax.set_xticklabels([])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel('Num. of clusters', fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    import matplotlib as mpl
    import seaborn as sns
    colors = sns.color_palette("hls", 5)
    cmap = mpl.colors.ListedColormap(colors)
    cmap.set_over('0')
    cmap.set_under('1')
    ax = fig.add_axes([0.3, 0.15, 0.65, 0.35])
    ax.imshow(hp_visualize, aspect='auto', cmap='viridis')
    ax.set_xticks([0, len(n_clusters) - 1])
    ax.set_xticklabels([1, len(n_clusters)])
    ax.set_yticks(range(len(hp_plots)))

    hp_plot_names = [HP_NAME[hp] for hp in hp_plots]
    ax.set_yticklabels(hp_plot_names, fontsize=7)
    ax.tick_params(length=0)
    [i.set_linewidth(0.1) for i in ax.spines.values()]
    ax.set_xlabel('Networks', labelpad=-5)
    # plt.title('target perf-min 0.9, total:'+str(len(n_clusters))) #
    plt.savefig(os.path.join(FIGPATH, 'NumClusters.pdf'), transparent=True)
    
    val = n_clusters_sorted
    fig = plt.figure(figsize=(1.0, 0.8))
    ax = fig.add_axes([0.2, 0.4, 0.7, 0.5])
    hist, bin_edges = np.histogram(val, density=True, range=(0, 30),
                                   bins=30)
    color = 'gray'
    ax.hist(val, range=(0, 30),
            density=True, bins=16, ec=color, facecolor=color,
            lw=1.5)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([0, 30])
    ax.set_xlim([0, 30])
    ax.set_xlabel('No. clusters', labelpad=-5)
    plt.tight_layout()
    figname = os.path.join(FIGPATH, 'NumClustersHist.pdf')
    plt.savefig(figname, transparent=True)
    

def _plot_n_cluster_hist(hp_plot, n_clusters=None, hp_list=None):
    """Plot histogram for number of clusters, separating by an attribute.
    
    Args:
        hp_plot: str, the attribute to separate histogram by
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    if hp_list is None:
        n_clusters, hp_list = get_n_clusters()

    # Compare activation, ignore tanh that can not be trained with LeakyRNN
    # hp_plot = 'activation'
    # hp_plot = 'rnn_type'
    # hp_plot = 'w_rec_init'

    n_cluster_dict = OrderedDict()
    hp_ranges = _get_hp_ranges()
    for key in hp_ranges[hp_plot]:
        n_cluster_dict[key] = list()

    for hp, n_cluster in zip(hp_list, n_clusters):
        # if hp_plot == 'activation' and hp['rnn_type'] != 'LeakyGRU':
            # For activation, only analyze LeakyGRU cells
        #     continue
        if hp_plot == 'rnn_type' and hp['activation'] in ['tanh', 'retanh']:
            # For rnn_type, exclude tanh units
            continue
        n_cluster_dict[hp[hp_plot]].append(n_cluster)

    label_map = {'softplus': 'Softplus',
                 'relu': 'ReLU',
                 'retanh': 'Retanh',
                 'tanh': 'Tanh',
                 'LeakyGRU': 'GRU',
                 'LeakyRNN': 'RNN',
                 'randortho': 'Rand.\nOrtho.',
                 'diag': 'Diag.'}
    # fig = plt.figure(figsize=(1.5, 1.2))
    # ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    f, axs = plt.subplots(len(n_cluster_dict), 1,
                          sharex=True, figsize=(1.2, 1.8))
    for i, (key, val) in enumerate(n_cluster_dict.items()):
        ax = axs[i]
        hist, bin_edges = np.histogram(val, density=True, range=(0, 30),
                                       bins=30)
        # plt.bar(bin_edges[:-1], hist, label=key)
        color_ind = i / (len(hp_ranges[hp_plot]) - 1.)
        color = mpl.cm.viridis(color_ind)
        if isinstance(key, float):
            label = '{:1.0e}'.format(key)
        else:
            label = label_map.get(key, str(key))
        ax.hist(val, label=label, range=(0, 30),
                density=True, bins=16, ec=color, facecolor=color,
                lw=1.5)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([0, 15, 30])
        ax.set_xlim([0, 30])
        ax.text(0.7, 0.7, label, fontsize=7, transform=ax.transAxes)
        if i == 0:
            ax.set_title(HP_NAME[hp_plot], fontsize=7)
    # ax.legend(loc=3, bbox_to_anchor=(1, 0), title=HP_NAME[hp_plot], frameon=False)
    ax.set_xlabel('Number of clusters')
    plt.tight_layout()
    figname = os.path.join(FIGPATH, 'NumClustersHist' + hp_plot + '.pdf')
    plt.savefig(figname, transparent=True)

    return n_cluster_dict


def plot_n_cluster_hist(n_clusters, hp_list):
    """Plot histogram of number of clusters.
    
    Args:
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    hp_plots = ['activation', 'rnn_type', 'w_rec_init', 'l1_h', 'l1_weight']
    # hp_plots = ['activation']
    for hp_plot in hp_plots:
        n_cluster_dict = _plot_n_cluster_hist(hp_plot, n_clusters, hp_list)


def get_model_by_activation(activation):
    hp_target = {'activation': activation,
                 'rnn_type': 'LeakyGRU',
                 'w_rec_init': 'diag',
                 'l1_h': 0,
                 'l1_weight': 0}

    return tools.find_model(DATAPATH, hp_target)


def plot_hist_varprop(activation):
    """Plot FTV distribution."""
    model_dir = get_model_by_activation(activation)
    variance.plot_hist_varprop_selection(model_dir, figname_extra='_tanh')


def pretty_singleneuron_plot(activation='tanh'):
    """Plot single neuron activity."""
    model_dir = get_model_by_activation(activation)
    standard_analysis.pretty_singleneuron_plot(
        model_dir, ['contextdm1', 'contextdm2'], range(2)
    )


def activity_histogram(activation):
    """Plot FTV distribution for tanh network."""
    model_dir = get_model_by_activation(activation)
    title = activation
    save_name = '_' + activation
    standard_analysis.activity_histogram(
        model_dir, ['contextdm1', 'contextdm2'], title=title,
        save_name=save_name
    )



if __name__ == '__main__':
    pass
    DATAPATH = os.path.join(os.getcwd(), 'data', 'varyhp')
    # model_dirs = tools.valid_model_dirs(DATAPATH)

    # compute_n_cluster()
    n_clusters, hp_list = get_n_clusters(DATAPATH)
    # plot_n_clusters(n_clusters, hp_list)
    plot_n_cluster_hist(n_clusters, hp_list)
    # pretty_singleneuron_plot('tanh')
    # pretty_singleneuron_plot('relu')
    # [activity_histogram(a) for a in ['tanh', 'relu', 'softplus', 'retanh']]

    
# =============================================================================
#     DATAPATH = os.path.join(os.getcwd(), 'data', 'varyhp_reg')
#     FIGPATH = os.path.join(os.getcwd(), 'figure')
#     model_dirs = tools.valid_model_dirs(DATAPATH)
#     
#     hp_list = list()
#     n_clusters = list()
#     logs = list()
#     perfs = list()
#     for i, model_dir in enumerate(model_dirs):
#         hp = tools.load_hp(model_dir)
#         log = tools.load_log(model_dir)
#         # check if performance exceeds target
#         perfs.append(log['perf_min'][-1])
#         if log['perf_min'][-1] > 0.8:
#             logs.append(log)
#             n_clusters.append(log['n_cluster'])
#             hp_list.append(hp)
# =============================================================================
        

