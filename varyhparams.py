"""Analyze the results after varying hyperparameters."""

from __future__ import division

from collections import defaultdict
from collections import OrderedDict
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tools
import variance #maddy added
import clustering
import standard_analysis #maddy added. 

matplotlib.rcParams.update({'font.size': 7})

DATAPATH = os.path.join(os.getcwd(), 'data', 'train_varyhparams')
FIGPATH = os.path.join(os.getcwd(), 'figure')

# Get all the subdirectories
train_dirs = [os.path.join(DATAPATH, d) for d in os.listdir(DATAPATH)]
train_dirs = [d for d in train_dirs if os.path.isdir(d)]

from os import listdir
#check if training is completed 
from os.path import isfile, join
train_dirs = [d for d in train_dirs if 'variance_rule.pkl' in [f for f in listdir(d) if isfile(join(d, f))]]#maddy added


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

"""
hparams_list = list()
for train_dir in train_dirs:
    
    hparams = tools.load_hparams(train_dir)
    #check if performance exceeds target
    log = tools.load_log(train_dir) 
    #if log['perf_avg'][-1] > hparams['target_perf']: 
    if log['perf_min'][-1] > hparams['target_perf']:         

            #if hparams['activation']=='tanh': #maddy added - check tanh
            if hparams['rnn_type']=='LeakyGRU': #maddy added - check tanh
            
                countval+=1
                #print countval, train_dir[-2:] #maddy added
        
                #print hparams['rnn_type'], log['perf_min'][-1] #maddy added 
                #if hparams['rnn_type'] == 'LeakyGRU':
                #    LeakyGRU_perfmin.append(log['perf_min'][-1])
                
                # variance.compute_variance(train_dir)
                analysis = clustering.Analysis(train_dir, 'rule')
                n_clusters.append(analysis.n_cluster)
                scores.append(analysis.scores)
                analysis.plot_cluster_score()
                hparams_list.append(hparams)
        
                log['n_cluster'] = analysis.n_cluster #maddy added save no of clusters. 
                log['train_dir'] =train_dir
                #tools.save_log(log)
                        
                analysis.plot_example_unit()
                analysis.plot_variance()
                analysis.plot_2Dvisualization()
    
            print "done"
"""

        
def plot_histogram():
    initdict = defaultdict(list)
    initdictother = defaultdict(list)
    initdictotherother = defaultdict(list)

    for train_dir in train_dirs:
        hparams = tools.load_hparams(train_dir)
        #check if performance exceeds target
        log = tools.load_log(train_dir) 
        #if log['perf_avg'][-1] > hparams['target_perf']: 
        if log['perf_min'][-1] > hparams['target_perf']:         
            print('no. of clusters', log['n_cluster'])
            n_clusters.append(log['n_cluster'])
            hparams_list.append(hparams)
            
            initdict[hparams['w_rec_init']].append(log['n_cluster'])
            initdict[hparams['activation']].append(log['n_cluster'])
    
            #initdict[hparams['rnn_type']].append(log['n_cluster']) 
            if hparams['activation'] != 'tanh':
                initdict[hparams['rnn_type']].append(log['n_cluster']) 
                initdictother[hparams['rnn_type']+hparams['activation']].append(log['n_cluster']) 
                initdictotherother[hparams['rnn_type']+hparams['activation']+hparams['w_rec_init']].append(log['n_cluster']) 
    
            if hparams['l1_h'] == 0:
                initdict['l1_h_0'].append(log['n_cluster'])   
            else: #hparams['l1_h'] == 1e-3 or 1e-4 or 1e-5:  
                keyvalstr = 'l1_h_1emin'+str(int(abs(np.log10(hparams['l1_h']))))
                initdict[keyvalstr].append(log['n_cluster'])   
                
            if hparams['l1_weight'] == 0:            
                initdict['l1_weight_0'].append(log['n_cluster'])  
            else: #hparams['l1_h'] == 1e-3 or 1e-4 or 1e-5:    
                keyvalstr = 'l1_weight_1emin'+str(int(abs(np.log10(hparams['l1_weight']))))
                initdict[keyvalstr].append(log['n_cluster'])   
                
            #initdict[hparams['l1_weight']].append(log['n_cluster'])      
            
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


def get_n_clusters():
    hparams_list = list()
    for i, train_dir in enumerate(train_dirs):
        if i % 10 == 0:
            print('Analyzing model {:d}/{:d}'.format(i, len(train_dirs)))
        hparams = tools.load_hparams(train_dir)
        log = tools.load_log(train_dir)
        # check if performance exceeds target
        if log['perf_min'][-1] > hparams['target_perf']:
            n_clusters.append(log['n_cluster'])
            hparams_list.append(hparams)
    return n_clusters, hparams_list


def plot_n_clusters():
    """Plot the number of clusters."""
    n_clusters, hparams_list = get_n_clusters()
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus', 'relu', 'tanh']
    hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU']
    hp_ranges['w_rec_init'] = ['diag', 'randortho']
    hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l2_h'] = [0, 1e-4]
    hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]

    # The hparams to show
    hp_plots = hp_ranges.keys()

    # Sort by number of clusters
    ind_sort = np.argsort(n_clusters)[::-1]
    n_clusters_sorted = [n_clusters[i] for i in ind_sort]
    hparams_list_sorted = [hparams_list[i] for i in ind_sort]

    # Fill a matrix with the index of hparams
    hparams_visualize = np.zeros([len(hp_plots), len(n_clusters)])
    for i, hparams in enumerate(hparams_list_sorted):
        for j, hp in enumerate(hp_plots):
            ind = hp_ranges[hp].index(hparams[hp])/1.0/len(hp_ranges[hp])
            hparams_visualize[j, i] = ind

    # Plot results
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.3, 0.6, 0.65, 0.3])
    ax.plot(n_clusters_sorted, '-')
    ax.set_xlim([0, len(n_clusters) - 1])
    ax.set_xticks([0, len(n_clusters) - 1])
    ax.set_xticklabels([])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel('Number of clusters', fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax = fig.add_axes([0.3, 0.15, 0.65, 0.35])
    ax.imshow(hparams_visualize, aspect='auto')
    ax.set_xticks([0, len(n_clusters) - 1])
    ax.set_xticklabels([1, len(n_clusters)])
    ax.set_yticks(range(len(hp_plots)))
    hp_name = {'activation': 'Activation Fun.',
               'rnn_type': 'Network type',
               'w_rec_init': 'Initialization',
               'l1_h': 'L1 rate',
               'l1_weight': 'L1 weight'}
    hp_plot_names = [hp_name[hp] for hp in hp_plots]
    ax.set_yticklabels(hp_plot_names, fontsize=7)
    ax.tick_params(length=0)
    [i.set_linewidth(0.1) for i in ax.spines.itervalues()]
    ax.set_xlabel('Networks', labelpad=-5)
    # plt.title('target perf-min 0.9, total:'+str(len(n_clusters))) #
    plt.savefig(os.path.join(FIGPATH, 'NumClusters.eps'), transparent=True)


n_clusters, hparams_list = get_n_clusters()

# Compare activation, ignore tanh that can not be trained with LeakyRNN
n_cluster_dict = defaultdict(list)
for hp, n_cluster in zip(hparams_list, n_clusters):
    n_cluster_dict[hp['activation']].append(n_cluster)
plt.figure()
for key, val in n_cluster_dict.items():
    hist, bin_edges = np.histogram(val, density=True, range=(0, 30), bins=30)
    plt.hist(val, label=key)
plt.legend()

def plot_hist_varprop_tanh():
    """Plot FTV distribution for tanh network."""
    hp_target = {'activation': 'tanh',
                 'rnn_type': 'LeakyGRU',
                 'w_rec_init': 'diag',
                 'l1_h': 0,
                 'l1_weight': 0}

    for i, train_dir in enumerate(train_dirs):
        hp = tools.load_hparams(train_dir)
        if all(hp[key]==val for key, val in hp_target.items()):
            break
    log = tools.load_log(train_dir)
    # check if performance exceeds target
    assert log['perf_min'][-1] > hp['target_perf']
    variance.plot_hist_varprop_selection(train_dir, figname_extra='_tanh')

