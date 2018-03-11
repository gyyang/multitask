"""Analyze the results after varying hyperparameters."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt

import tools
import variance #maddy added
import clustering
import standard_analysis #maddy added. 

DATAPATH = os.path.join(os.getcwd(), 'data', 'debug')

# Get all the subdirectories
train_dirs = [os.path.join(DATAPATH, d) for d in os.listdir(DATAPATH)]
train_dirs = [d for d in train_dirs if os.path.isdir(d)]

from os import listdir
#check if training is completed 
from os.path import isfile, join
train_dirs = [d for d in train_dirs if 'variance_rule.pkl' in [f for f in listdir(d) if isfile(join(d, f))]]#maddy added


# Compute the number of clusters
n_clusters = list()
scores = list()

LeakyGRU_perfmin = list() #maddy added. check how many perform. 
countval = 0

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

#maddy added -- check no of clusters under various conditions. 
hparams_list = list()
#initdict = {'diag':[], 'randortho':[],'softplus':[], 'relu':[], 'tanh':[],'LeakyRNN':[], 'LeakyGRU':[]}
initdict = {'diag':[], 'randortho':[],'softplus':[], 'relu':[], 'tanh':[],'LeakyRNN':[], 
            'LeakyGRU':[],'l1_h_0':[], 'l1_weight_0':[],'l1_h_1emin5':[], 'l1_weight_1emin5':[],
            'l1_h_1emin4':[], 'l1_weight_1emin4':[],'l1_h_1emin3':[], 'l1_weight_1emin3':[] }

initdictother = {'LeakyRNNsoftplus':[], 'LeakyRNNrelu':[],'LeakyGRUsoftplus':[], 'LeakyGRUrelu':[]}
initdictotherother = {'LeakyRNNsoftplusdiag':[], 'LeakyRNNreludiag':[],'LeakyGRUsoftplusdiag':[],
                      'LeakyGRUreludiag':[],'LeakyRNNsoftplusrandortho':[], 'LeakyRNNrelurandortho':[],
                      'LeakyGRUsoftplusrandortho':[], 'LeakyGRUrelurandortho':[]}

#actdict = {'softplus':[], 'relu':[], 'tanh':[]}
#rnndict = {'LeakyRNN':[], 'LeakyGRU':[]}

for train_dir in train_dirs:
    
    hparams = tools.load_hparams(train_dir)
    #check if performance exceeds target
    log = tools.load_log(train_dir) 
    #if log['perf_avg'][-1] > hparams['target_perf']: 
    if log['perf_min'][-1] > hparams['target_perf']:         
        print "no. of clusters ", log['n_cluster']
        n_clusters.append(log['n_cluster'])
        
        initdict[hparams['w_rec_init']].append(log['n_cluster'])
        initdict[hparams['activation']].append(log['n_cluster'])

        #initdict[hparams['rnn_type']].append(log['n_cluster']) 
        if hparams['activation'] != 'tanh':#maddy added added to compare leakyrnn/gru without tanh.            
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
        

 #maddy added -- check no of clusters under various conditions.    
f, axarr = plt.subplots(7, 1,figsize=(3,12),sharex=True)
u = 0
for key in initdict.keys():
    if 'l1_' not in key:        
        axarr[u].set_title(key + ' ' + str(len(initdict[key]))+ ' mean: '+str(round(np.mean(initdict[key]),2)) )
        axarr[u].hist(initdict[key])
        u += 1
f.subplots_adjust(wspace=.3, hspace=0.3)    
#plt.savefig('./figure/histforcases_96nets.png')
#plt.savefig('./figure/histforcases__pt9_192nets.pdf')
#plt.savefig('./figure/histforcases___leakygrunotanh_pt9_192nets.pdf')

f, axarr = plt.subplots(4, 1,figsize=(3,8),sharex=True)
u = 0
for key in initdictother.keys():
    if 'l1_' not in key:        
        axarr[u].set_title(key + ' ' + str(len(initdictother[key]))+ ' mean: '+str(round(np.mean(initdictother[key]),2)) )
        axarr[u].hist(initdictother[key])
        u += 1
f.subplots_adjust(wspace=.3, hspace=0.3)    
#plt.savefig('./figure/histforcases__leakyrnngrurelusoftplus_pt9_192nets.pdf')


f, axarr = plt.subplots(4, 1,figsize=(3,6),sharex=True)
u = 0
for key in initdictotherother.keys():
    if 'l1_' not in key and 'diag' not in key:        
        axarr[u].set_title(key + ' ' + str(len(initdictotherother[key]))+ ' mean: '+str(round(np.mean(initdictotherother[key]),2)) )
        axarr[u].hist(initdictotherother[key])
        u += 1
f.subplots_adjust(wspace=.3, hspace=0.3)    
#plt.savefig('./figure/histforcases_randortho_notanh_pt9_192nets.pdf')

f, axarr = plt.subplots(4, 1,figsize=(3,6),sharex=True)
u = 0
for key in initdictotherother.keys():
    if 'l1_' not in key and 'randortho' not in key:        
        axarr[u].set_title(key + ' ' + str(len(initdictotherother[key]))+ ' mean: '+str(round(np.mean(initdictotherother[key]),2)) )
        axarr[u].hist(initdictotherother[key])
        u += 1
f.subplots_adjust(wspace=.3, hspace=0.3)    
#plt.savefig('./figure/histforcases_diag_notanh_pt9_192nets.pdf')


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


#maddy added -- plot no of clusters         
#"""
plt.figure()
plt.plot(n_clusters,'.-')
plt.xlabel('networks')
plt.ylabel('no. of clusters')
plt.title('target perf-min 0.9, total:'+str(len(n_clusters))) #
#plt.savefig('./figure/noofclusters_pt9_192nets.pdf')

#"""
#plt.savefig('./figure/noofclusters_pt9_96nets.pdf')
#min score 90% -- pick cluster with max sil score. 

