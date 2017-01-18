"""
Data analysis
Delay-match-to-category task: The boundary is at 45 and 225 for both monkeys
"""

from __future__ import division

import os
import sys
import numpy as np
import pickle
from collections import OrderedDict
import copy
import h5py
import bisect
import matplotlib.pyplot as plt
from scipy.io import loadmat, whosmat
import seaborn.apionly as sns # If you don't have this, then some colormaps won't work

f_dprime = lambda x, y: (abs(np.mean(x)-np.mean(y)))/np.sqrt((np.var(x)+np.var(y))/2)

get_cat = lambda dir : (dir>45)*(dir<225)

map2num = {'m':0, 'b':1, 'DMC':0, '1ID':1, 'MGS':2}

monkey = 'b'
neuron_id = 2

datapath = 'dmc_1id_dataset/data/'

with open(datapath+'code_mapping.pkl','rb') as f:
    code_mapping = pickle.load(f)
with open(datapath+'cond_mapping.pkl','rb') as f:
    cond_mapping = pickle.load(f)

def convert_rawdata(monkey, neuron_id):
    if monkey == 'b':
        fname_base = '/Users/guangyuyang/Dropbox/freedman/oic_dmc_lip/Krithika Mohan - bootsy_dmc_1id_lip_'
    elif monkey == 'm':
        fname_base = '/Users/guangyuyang/Dropbox/freedman/oic_dmc_lip/Krithika Mohan - maceo_dmc_1id_lip'
    else:
        raise ValueError

    fname1 = fname_base + '{:01d}.mat'.format(neuron_id)
    fname2 = fname_base + '{:02d}.mat'.format(neuron_id)
    if os.path.isfile(fname1):
        fname = fname1
    elif os.path.isfile(fname2):
        fname = fname2
    else:
        fname = None

    if fname is not None:
        mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)
        data = mat_dict['data'].__dict__ # as dictionary
        # print whosmat(fname)

        ################# Preprocessing data #############################################

        # Behavior data
        bhv = data['BHV2'].__dict__

        # The code numbers. 18 possible codes in total
        # example code 3: fixation occurs
        code_numbers_used = list(bhv['CodeNumbersUsed'])
        # The name of each code
        code_names_used = list(bhv['CodeNamesUsed'])

        # The type of blocks. 1 means DMC, 3 means 1id.
        block_number = bhv['BlockNumber']
        # Reaction time, but many are NaN, don't understand
        reaction_time = bhv['ReactionTime']
        # Seems to be error information of each trial, but contains 5 possible values
        trial_error = bhv['TrialError']
        # The type of trial condition (261 possibilities, ranging from 0 to 360)
        # Roughly speaking, trial condition<300 belongs to Block 1
        # trial condition>300 belongs to Block 3
        condition_number = bhv['ConditionNumber']
        # The meaning of each condition number can be looked up in bhv.InfoByCond


        n_trial = len(block_number)
        n_trial_bhv = n_trial


        # neural data
        neuro = data['NEURO'].__dict__
        # Notice the number of trials is different in neuro and bhv, but are right-aligned

        n_trial_neuro = neuro['NumTrials']
        trial_times = neuro['TrialTimes']
        trial_durations = neuro['TrialDurations']

        # all spike times
        neuron = neuro['Neuron'].__dict__
        if 'sig001a' in neuron:
            spiketime_all = neuron['sig001a']
        elif 'SPK01a' in neuron:
            spiketime_all = neuron['SPK01a']
        else:
            print('unknown spike entry')
            raise AttributeError

        code_times = list()
        code_numbers = list()
        spike_times = list()
        infos = list()
        for i_bhv in range(n_trial_bhv):
            i_neuro = i_bhv + n_trial_neuro-n_trial_bhv # because right-alignment
            trial_time = trial_times[i_neuro] # start of trial
            trial_dur  = trial_durations[i_neuro]

            ind_start = bisect.bisect_left(spiketime_all, trial_time)
            ind_end   = bisect.bisect_left(spiketime_all, trial_time+trial_dur)
            spike_times.append(spiketime_all[ind_start:ind_end]-trial_time)

            ind_start = bisect.bisect_left(neuro['CodeTimes'], trial_time)
            ind_end   = bisect.bisect_left(neuro['CodeTimes'], trial_time+trial_dur)
            code_times.append(neuro['CodeTimes'][ind_start:ind_end]-trial_time)
            code_numbers.append(neuro['CodeNumbers'][ind_start:ind_end])

            cond = condition_number[i_bhv] # condition number for this trial
            info = bhv['InfoByCond'][cond-1].__dict__ # Convert to dictionary, notice the minus 1 is important
            infos.append(info)

        data_new = {'block_number' : block_number,
                    'reaction_time': reaction_time,
                    'trial_error'  : trial_error,
                    'code_times'   : code_times,
                    'code_numbers' : code_numbers,
                    'spike_times'  : spike_times,
                    'infos'        : infos,
                    'monkey'       : monkey,
                    'neuron_id'    : neuron_id}

        # data_new = spike_times

        fname = monkey + str(neuron_id) + '.pkl'
        with open('data/'+fname,'wb') as f:
            pickle.dump(data_new, f)

def get_mapping():
    fname = '/Users/guangyuyang/Dropbox/freedman/oic_dmc_lip/Krithika Mohan - bootsy_dmc_1id_lip_2.mat'

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)
    data = mat_dict['data'].__dict__ # as dictionary

    # Behavior data
    bhv = data['BHV2'].__dict__

    # The code numbers. 18 possible codes in total
    # example code 3: fixation occurs
    code_numbers_used = list(bhv['CodeNumbersUsed'])
    # The name of each code
    code_names_used = list(bhv['CodeNamesUsed'])

    code_mapping = OrderedDict()
    for number, name in zip(code_numbers_used, code_names_used):
        code_mapping[number] = name

    cond_mapping = OrderedDict()
    for i, info_by_cond in enumerate(bhv['InfoByCond']):
        cond_mapping[i+1] = info_by_cond.__dict__

    with open('data/code_mapping.pkl','wb') as f:
        pickle.dump(code_mapping, f)

    with open('data/cond_mapping.pkl','wb') as f:
        pickle.dump(cond_mapping, f)

def print_code(data, trial):
    print('condition:')
    cond = cond_mapping[data['task_cond_num'][trial]]
    cond.pop('_fieldnames')
    print(cond)
    for t,code in zip(data['code_times'][trial], data['code_numbers'][trial]):
        print t, code, code_mapping.get(code, 'undefined')

def get_processeddata(monkey, neuron_id, rule_codes=None):
    # rule_codes is a list of pairs (rule, code)
    # If rule_codes is given, then for each rule, only select trials that have the code
    if monkey == 'b':
        fname_base = '/Users/guangyuyang/Dropbox/freedman/oic_dmc_lip/Krithika Mohan - x_'
    elif monkey == 'm':
        fname_base = '/Users/guangyuyang/Dropbox/freedman/oic_dmc_lip/Krithika  Mohan - x_'
    else:
        raise ValueError

    fname1 = fname_base + '{:01d}.mat'.format(neuron_id)
    fname2 = fname_base + '{:02d}.mat'.format(neuron_id)
    if os.path.isfile(fname1):
        fname = fname1
    elif os.path.isfile(fname2):
        fname = fname2
    else:
        return None


    print('Converting neuron {:d} from monkey '.format(neuron_id) + monkey)
    #fname = '/Users/guangyuyang/Dropbox/freedman/oic_dmc_lip/Krithika  Mohan - x_01.mat'

    mat_dict = loadmat(fname, squeeze_me=True, struct_as_record=False)
    data_ = mat_dict['x']

    names = ['block_number', 'code_numbers', 'trial_type', 'spike_times', 'lever_release',
     'code_times', 'reward_time', 'ISI_start', 'ISI_end', 'fixation_on', 'fixation_acquired',
     'TrialError', 'fixation_off', 'task_cond_num', 'lever_hold', 'trial_starts', 'trial_num']

    if monkey=='m' and neuron_id in [47,48]: # workaround:
        trial_starts = np.concatenate((data_[0].trial_starts, data_[-1].trial_starts))
    else:
        trial_starts = data_[0].trial_starts

    data = dict()
    data['trial_starts'] = trial_starts
    for name in names:
        data[name] = list()

    for i, d in enumerate(data_):
        trial_start = trial_starts[i]
        d = d.__dict__
        #if d['trial_type'] == u'1-INT/DMC Task': # Only record these
        if True:
            for name in names:
                if name == 'code_times':
                    data[name].append(d[name]-trial_start)
                elif name == 'trial_num':
                    data[name].append(i+1) # plus 1 to start from 1 (Matlab tradition)
                else:
                    data[name].append(d[name])

    # Separate by the three tasks
    block_number = data['block_number']

    datas = {'MGS'  : dict(),
             '1ID'  : dict(),
             'DMC'  : dict()}

    for name, value in data.iteritems():
        for rule, block_n in zip(['MGS', '1ID', 'DMC'], [4,3,1]):
            datas[rule][name] = [data[name][i] for i, block in enumerate(block_number) if block==block_n]

    if rule_codes is not None:
        for rule_code in rule_codes:
            rule, code = rule_code
            data = datas[rule]
            # Get trials that have the corresponding code
            data_code = dict()
            code_numbers = np.array(data['code_numbers'])
            for name, value in data.iteritems():
                data_code[name] = [data[name][i] for i, cn in enumerate(code_numbers) if code in cn]
            datas[rule] = data_code

    return datas


def get_rate(t, spike_times):
    # get firing rate at time t given the list of spike_times
    # Simplest boxcar filter (square wave)
    if hasattr(t, '__iter__'):
        r = list()
        for t_ in t:
            r.append(get_rate(t_, spike_times))
    else:
        if isinstance(spike_times, np.ndarray):
            ind_left = bisect.bisect_left(spike_times, t-200)
            ind_right = bisect.bisect_left(spike_times, t)
            return (ind_right-ind_left)/200.*1000.
        elif isinstance(spike_times, int):
            return (spike_times>t-200)*(spike_times<t)/200.*1000.
        else:
            return 0

def get_rate_bycode(data, rule, code, tshift=0):
    # Get rate for data at time of code
    # And this function only takes data that has one spike train
    n_trial = len(data['block_number'])
    rates = np.zeros(n_trial)
    for i in range(n_trial):
        code_times = data['code_times'][i] # code_times for this trial
        ind = np.argwhere(data['code_numbers'][i]==code)[0][0] # index where code is 26
        t = code_times[ind] # time for this code
        rates[i] = get_rate(t+tshift, data['spike_times'][i]) # get rate at this time

    # Average by conditions
    # Change info into lists
    cond = dict()
    info = cond_mapping[data['task_cond_num'][0]]
    names = info['_fieldnames']
    # initialize
    for name in names:
        cond[name] = np.zeros(n_trial)
    for i in range(n_trial):
        info = cond_mapping[data['task_cond_num'][i]]
        for name in names:
            cond[name][i] = info[name]

    if rule == 'MGS':
        cond_name = 'angle'
    elif rule == '1ID':
        cond_name = 'sample'
    elif rule=='DMC' and code in [26,27]:
        cond_name = 'sample' # condition to average within
    elif rule == 'DMC' and code==4:
        cond_name = 'test1'
    else:
        raise NotImplementedError

    all_conds = cond[cond_name] # all conditions
    return rates, all_conds

def get_rate_bycode_trialaverage(data, rule, code, tshift=0, shuffle_cond=False):
    # Get trial-averaged rate for data at time of code
    # And this function only takes data that has one spike train

    rates, all_conds = get_rate_bycode(data, rule, code, tshift=tshift)
    unique_conds = np.unique(all_conds) # all unique conditions to average within
    rates_averaged = np.zeros(len(unique_conds))
    for i, unique_cond in enumerate(unique_conds):
        ind = all_conds==unique_cond # trials for this sample
        if shuffle_cond:
            # Shuffle conditions
            np.random.shuffle(ind)
        rate_averaged = np.mean(rates[ind]) # average across trials for this cond
        rates_averaged[i] = rate_averaged

    return rates_averaged

def check_singlechannel(data):
    # Is this data single-channel or not?
    spike_times_temp = data['spike_times'][0] # spike times for trial 0
    n_channel = 1
    if not hasattr(spike_times_temp, '__len__'): # must be an int
        single_channel = True
    elif len(spike_times_temp) == 0:
        single_channel = True
    elif hasattr(spike_times_temp[0], '__len__'):
        # is there more than one spike train?
        single_channel = False
        n_channel = len(spike_times_temp)
    else:
        single_channel = True
    return single_channel, n_channel

def get_channel(data, i_channel):
    # get the i-th channel data for a multi-channel unit
    data_channel = copy.deepcopy(data)
    for i_trial, spike_times in enumerate(data['spike_times']):
        data_channel['spike_times'][i_trial] = data['spike_times'][i_trial][i_channel]
    return data_channel

def get_meanstd_byrule(datas, rule):
    # Get rate means and stds for datas given rule
    #rule = 'DMC'
    data = datas[rule]
    if rule == 'MGS':
        # Analyze the memory-guided saccade task
        # There are two epochs analyzed:
        # 1. target epoch. 25 TaskObject-2 ON,  26 TaskObject-2 OFF
        # 2. delay epoch.  26 TaskObject-2 OFF, 36 Fixation spot OFF
        codes = [26, 36]
    elif rule == '1ID':
        # Analyze the one-interval categorization task
        # There are two epochs:
        # 1. target epoch1. 29 TaskObject-4 ON, 25 TaskObject-2 ON
        # 2. target epoch2. 25 TaskObject-2 ON, 36 Fixation spot OFF
        codes = [25, 36]
    elif rule == 'DMC':
        # Analyze the delay-match-to-category task
        # There are two epochs:
        # 1. target epoch1. 25 TaskObject-2 ON,  26 TaskObject-2 OFF
        # 2. delay epoch.   26 TaskObject-2 OFF, 27 TaskObject-3 ON
        # 2. target epoch2. 27 TaskObject-3 ON,  4 Bar / Joystick up
        codes = [26, 27, 4]

    # First only analyze correct trials, i.e. TrialError==0
    # Do not use the following code until better understanding!
    # TrialError = np.array(data['TrialError'])
    # for name, value in data.iteritems():
    #     data[name] = [data[name][i] for i, te in enumerate(TrialError) if te==0]

    # 96 is the code for reward delivery
    # code_numbers = np.array(data['code_numbers'])
    # for name, value in data.iteritems():
    #     data[name] = [data[name][i] for i, cn in enumerate(code_numbers) if 96 in cn]


    rate_means = list()
    rate_stds  = list()
    for code in codes:
        # Get trials that have the corresponding code
        data_code = dict()
        code_numbers = np.array(data['code_numbers'])
        for name, value in data.iteritems():
            data_code[name] = [data[name][i] for i, cn in enumerate(code_numbers) if code in cn]

        # Check if there is just one channel
        single_channel, n_channel = check_singlechannel(data_code)

        if single_channel:
            rates = get_rate_bycode_trialaverage(data_code, rule, code)
            rate_means.append(np.mean(rates))
            rate_stds.append(np.std(rates))
        else:
            for i_channel in range(n_channel):
                data_channel = get_channel(data, i_channel)
                rates = get_rate_bycode_trialaverage(data_channel, rule, code)
                rate_means.append(np.mean(rates))
                rate_stds.append(np.std(rates))

    #data['angle'] = [cond_mapping[data['task_cond_num'][i]]['angle'] for i in range(n_trial)]

    return rate_means, rate_stds

def plot_meanstd():
    rate_means_all = list()
    rate_stds_all  = list()
    for monkey in ['b', 'm']:
    #for monkey in ['b']:
        for neuron_id in range(80):
            datas = get_processeddata(monkey=monkey, neuron_id=neuron_id)
            if datas is not None and not (monkey=='b' and neuron_id==23):
                for rule in ['MGS', '1ID', 'DMC']:
                    rate_means, rate_stds = get_meanstd_byrule(datas, rule)
                    rate_means_all.extend(rate_means)
                    rate_stds_all.extend(rate_stds)

    rate_means_all = np.array(rate_means_all)
    rate_stds_all = np.array(rate_stds_all)


    plt.figure()
    plt.plot(rate_means_all, rate_stds_all, 'o', alpha=0.1)
    plt.xlabel('mean')
    plt.ylabel('std')
    plt.plot([0,100],[0,100])
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.show()

    d_mean_plot, d_std_plot = rate_means_all, rate_stds_all

    color = 'black'
    # ind = d_mean_all>0.5
    prop = 0.5
    ind = np.argsort(d_mean_plot)
    ind = ind[-int(prop*len(ind)):]
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.3,0.3,0.6,0.5])
    hist, bins_edge = np.histogram(d_std_plot[ind]/d_mean_plot[ind], range=(0,3), bins=30)
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=color, edgecolor='none')
    # _ = ax.hist(d_std_plot[ind]/d_mean_plot[ind], range=(0,3), bins=30, color=color, edgecolor='none')
    plt.xlabel('stimulus std/mean', fontsize=7)
    plt.ylabel('count', fontsize=7)
    plt.xticks([0,1,2,3])
    plt.locator_params(axis='y', nbins=2)
    plt.title('Mean activity in top {:d}%'.format(int(100*prop)), fontsize=7)
    plt.xlim([0,3])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.savefig('figure/meanvsstd_hist'+self.config['save_addon']+'.pdf', transparent=True)
    plt.show()

def plot_singleneuron(monkey, neuron_id, save=False):
    # monkey = 'm'
    # neuron_id = 6
    rule_codes = [('DMC', 26), ('1ID', 25), ('MGS', 26)]
    datas = get_processeddata(monkey=monkey, neuron_id=neuron_id, rule_codes=rule_codes)

    if datas is None:
        return None

    fig = plt.figure(figsize=(6.0,1.5))

    for i_rule, rule in enumerate(['DMC','1ID','MGS']):
        data = datas[rule]
        single_channel, n_channel = check_singlechannel(data) # Check if there is just one channel
        if not single_channel:
            print('Not single channel:' +rule+monkey+str(neuron_id))
            data = get_channel(data, 0) # get first channel
        code = dict(rule_codes)[rule]

        tshifts = np.arange(-700,700,30)
        rates_shifts = list()
        for tshift in tshifts:
            rates = get_rate_bycode_trialaverage(data, rule, code, tshift=tshift)
            rates_shifts.append(rates)
        rates_shifts = np.array(rates_shifts)

        if rule in ['DMC','1ID']:
            sample_dir = np.array([15.,   35.,   55.,   75.,  135.,  195.,  215.,  235.,  255.,  315.])
            sample_cat = np.array([0,      0,    1,     1,     1,     1,     1,     0,     0,     0])
        elif rule == 'MGS':
            sample_dir = np.array([0, 45, 90, 135, 180, 225, 270, 315])
            sample_cat = np.array([0,  2,  1,   1,   1,   2,   0,   0]) # Would be

        zerotimelabels = {'DMC' : 'delay on',
                          '1ID' : 'color target on',
                          'MGS' : 'delay on'}

        ax = fig.add_axes([0.2+0.25*i_rule,0.25,0.2,0.6])

        _ = ax.plot(tshifts, rates_shifts[:,sample_cat==0], color='red')
        _ = ax.plot(tshifts, rates_shifts[:,sample_cat==1], color='blue')
        if rule == 'MGS':
            _ = ax.plot(tshifts, rates_shifts[:,sample_cat==2], color='green')
        if rule == 'DMC':
            plt.ylabel('rate (Hz)',fontsize=7)
        title = rule + ' ' + monkey + str(neuron_id)
        plt.title(title, fontsize=7)
        plt.locator_params(nbins=4)
        plt.xlim([tshifts[0],tshifts[-1]])
        plt.ylim(bottom=0)
        plt.xticks([tshifts[0],0,tshifts[-1]],
                   [str(tshifts[0]),zerotimelabels[rule],str(tshifts[-1])])
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    if save:
        plt.savefig('figure/trace_singleneuron'+monkey+str(neuron_id)+'.pdf', transparent=True)
    plt.show()


for monkey in ['m','b']:
    for neuron_id in range(60):
        #plot_singleneuron(monkey, neuron_id, save=True)
        pass

def get_dprime(monkey, neuron_id, trial_average=False):
    rule_codes = [('DMC', 26), ('1ID', 25), ('MGS', 26)]
    datas = get_processeddata(monkey=monkey, neuron_id=neuron_id, rule_codes=rule_codes)

    if datas is None:
        return None

    dprimes = list()
    for rule in ['DMC', '1ID']:
        data = datas[rule]
        single_channel, n_channel = check_singlechannel(data) # Check if there is just one channel
        if not single_channel:
            print('Not single channel:' +rule+monkey+str(neuron_id))
            data = get_channel(data, 0) # get first channel
        code = dict(rule_codes)[rule]

        tshift = 100

        sample_dir = np.array([15.,   35.,   55.,   75.,  135.,  195.,  215.,  235.,  255.,  315.])
        sample_cat = np.array([0,      0,    1,     1,     1,     1,     1,     0,     0,     0])

        if trial_average:
            rates = get_rate_bycode_trialaverage(data, rule, code, tshift=tshift)
            rates_cat0 = rates[sample_cat==0]
            rates_cat1 = rates[sample_cat==1]
        else:
            rates, all_conds = get_rate_bycode(data, rule, code, tshift=tshift)
            rates_cat0 = [rate for rate, cond in zip(rates, all_conds) if get_cat(cond)==0]
            rates_cat1 = [rate for rate, cond in zip(rates, all_conds) if get_cat(cond)==1]

        if np.mean(rates_cat0)<0.001 and np.mean(rates_cat1)<0.001:
            continue

        dprime = f_dprime(rates_cat0, rates_cat1)
        dprimes.append(dprime)

    return dprimes

def plot_dprimes():
    dprimes = list()
    for monkey in ['m','b']:
        for neuron_id in range(60):
            dprimes_temp = get_dprime(monkey, neuron_id, trial_average=True)
            if dprimes_temp is not None:
                dprimes.extend(dprimes_temp)


    plt.figure()
    _ = plt.hist(dprimes, bins=30)
    plt.show()

def get_vartimestim(monkey, neuron_id, rule, shuffle_cond=False):
    rule_codes = [('DMC', 26), ('1ID', 25), ('MGS', 26)]
    datas = get_processeddata(monkey=monkey, neuron_id=neuron_id, rule_codes=rule_codes)

    if datas is None:
        return None

    data = datas[rule]
    single_channel, n_channel = check_singlechannel(data) # Check if there is just one channel
    if not single_channel:
        print('Not single channel:' +rule+monkey+str(neuron_id))
        data = get_channel(data, 0) # get first channel
    code = dict(rule_codes)[rule]

    tshifts = np.arange(-700,700,30)
    rates_shifts = list()
    for tshift in tshifts:
        rates = get_rate_bycode_trialaverage(data, rule, code, tshift=tshift, shuffle_cond=shuffle_cond)
        rates_shifts.append(rates)
    rates_shifts = np.array(rates_shifts) # Time * Condition

    rates = rates_shifts

    r_mean     = rates.mean()
    h_var_time = rates.mean(axis=1).var(axis=0)
    h_var_stim = rates.var(axis=1).mean(axis=0)
    h_var_all  = rates.flatten().var(axis=0)
    return h_var_time, h_var_stim, r_mean



def computeall_vartimestim(shuffle_cond=False):

    rules = ['DMC', '1ID', 'MGS']
    res = {k:[] for k in ['h_var_stim', 'h_var_time', 'h_mean', 'monkey', 'neuron_id', 'rule']}
    for monkey in ['m','b']:
        for neuron_id in range(60):
            temp_list = list()
            for rule in rules:
                temp = get_vartimestim(monkey, neuron_id, rule, shuffle_cond)
                temp_list.append(temp)
            if None not in temp_list:
                for temp, rule in zip(temp_list, rules):
                    # Make sure every unit here has all three rules
                    h_var_time, h_var_stim, h_mean = temp
                    res['h_var_stim'].append(h_var_stim)
                    res['h_var_time'].append(h_var_time)
                    res['h_mean'].append(h_mean)
                    res['monkey'].append(map2num[monkey])
                    res['neuron_id'].append(neuron_id)
                    res['rule'].append(map2num[rule])

    # h_mean = np.array(res['h_mean'])
    for k in res:
        # res[k] = np.array(res[k])[h_mean>0]
        res[k] = np.array(res[k])

    savename = datapath+'h_var'
    if shuffle_cond:
        savename += '_shuffle'
    with open(savename+'.pkl','wb') as f:
        pickle.dump(res, f)

def plot_varstim():
    with open(datapath+'h_var.pkl','rb') as f:
        res = pickle.load(f)

    h_varprop_stim = res['h_var_stim']/(res['h_var_stim']+res['h_var_time'])
    res['h_var'] = res['h_var_stim'] + res['h_var_time']

    rule = 'MGS'
    h_varprop_stim1 = h_varprop_stim[res['rule']==map2num[rule]]
    h_var = res['h_var'][res['rule']==map2num[rule]]
    color = sns.xkcd_palette(['cerulean'])[0]
    fig = plt.figure(figsize=(1.5,1.2))
    ax = fig.add_axes([0.35,0.3,0.55,0.45])
    # hist, bins_edge = np.histogram(h_varprop_stim1, bins=15)
    hist, bins_edge = np.histogram(h_varprop_stim1[h_var>10], bins=15)
    ax.bar(bins_edge[:-1], hist, width=bins_edge[1]-bins_edge[0], color=color, edgecolor='none')
    plt.xlabel('Stim. var. prop. '+rule, fontsize=7)
    plt.ylabel('unit counts', fontsize=7)
    plt.title('Data - Freedman', fontsize=7)
    plt.xticks([0,0.5,1])
    plt.locator_params(axis='y', nbins=2)
    plt.xlim([-0.1,1.1])
    plt.ylim([0, hist.max()*1.2])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #plt.savefig('figure/data_var_stim_prop_hist'+rule+'.pdf', transparent=True)
    plt.show()



    rules = ['DMC', '1ID']
    color = sns.xkcd_palette(['cerulean'])[0]
    fig = plt.figure(figsize=(1.2,1.2))
    ax = fig.add_axes([0.35,0.3,0.55,0.55])
    ax.plot(h_varprop_stim[res['rule']==map2num[rules[0]]],
            h_varprop_stim[res['rule']==map2num[rules[1]]],
            'o', markerfacecolor=color, markeredgecolor=color,markersize=1.0,alpha=0.5)
    plt.xlabel(rules[0], fontsize=7)
    plt.ylabel(rules[1], fontsize=7)
    plt.title('Stim. var. prop.', fontsize=7)
    plt.xticks([0,0.5,1])
    plt.yticks([0,0.5,1])
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('figure/data_var_stim_prop_rulecomparison.pdf', transparent=True)
    plt.show()


# computeall_vartimestim(shuffle_cond=True)

with open(datapath+'h_var.pkl','rb') as f:
    res = pickle.load(f)

with open(datapath+'h_var_shuffle.pkl','rb') as f:
    res_shuffle = pickle.load(f)

# rules = ['DMC', '1ID']
# rules = ['MGS', '1ID']
rules = ['MGS', 'DMC']
h_var_stim_rule0 = res['h_var_stim'][res['rule']==map2num[rules[0]]]
h_var_stim_rule1 = res['h_var_stim'][res['rule']==map2num[rules[1]]]

# Subtract noise variance
h_var_stim_rule0_shuffle = res_shuffle['h_var_stim'][res['rule']==map2num[rules[0]]]
h_var_stim_rule1_shuffle = res_shuffle['h_var_stim'][res['rule']==map2num[rules[1]]]

# h_var_stim_rule0 -= h_var_stim_rule0_shuffle
# h_var_stim_rule1 -= h_var_stim_rule1_shuffle

relu = lambda x : x*(x>0)

h_var_stim_rule0 = relu(h_var_stim_rule0)
h_var_stim_rule1 = relu(h_var_stim_rule1)

var_stim_sum = h_var_stim_rule0+h_var_stim_rule1
var_stim_ratio = h_var_stim_rule0/(h_var_stim_rule0+h_var_stim_rule1)


var_stim_ratio_shuffle = h_var_stim_rule0_shuffle/(h_var_stim_rule0_shuffle+h_var_stim_rule1_shuffle)



plt.hist(var_stim_ratio[var_stim_sum>0], bins=20)
plt.xlim([0, 1])
plt.show()

plt.hist(var_stim_ratio_shuffle, bins=20)
plt.xlim([0, 1])
plt.show()

plt.scatter(h_var_stim_rule0, h_var_stim_rule0_shuffle)
plt.plot([0,300], [0, 300])
lim = 30
plt.xlim([0,lim])
plt.ylim([0,lim])

#==============================================================================
# plt.figure()
# plt.hist(h_var_stim_rule0)
# plt.hist(h_var_stim_rule0_shuffle)
#==============================================================================


