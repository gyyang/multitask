"""
Collections of tasks
"""

from __future__ import division
import os
import numpy as np

#-----------------------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------------------
setup_type = 'standard'
if setup_type == 'standard':
    N_RULE          = 20

    FDGO, REACTGO, DELAYGO,\
    FDANTI, REACTANTI, DELAYANTI,\
    CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT,\
    DMSGO, DMSNOGO, DMCGO, DMCNOGO = range(N_RULE)

    CHOICEDELAY_MOD1_COPY = FIXATION = \
        TIMEDGO = DELAYTIMEDGO = INTREPRO = OIC = DMC = -2 # dummy

    TEST_INIT = -1

elif setup_type == 'nodmc':

    N_RULE          = 16

    REACTGO, FDGO, DELAYGO,\
    REACTANTI, FDANTI, DELAYANTI,\
    CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAYATTEND_MOD1, CHOICEDELAYATTEND_MOD2, CHOICEDELAY_INT = \
        range(N_RULE)

    DMSGO = DMSNOGO = DMCGO = DMCNOGO = CHOICEDELAY_MOD1_COPY = FIXATION = \
        TIMEDGO = DELAYTIMEDGO = INTREPRO = OIC = DMC = -2 # dummy

    TEST_INIT = -1

elif setup_type == 'old_standard':

    N_RULE          = 17

    REACTGO, FDGO, DELAYGO,\
    CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
    REACTANTI, FDANTI, DELAYANTI,\
    DMSGO, DMSNOGO, DMCGO, DMCNOGO = range(N_RULE)

    CHOICEDELAYATTEND_MOD1 = CHOICEDELAYATTEND_MOD2 = CHOICEDELAY_INT =\
    CHOICEDELAY_MOD1_COPY = FIXATION = TIMEDGO = DELAYTIMEDGO = INTREPRO = OIC = DMC = -2 # dummy

    TEST_INIT = -1

elif setup_type == 'OICDMC':

    N_RULE          = 2

    OIC, DMC = range(N_RULE)

    REACTGO, FDGO, DELAYGO,\
    CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
    CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
    REACTANTI, FDANTI, DELAYANTI,\
    DMSGO, DMSNOGO, DMCGO, DMCNOGO = [-2] * 17

    CHOICEDELAYATTEND_MOD1 = CHOICEDELAYATTEND_MOD2 = CHOICEDELAY_INT =\
    CHOICEDELAY_MOD1_COPY = FIXATION = TIMEDGO = DELAYTIMEDGO = INTREPRO = -2 # dummy

    TEST_INIT = -1


#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

# Time constant
TAU                 = 100 # ms, current setting
# TAU                 = 50 # ms
# TAU                 = 10 # ms

def get_dist(original_dist): # Get the distance in periodic boundary conditions
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

#-----------------------------------------------------------------------------------------
# Tasks
#-----------------------------------------------------------------------------------------

class Task(object):
    def __init__(self, config, tdim, batch_size):
        '''
        Initialize
        :param config: dictionary of configuration
        :param tdim:
        :param batch_size:
        :return:
        '''
        ## TEMPORARY FOR BACKWARD COMPATIBILITY
        if 'num_ring' not in config:
            config['num_ring'] = 2 # default number
        if 'rule_start' not in config:
            config['rule_start'] = 1+config['num_ring']*config['N_RING']
        if 'sigma_x' not in config:
            config['sigma_x'] = 0.01


        self.float_type = 'float32' # This should be the default
        self.config     = config
        self.dt         = self.config['dt']
        N_RING          = self.config['N_RING']
        num_ring        = self.config['num_ring']

        self.slices     = {'fix_in'     : 0,
                           'fix_out'    : 0,
                           'out'        : slice(1, N_RING+1)
                           }

        # Add the num_ring rings of stimulus inputs
        for ring in range(num_ring):
            self.slices['stim_mod{:d}'.format(ring+1)] = slice(1+ring*N_RING,
                                                              1+(ring+1)*N_RING)

        XDIM            = 1 + num_ring*N_RING + N_RULE
        YDIM            = 1 + N_RING
        self.pref       = np.arange(0,2*np.pi,2*np.pi/N_RING) # preferences
        self.N_RING     = N_RING

        self.batch_size = batch_size
        self.tdim       = tdim
        self.x          = np.zeros((tdim, batch_size, XDIM), dtype=self.float_type)
        self.y          = np.zeros((tdim, batch_size, YDIM), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:,:,:]   = 0.05
        # y_loc is the stimget location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc      = -np.ones((tdim, batch_size)      , dtype=self.float_type)

        self.XDIM       = XDIM
        self.YDIM       = YDIM

    def expand(self, var):
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        '''
        Add an input or stimget output
        locs not needed for fix_in or fix_out loc_type
        '''
        ons         = self.expand(ons)
        offs        = self.expand(offs)
        strengths   = self.expand(strengths)
        mods        = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]:offs[i],i,self.slices[loc_type]] = 1
            elif loc_type == 'stim':
                mod = 'stim_mod{:d}'.format(mods[i])
                self.x[ons[i]:offs[i],i,self.slices[mod]] += self.add_x_loc(locs[i])*strengths[i]
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i],i,self.slices[loc_type]] = 0.8
                else:
                    self.y[ons[i]:offs[i],i,self.slices[loc_type]] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]:offs[i],i,self.slices[loc_type]] += self.add_y_loc(locs[i])*strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]:offs[i],i,self.slices[loc_type]] += y_tmp
                self.y_loc[ons[i]:offs[i],i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self, dt):
        '''
        Add input noise
        :param sigma:
        :return:
        '''
        self.x += self.config['rng'].randn(*self.x.shape)*self.config['sigma_x']*np.sqrt(2/dt*TAU)

    def add_c_mask(self, pre_offs, post_ons):
        '''
        Add a cost mask
        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        '''
        pre_on   = int(100/self.dt) # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        # if post_ons[0] is None:
        #     ValueError('Post_on can no longer be None')

        # for i in range(self.batch_size):
        #     # Post response periods usually have the same length across tasks
        #     self.c_mask[post_ons[i]:, i, :] = 1.
        #     # Pre-response periods usually have different lengths across tasks
        #     # To keep cost comparable across tasks
        #     # Scale the cost mask of the pre-response period by a factor
        #     self.c_mask[pre_on:pre_offs[i], i, :] = (self.tdim-post_ons[i])/(pre_offs[i]-pre_on)



        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.YDIM), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                c_mask[post_ons[i]:, i, :] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 1.

            # self.c_mask[:, :, 0] *= self.N_RING # Fixation is important
            c_mask[:, :, 0] *= 2. # Fixation is important

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size, self.YDIM))
        else:
            c_mask = np.zeros((self.tdim, self.batch_size), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                # Having it larger than 1 encourages the network to achieve higher performance
                c_mask[post_ons[i]:, i] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i] = 1.

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size,))
            self.c_mask /= self.c_mask.mean()

    def add_rule(self, rule, on=None, off=None, strength=1.):
        self.x[on:off,:,self.config['rule_start']+rule] = strength # Have rule input

    def add_x_loc(self, x_loc):
        dist = get_dist(x_loc-self.pref) # periodic boundary
        dist /= np.pi/8
        return 0.8*np.exp(-dist**2/2)

    def add_y_loc(self, y_loc):
        dist = get_dist(y_loc-self.pref) # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi/8
            y    = 0.8*np.exp(-dist**2/2)
        else:
            # One-hot output
            y = np.zeros_like(dist)
            ind = np.argmin(dist)
            y[ind] = 1.
        return y

def test_init(config, mode, **kwargs):
    '''
    Test initialization of model. mode is not actually used
    Fixation is on then off.
    '''
    dt = config['dt']
    tdim = int(10000/dt)
    fix_offs  = [int(800/dt)]
    batch_size = 1

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)

    return task

def fixation(config, mode, **kwargs):
    '''
    Direct tracking: follow the fixation input
    Generate one batch of trials
    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    :param dt: dt of time discretization
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random':
        batch_size = kwargs['batch_size']
        tdim = int(rng.uniform(1000,2000)/dt)

    elif mode == 'sample':
        batch_size = 1
        tdim = int(kwargs['t_tot']/dt)

    elif mode == 'test':
        batch_size = 2**BS_EXPO
        tdim = int(1000/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('fix_out')
    task.add_c_mask(pre_offs=None, post_ons=None)

    task.epochs = {'fix1'     : (None, None)}

    return task

def go_obsolete(config, mode, anti_response, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    A stimget will be shown once the fixation is off
    And output should saccade to the stimget location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimget is shown between (fix_off,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimget location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        # A list of locations of fixation points and fixation off time
        fix_offs = int(rng.uniform(500,1500)/dt)
        tdim = int(500/dt) + fix_offs

        # A list of locations of stimgets (they are always on)
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stim_mod  = rng.choice([1,2])

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(1500/dt)])
        stim_locs  = [1.5*np.pi]
        stim_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        fix_offs  = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_mod   = ind_stim_mod + 1

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        batch_size = len(stim_locs)

        # Time of stimgets on/off
        fix_offs = int(1000/dt)
        tdim = int(400/dt) + fix_offs
        stim_mod   = 1

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim_locs, ons=fix_offs, mods=stim_mod)
    task.add('fix_out', offs=fix_offs)
    task.add('out', response_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def go_(config, mode, anti_response, **kwargs):
    '''
    Fixate when fixation point is shown,
    A stimget will be shown, and the output should saccade to the stimget location
    Generate one batch of trials

    The fixation is shown between (0, T)
    The stimget is shown between (fix_off,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimget location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        # A list of locations of fixation points and fixation off time
        stim_ons = int(rng.uniform(500,2500)/dt)
        tdim = int(500/dt) + stim_ons

        # A list of locations of stimgets (they are always on)
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stim_mod  = rng.choice([1,2])

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        stim_ons  = np.array([int(1500/dt)])
        stim_locs  = [1.5*np.pi]
        stim_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        stim_ons  = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_mod   = ind_stim_mod + 1

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        batch_size = len(stim_locs)

        # Time of stimgets on/off
        stim_ons = int(1000/dt)
        tdim = int(400/dt) + stim_ons
        stim_mod   = 1

    # time to check the saccade location
    check_ons  = stim_ons + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    task.add('fix_out', offs=stim_ons)
    task.add('out', response_locs, ons=stim_ons)
    task.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'go1'      : (stim_ons, None)}

    return task

def go(config, mode, **kwargs):
    return go_(config, mode, False, **kwargs)

def remapgo(config, mode, **kwargs):
    return go_(config, mode, True, **kwargs)

def inhgo_(config, mode, anti_response, **kwargs):
    '''
    Go with inhibitory control. Important difference with Go task is that
    the stimulus is presented from the beginning.

    Fixate whenever fixation point is shown,
    A stimget will be shown from the beginning
    And output should saccade to the stimget location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimget is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimget location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        # A list of locations of fixation points and fixation off time

        # A list of locations of stimulus (they are always on)
        stim_locs = rng.rand(batch_size)*2*np.pi
        stim_mod  = rng.choice([1,2])
        stim_ons  = int(rng.uniform(300,700)/dt)

        fix_offs  = stim_ons + int(rng.uniform(500,1500)/dt)
        tdim      = int(500/dt) + fix_offs

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(1500/dt)])
        stim_locs  = [1.5*np.pi]
        stim_ons   = np.array([int(300/dt)])
        stim_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2000/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        stim_ons   = int(500/dt)
        fix_offs   = int(1500/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_mod   = ind_stim_mod + 1

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        stim_time = int(p['stim_time']/dt)
        batch_size = len(stim_locs)

        # Time of stimgets on/off
        stim_ons   = int(300/dt)
        fix_offs  = stim_ons + stim_time
        tdim      = int(400/dt) + fix_offs
        stim_mod   = 1

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    task.add('fix_out', offs=fix_offs)
    task.add('out', response_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def inhgo(config, mode, **kwargs):
    return inhgo_(config, mode, False, **kwargs)

def inhremapgo(config, mode, **kwargs):
    return inhgo_(config, mode, True, **kwargs)

def delaygo_(config, mode, anti_response, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    saccade to the location of the previously shown stimget
    whenever the fixation point is off
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimget is shown between (stim_on, stim_off)

    The output should be fixation location for (0, fix_off)
    and the stimget location for (fix_off, T)

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimgets and on/off time
        stim_locs = rng.rand(batch_size)*2*np.pi
        stim_ons  = int(rng.choice([300, 500, 700])/dt)
        stim_offs = stim_ons + int(rng.choice([150, 200, 250])/dt)
        fix_offs  = stim_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        tdim      = fix_offs + int(500/dt)
        stim_mod  = rng.choice([1,2])

    elif mode == 'sample':
        tdim = int(2000/dt)
        fix_offs   = np.array([int(1500/dt)])
        stim_locs  = [1.5*np.pi]
        stim_ons   = [int(500/dt)]
        stim_offs  = [int(700/dt)]
        stim_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        fix_offs   = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_ons   = int(500/dt)
        stim_mod   = ind_stim_mod + 1
        stim_offs  = int(1000/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        # Time of stimgets on/off
        stim_ons    = int(p['stim_ons']/dt)
        stim_offs   = int(p['stim_offs']/dt)
        delay_time = int(p['delay_time']/dt)
        fix_offs   = stim_offs + delay_time
        tdim       = int(400/dt) + fix_offs
        stim_mod    = 1

        batch_size = len(stim_locs)

    check_ons= fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    task.add('fix_out', offs=fix_offs)
    task.add('out', response_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, stim_offs),
                   'delay1'   : (stim_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def delaygo(config, mode, **kwargs):
    return delaygo_(config, mode, False, **kwargs)

def delayremapgo(config, mode, **kwargs):
    return delaygo_(config, mode, True, **kwargs)

def choicego_(config, mode, stim_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimgets are shown, saccade to the one with higher intensity
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimgets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimget

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimgets (they are always on)
        stim_dist = rng.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        # Target strengths
        stims_mean = rng.uniform(0.8,1.2,(batch_size,))
        # stims_diff = rng.uniform(0.01,0.2,(batch_size,))
        # stims_diff = rng.choice([0.02, 0.04, 0.08], (batch_size,)) # Encourage integration
        # stims_coh  = rng.choice([0.16, 0.32, 0.64], (batch_size,))

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if ('easy_task' in config) and config['easy_task']:
            stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])
        stims_coh  = rng.choice(stim_coh_range, (batch_size,))
        stims_sign = rng.choice([1,-1], (batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign

        # Time of stimgets on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        # stim_dur = int(rng.uniform(300,1500)/dt)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(0.75*tdim)])
        stim1_locs = [0.5*np.pi]
        stim2_locs = [1.5*np.pi]
        stim1_strengths = [0.9]
        stim2_strengths = [1.1]
        stim_ons  = np.array([int(0.15*tdim)])
        batch_size = 1

    elif mode == 'test':
        # Dense coverage of the stimulus space
        tdim = int(2500/dt)
        n_stim_loc, n_stim1_strength = batch_shape = 20, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim1_strength = np.unravel_index(range(batch_size),batch_shape)
        fix_offs  = int(2000/dt)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_strengths = 0.4*ind_stim1_strength/n_stim1_strength+0.8
        stim2_strengths = 2 - stim1_strengths
        stim_ons  = int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_strengths = p['stim1_strengths']
        stim2_strengths = p['stim2_strengths']
        stim_time = int(p['stim_time']/dt)
        batch_size = len(stim1_locs)

        # Time of stimgets on/off
        stim_ons = int(300/dt)
        fix_offs = int(300/dt) + stim_time
        tdim = int(400/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)


    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_strengths, mods=stim_mod)
    task.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_strengths, mods=stim_mod)
    task.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    task.add('out', stim_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicego_mod1(config, mode, **kwargs):
    return choicego_(config, mode, 1, **kwargs)

def choicego_mod2(config, mode, **kwargs):
    return choicego_(config, mode, 2, **kwargs)

def choiceattend_genstim(batch_size, rng, stim_coh_range=None):
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.array([0.16, 0.32, 0.64])*1.0
    stim_coh  = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([+1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh*stim_sign
    stim2_strengths = stim_mean - stim_coh*stim_sign
    return stim1_strengths, stim2_strengths

def choicego_attend_(config, mode, attend_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimgets are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimgets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimget

    In this task, if the model's strategy is to ignore context, and integrate both,
    then the maximum performance is 75%. So we need to make the highest correct performance
    much higher than that.

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimgets, same locations for both modalities
        stim_dist = rng.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if ('easy_task' in config) and config['easy_task']:
            stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = choiceattend_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = choiceattend_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = choiceattend_genstim(batch_size, rng, stim_coh_range)

            stim1_mod12_diff = stim1_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff/2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff/2

            stim2_mod12_diff = stim2_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff/2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff/2

        # Time of stimgets on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        # stim_dur = int(500/dt) # temp, for fast training
        # stim_dur = int(rng.uniform(300, 1500)/dt)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        # stim_dur = rng.choice((np.array([600, 900, 1350, 2000])/dt).astype(int)) # Current setting
        # stim_dur = int(800/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(0.75*tdim)])
        stim1_locs = [0.5*np.pi]
        stim2_locs = [1.5*np.pi]
        stim1_mod1_strengths = [0.9]
        stim2_mod1_strengths = [1.1]
        stim1_mod2_strengths = [1.1]
        stim2_mod2_strengths = [0.9]
        stim_ons  = np.array([int(0.15*tdim)])
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)
        fix_offs  = int(2000/dt)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths
        stim_ons  = int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_mod1_strengths = p['stim1_mod1_strengths']
        stim2_mod1_strengths = p['stim2_mod1_strengths']
        stim1_mod2_strengths = p['stim1_mod2_strengths']
        stim2_mod2_strengths = p['stim2_mod2_strengths']
        stim_time = int(p['stim_time']/dt)
        batch_size = len(stim1_locs)

        # Time of stimgets on/off
        stim_ons = int(400/dt)
        fix_offs = int(400/dt) + stim_time
        tdim = int(400/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_mod1_strengths, mods=1)
    task.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_mod1_strengths, mods=1)
    task.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_mod2_strengths, mods=2)
    task.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_mod2_strengths, mods=2)
    task.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    task.add('out', stim_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicego_attend_mod1(config, mode, **kwargs):
    return choicego_attend_(config, mode, 1, **kwargs)

def choicego_attend_mod2(config, mode, **kwargs):
    return choicego_attend_(config, mode, 2, **kwargs)

def choicego_int(config, mode, **kwargs):
    return choicego_attend_(config, mode, 'both', **kwargs)

def choicedelaygo_(config, mode, stim_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimgets are shown at different time, with different intensities

    The fixation is shown between (0, fix_off)
    The two stimgets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimget

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimgets (they are always on)
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stims_mean = rng.uniform(0.8,1.2,(batch_size,))
        # stims_diff = rng.choice([0.32,0.64,1.28],(batch_size,))

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            stim_coh_range = np.array([0.16,0.32,0.64])

        stims_coh  = rng.choice(stim_coh_range,(batch_size,))
        stims_sign = rng.choice([1,-1], (batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign

        # stim1_strengths = rng.uniform(0.25,1.75,(batch_size,))
        # stim2_strengths = rng.uniform(0.25,1.75,(batch_size,))

        # Time of stimgets on/off
        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(300/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        stim2_offs = stim2_ons + int(300/dt)
        fix_offs  = stim2_offs + int(rng.uniform(100,300)/dt)

        # stim2_ons  = (np.ones(batch_size)*rng.choice([400,500,600,700,1400])/dt).astype(int)
        # stim2_ons  = (np.ones(batch_size)*rng.choice([400,600,1000,1400,2000])/dt).astype(int)
        # stim2_ons  = (np.ones(batch_size)*rng.uniform(2800,3200)/dt).astype(int)

        # each batch consists of sequences of equal length
        tdim = fix_offs + int(500/dt) # longest trial

    elif mode == 'sample':
        stim1_locs = [0.5*np.pi]
        stim2_locs = [1.5*np.pi]
        stim1_strengths = [2.0] # always make stim1 stronger
        stim2_strengths = [0.75]
        stim1_ons = [int(100/dt)]
        stim1_offs = [int(300/dt)]

        tdim = int(2000/dt)
        fix_offs  = np.array([int(1800/dt)])
        stim2_ons = [int(1500/dt)]
        stim2_offs = [int(1700/dt)]
        batch_size = 1

    elif mode == 'test':
        tdim = int(3000/dt)
        n_stim_loc, n_stim1_strength = batch_shape = 20, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim1_strength = np.unravel_index(range(batch_size),batch_shape)

        fix_offs  = int(2700/dt)
        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_strengths = 1.0*ind_stim1_strength/n_stim1_strength+0.5
        stim2_strengths = 2 - stim1_strengths
        stim1_ons = int(500/dt)
        stim1_offs = int(1000/dt)
        stim2_ons = int(2000/dt)
        stim2_offs = int(2500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs       = p['stim1_locs']
        stim2_locs       = p['stim2_locs']
        stim1_strengths  = p['stim1_strengths']
        stim2_strengths  = p['stim2_strengths']
        stim1_ons        = int(p['stim1_ons']/dt)
        stim1_offs       = int(p['stim1_offs']/dt)
        stim2_ons        = int(p['stim2_ons']/dt)
        stim2_offs       = int(p['stim2_offs']/dt)
        batch_size = len(stim1_locs)

        fix_offs = int(200/dt) + stim2_offs
        tdim = int(300/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_strengths, mods=stim_mod)
    task.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_strengths, mods=stim_mod)
    task.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    task.add('out', stim_locs, ons=fix_offs)


    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'stim2'     : (stim2_ons, stim2_offs),
                   'delay2'   : (stim2_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicedelaygo_mod1(config, mode, **kwargs):
    return choicedelaygo_(config, mode, 1, **kwargs)

def choicedelaygo_mod2(config, mode, **kwargs):
    return choicedelaygo_(config, mode, 2, **kwargs)

def choicegodelay_attend_(config, mode, attend_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimgets are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimgets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimget

    In this task, if the model's strategy is to ignore context, and integrate both,
    then the maximum performance is 75%. So we need to make the highest correct performance
    much higher than that.

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of stimgets, same locations for both modalities
        stim_dist = rng.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            stim_coh_range = np.array([0.16, 0.32, 0.64])

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = \
                choiceattend_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = \
                choiceattend_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = \
                choiceattend_genstim(batch_size, rng, stim_coh_range)

            stim1_mod12_diff = stim1_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim1_mod1_strengths = stim1_strengths + stim1_mod12_diff/2
            stim1_mod2_strengths = stim1_strengths - stim1_mod12_diff/2

            stim2_mod12_diff = stim2_strengths * \
                               np.random.uniform(0.2, 0.8, (batch_size,)) * \
                               np.random.choice([+1, -1], (batch_size,))
            stim2_mod1_strengths = stim2_strengths + stim2_mod12_diff/2
            stim2_mod2_strengths = stim2_strengths - stim2_mod12_diff/2

        # Time of stimgets on/off
        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(300/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        stim2_offs = stim2_ons + int(300/dt)
        fix_offs  = stim2_offs + int(rng.uniform(100,300)/dt)

        # each batch consists of sequences of equal length
        tdim = fix_offs + int(500/dt) # longest trial

    elif mode == 'sample':
        stim1_locs = [0.5*np.pi]
        stim2_locs = [1.5*np.pi]
        stim1_mod1_strengths = [1.2]
        stim2_mod1_strengths = [0.8]
        stim1_mod2_strengths = [0.8]
        stim2_mod2_strengths = [1.2]
        batch_size = 1

        stim1_ons = [int(100/dt)]
        stim1_offs = [int(300/dt)]
        stim2_ons = [int(1500/dt)]
        stim2_offs = [int(1700/dt)]
        fix_offs  = np.array([int(1800/dt)])
        tdim = int(2000/dt)

    elif mode == 'test':
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths

        stim1_ons = int(500/dt)
        stim1_offs = int(1000/dt)
        stim2_ons = int(2000/dt)
        stim2_offs = int(2200/dt)
        fix_offs  = int(2300/dt)
        tdim = int(2500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim1_mod1_strengths = p['stim1_mod1_strengths']
        stim2_mod1_strengths = p['stim2_mod1_strengths']
        stim1_mod2_strengths = p['stim1_mod2_strengths']
        stim2_mod2_strengths = p['stim2_mod2_strengths']
        # stim1_ons        = int(500/dt)
        # stim1_offs       = int(1000/dt)
        # stim2_ons        = int(p['stim_time']/dt) + stim1_offs
        # stim2_offs       = int(500/dt) + stim2_ons
        stim1_ons        = int(300/dt)
        stim1_offs       = int(600/dt)
        stim2_ons        = int(p['stim_time']/dt) + stim1_offs
        stim2_offs       = int(300/dt) + stim2_ons
        batch_size = len(stim1_locs)

        # Time of stimgets on/off
        fix_offs = int(200/dt) + stim2_offs
        tdim = int(300/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod1_strengths, mods=1)
    task.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod1_strengths, mods=1)
    task.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod2_strengths, mods=2)
    task.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod2_strengths, mods=2)
    task.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    task.add('out', stim_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'stim2'     : (stim2_ons, stim2_offs),
                   'delay2'   : (stim2_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicegodelay_attend_mod1(config, mode, **kwargs):
    return choicegodelay_attend_(config, mode, 1, **kwargs)

def choicegodelay_attend_mod2(config, mode, **kwargs):
    return choicegodelay_attend_(config, mode, 2, **kwargs)

def choicegodelay_int(config, mode, **kwargs):
    return choicegodelay_attend_(config, mode, 'both', **kwargs)

def delaymatchsample_(config, mode, matchnogo, **kwargs):
    '''
    Delay-match-to-sample

    Two stimuli are shown, separated in time, either at the same location or not
    Fixate before the second stimulus is shown

    If matchnogo is one, then:
    If the two stimuli are the same, then keep fixation.
    If the two stimuli are different, then saccade to the location of the stimulus

    If matchnogo is zero, then:
    If the two stimuli are different, then keep fixation.
    If the two stimuli are the same, then saccade to the location of the stimulus

    The first stimget is shown between (stim1_on, stim1_off)
    The second stimget is shown between (stim2_on, T)

    The output should be fixation location for (0, stim2_on)
    If two stimuli the different location, then for (stim2_on, T) go to stim2_loc
    Otherwise keep fixation

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        stim1_mod  = rng.choice([1,2])
        stim2_mod  = rng.choice([1,2])
        # A list of locations of stimgets
        # Since stim1 is always shown first, it's important that we completely randomize their relative positions
        matchs    = rng.choice([0,1],(batch_size,)) # match or not?
        # stim_dist range between 1/18*pi and (2-1/18*pi), corresponding to 10 degree to 350 degree
        stim_dist  = rng.uniform(np.pi/9,np.pi*17./9.,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist*(1-matchs))%(2*np.pi)

        # Time of stimgets on/off
        stim1_ons  = int(rng.choice([100, 300, 500])/dt)
        stim1_offs = stim1_ons + int(500/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        tdim       = stim2_ons + int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        stim1_mod = 1
        stim2_mod = 1
        matchs    = np.array([0])
        stim_dist = 0.5*np.pi
        stim1_locs = np.array([0.5*np.pi])
        stim2_locs = np.array([(0.5*np.pi+stim_dist*(1-matchs))%(2*np.pi)])
        stim1_ons = np.array([int(300/dt)])
        stim1_offs = stim1_ons + int(200/dt)
        stim2_ons = stim1_offs + int(1000/dt)
        batch_size = 1

    elif mode == 'test':
        # Set this test so the model always respond
        n_stim_loc, n_mod1, n_mod2 = batch_shape = 20, 2, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size),batch_shape)

        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        matchs = (1 - matchnogo)*np.ones(batch_size) # make sure the response is Go
        stim2_locs = (stim1_locs+np.pi*(1-matchs))%(2*np.pi)

        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(500/dt)
        stim2_ons  = stim1_offs + int(1200/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        matchs = get_dist(stim1_locs-stim2_locs)<np.pi/36. # 5 degree
        batch_size = len(stim1_locs)

        tdim = int(2500/dt)
        stim1_ons  = int(500/dt)
        stim1_offs = int(800/dt)
        stim2_ons  = int(2000/dt)
        stim1_mod = 1
        stim2_mod = 1

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    task = Task(config, tdim, batch_size)

    task.add('fix_in')
    task.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    task.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimget location


    task.add('fix_out', offs=fix_out_offs)
    task.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    task.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return task

def delaymatchsamplego(config, mode, **kwargs):
    return delaymatchsample_(config, mode, 0, **kwargs)

def delaymatchsamplenogo(config, mode, **kwargs):
    return delaymatchsample_(config, mode, 1, **kwargs)

def delaymatchcategory_(config, mode, matchnogo, **kwargs):
    '''
    Delay-match-to-category

    Two stimuli are shown, separated in time, either at the locations of the same category or not
    Fixate before the second stimulus is shown

    If matchnogo is one, then:
    If the two stimuli are the same, then keep fixation.
    If the two stimuli are different, then saccade to the location of the stimulus

    If matchnogo is zero, then:
    If the two stimuli are different, then keep fixation.
    If the two stimuli are the same, then saccade to the location of the stimulus

    The first stimget is shown between (stim1_on, stim1_off)
    The second stimget is shown between (stim2_on, T)

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length

        # Use only mod 1 for input
        stim1_mod  = rng.choice([1,2])
        stim2_mod  = rng.choice([1,2])
        # A list of locations of stimgets
        # Since stim1 is always shown first, it's important that we completely randomize their relative positions
        # stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        # stim2_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim1_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))
        stim2_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))

        # Time of stimgets on/off
        stim1_ons  = int(rng.choice([100, 300, 500])/dt)
        stim1_offs = stim1_ons + int(500/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        tdim       = stim2_ons + int(500/dt)

    elif mode == 'sample':
        stim1_mod = 1
        stim2_mod = 1
        stim1_locs = np.array([0.25*np.pi])
        stim2_locs = np.array([0.25*np.pi])
        stim1_ons = np.array([int(300/dt)])
        stim1_offs = stim1_ons + int(200/dt)
        stim2_ons = stim1_offs + int(1000/dt)
        batch_size = 1
        tdim = stim2_ons[0] + int(500/dt)

    elif mode == 'test':
        # Set this test so the model always respond
        n_stim_loc, n_mod1, n_mod2 = batch_shape = 20, 2, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_mod1, ind_mod2 = np.unravel_index(range(batch_size),batch_shape)

        stim1_mod = ind_mod1 + 1
        stim2_mod = ind_mod2 + 1

        n_stim_loc2 = n_stim_loc/2
        stim1_locs_ = np.concatenate(((0.1+0.8*np.arange(n_stim_loc2)/n_stim_loc2),
                                    (1.1+0.8*np.arange(n_stim_loc2)/n_stim_loc2)))*np.pi
        stim1_locs = np.array([stim1_locs_[i] for i in ind_stim_loc])
        matchs = (1 - matchnogo)*np.ones(batch_size) # make sure the response is Go
        stim2_locs = (stim1_locs+np.pi*(1-matchs))%(2*np.pi)

        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(500/dt)
        stim2_ons  = stim1_offs + int(1200/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        batch_size = len(stim1_locs)

        tdim = int(2500/dt)
        stim1_ons  = int(500/dt)
        stim1_offs = int(800/dt)
        stim2_ons  = int(2000/dt)
        stim1_mod = 1
        stim2_mod = 1

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimget 1
    stim2_cats = stim2_locs<np.pi # Category of stimget 2
    matchs    = stim1_cats==stim2_cats

    task = Task(config, tdim, batch_size)

    task.add('fix_in')
    task.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    task.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimget location


    task.add('fix_out', offs=fix_out_offs)
    task.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    task.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return task

def delaymatchcategorygo(config, mode, **kwargs):
    return delaymatchcategory_(config, mode, 0, **kwargs)

def delaymatchcategorynogo(config, mode, **kwargs):
    return delaymatchcategory_(config, mode, 1, **kwargs)


def oic(config, mode, **kwargs):
    '''
    One-interval categorization

    One stimuli is shown in ring 1 for 1000ms,
    then two stimgets are shown in rings 2 and 3.
    If the stimulus is category 1, then go to the location of ring 2, otherwise ring 3

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''

    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        # A list of locations of stimgets
        stim1_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))

        # Color stimget
        stim2_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim3_locs = (stim2_locs+np.pi)%(2*np.pi)

        # Time of stimgets on/off
        stim1_ons  = int(rng.uniform(100,600)/dt)
        fix_offs  = stim1_ons + int(1000/dt)

        tdim = fix_offs + int(500/dt)

    elif mode == 'sample':
        batch_size = 1

        stim1_locs = np.array([1.25*np.pi])
        stim2_locs = np.array([0.5*np.pi])
        stim3_locs = np.array([1.5*np.pi])

        stim1_ons  = int(500/dt)
        fix_offs  = stim1_ons + int(1000/dt)
        tdim = fix_offs + int(500/dt)

    elif mode == 'test':
        a = 2**(BS_EXPO-1)
        batch_size = 2**BS_EXPO
        stim1_locs = np.concatenate(((0.1+0.8*np.arange(a)/a),(1.1+0.8*np.arange(a)/a)))*np.pi
        stim2_locs = stim1_locs
        stim3_locs = (stim2_locs+np.pi)%(2*np.pi)

        stim1_ons  = int(500/dt)
        fix_offs  = stim1_ons + int(1000/dt)
        tdim = fix_offs + int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        stim3_locs = p['stim3_locs']
        batch_size = len(stim1_locs)

        stim1_ons  = int(500/dt)
        fix_offs  = stim1_ons + int(1000/dt)
        tdim = fix_offs + int(500/dt)

    # time to check the saccade location
    check_ons = fix_offs + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimget 1

    task = Task(config, tdim, batch_size)

    task.add('fix_in')
    task.add('stim_mod1', stim1_locs, ons=stim1_ons)
    task.add('stim_mod2', stim2_locs, ons=fix_offs)
    task.add('stim_mod3', stim3_locs, ons=fix_offs)

    # Target location
    stim_locs = list()
    for i in range(batch_size):
        if stim1_cats[i] == 0:
            stim_locs.append(stim2_locs[i])
        else:
            stim_locs.append(stim3_locs[i])

    task.add('fix_out', offs=fix_offs)
    task.add('out', stim_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def delaymatchcategory_original(config, mode, **kwargs):
    '''
    Delay-match-to-category.
    Tailored to the Freedman experiment. Notably some intervals are fixed during training

    Two or three stimuli are shown in ring 1, separated in time, either at the locations of the same category or not
    Fixate before the second stimulus is shown

    If the two stimuli are different, then keep fixation.
    If the two stimuli are match, then saccade to the location of the stimulus

    The first stimget is shown between (stim1_on, stim1_off)
    The second stimget is shown between (stim2_on, T)

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length

        # Use only ring 1 for stimulus input to be consistent with OIC
        stim1_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))
        stim2_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))

        # Time of stimgets on/off
        stim1_ons  = int(rng.uniform(100,600)/dt)
        stim1_offs = stim1_ons + int(1000/dt)
        stim2_ons  = stim1_offs + int(1000/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'sample':
        batch_size = 1

        stim1_locs = np.array([1.25*np.pi])
        stim2_locs = np.array([0.25*np.pi])
        stim1_ons = int(500/dt)
        stim1_offs = stim1_ons + int(1000/dt)
        stim2_ons = stim1_offs + int(1000/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'test':
        # Set this test so the model always respond
        a = 2**(BS_EXPO-1)
        batch_size = 2**BS_EXPO
        stim1_locs = np.concatenate(((0.1+0.8*np.arange(a)/a),(1.1+0.8*np.arange(a)/a)))*np.pi
        stim2_locs = stim1_locs
        stim1_ons  = int(500/dt)
        stim1_offs = stim1_ons + int(1000/dt)
        stim2_ons  = stim1_offs + int(rng.uniform(800,1200)/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim1_locs = p['stim1_locs']
        stim2_locs = p['stim2_locs']
        batch_size = len(stim1_locs)

        tdim = int(3000/dt)
        stim1_ons  = int(500/dt)
        stim1_offs = int(1500/dt)
        stim2_ons  = int(2500/dt)

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimget 1
    stim2_cats = stim2_locs<np.pi # Category of stimget 2
    matchs    = stim1_cats==stim2_cats

    task = Task(config, tdim, batch_size)

    task.add('fix_in')
    task.add('stim_mod1', stim1_locs, ons=stim1_ons, offs=stim1_offs)
    task.add('stim_mod1', stim2_locs, ons=stim2_ons)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == 0: # If non-match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimget location


    task.add('fix_out', offs=fix_out_offs)
    task.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    task.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return task

def timedgo(config, mode, **kwargs):
    '''
    Fixation point is always on
    Saccade to the stimget location after a fixed interval following the stimget onset
    Generate one batch of trials

    The fixation is shown between (0, T)
    The stimget is shown between (stim_on, T)
    The time difference between stim_on and the supposed saccade time sac_on is fixed:
    sac_on = stim_on + 500

    The output should be fixation location for (0, sac_on)
    Otherwise should be the stimget location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    timed_interval = 1000

    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # Fixation and stimget locations
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        # Target onset and fixation offset
        stim_ons  = (rng.uniform(200,600)*np.ones(batch_size)/dt).astype(int)
        sac_ons  = stim_ons + int(timed_interval/dt)

        # each batch consists of sequences of equal length
        tdim = max(sac_ons) + int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        stim_ons   = [int(200/dt)]
        sac_ons   = np.array([int((200+timed_interval)/dt)])
        stim_locs  = [1.5*np.pi]
        batch_size = 1

    elif mode == 'test':
        tdim = int(2000/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        stim_ons   = int(500/dt)
        sac_ons   = int(1000/dt)
        stim_locs  = 2*np.pi*np.arange(a)/a

    # time to check the saccade location
    check_ons  = sac_ons + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('stim', stim_locs, ons=stim_ons)
    task.add('fix_out', offs=sac_ons)
    task.add('out', stim_locs, ons=sac_ons)
    task.add_c_mask(pre_offs=sac_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, sac_ons),
                   'go1'      : (sac_ons, None)}

    return task

def delaytimedgo(config, mode, **kwargs):
    '''
    Fixation point is always on
    Saccade to the stimget location after a fixed interval following the stimget offset
    Generate one batch of trials

    The fixation is shown between (0, T)
    The stimget is shown between (stim_on, stim_off)
    The time difference between stim_off and the supposed saccade time sac_on is fixed:
    sac_on = stim_off + 500

    The output should be fixation location for (0, sac_on)
    Otherwise should be the stimget location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    timed_interval = 1000

    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # Fixation and stimget locations
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        # Target onset and fixation offset
        stim_ons  = (rng.uniform(200,600)*np.ones(batch_size)/dt).astype(int)
        stim_offs = stim_ons + (rng.uniform(200,400)*np.ones(batch_size)/dt).astype(int)
        sac_ons  = stim_offs + int(timed_interval/dt)

        # each batch consists of sequences of equal length
        tdim = max(sac_ons) + int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        stim_ons   = [int(200/dt)]
        stim_offs  = [int(300/dt)]
        sac_ons   = np.array([int((300+timed_interval)/dt)])
        stim_locs  = [1.5*np.pi]
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        sac_ons   = int(2000/dt)
        stim_locs  = 2*np.pi*np.arange(a)/a
        stim_ons   = int(500/dt)
        stim_offs  = int(1000/dt)

    # time to check the saccade location
    check_ons  = sac_ons + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('stim', stim_locs, ons=stim_ons, offs=stim_offs)
    task.add('fix_out', offs=sac_ons)
    task.add('out', stim_locs, ons=sac_ons)
    task.add_c_mask(pre_offs=sac_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, stim_offs),
                   'delay1'   : (stim_offs, sac_ons),
                   'go1'      : (sac_ons, None)}

    return task

def intervalreproduction(config, mode, **kwargs):
    '''
    Reproduce a temporal interval
    A stimulus in mod 2 is shown twice, with a certain time interval Dt
    The model should go to the location of the location of the stimulus in mod 1 after time Dt
    following the second appearance of the stimulus in mod 2

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    rng = config['rng']
    stim1_mod2_ons = int(200/dt)
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        interval  = int(rng.uniform(500,1000)/dt)

        # Location of stimget
        stim_mod1_locs  = rng.rand(batch_size)*2*np.pi

        stim1_mod2_ons  = int(rng.uniform(100,300)/dt)

    elif mode == 'sample':
        batch_size     = 1
        stim_mod1_locs  = np.array([1.0*np.pi])
        interval  = int(750/dt)

    elif mode == 'test':
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        stim_mod1_locs  = 2*np.pi*np.arange(a)/a
        interval  = int(750/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        interval      = int(p['interval']/dt)
        stim_mod1_locs = p['stim_mod1_locs']
        batch_size    = len(stim_mod1_locs)


    stim1_mod2_locs  = (stim_mod1_locs+np.pi)%(2*np.pi)
    stim2_mod2_locs  = (stim_mod1_locs+np.pi/2)%(2*np.pi)
    stim1_mod2_offs = stim1_mod2_ons + int(100/dt)
    stim2_mod2_ons  = stim1_mod2_ons + interval
    stim2_mod2_offs = stim2_mod2_ons + int(100/dt)
    sac_ons        = stim2_mod2_ons + interval
    tdim           = sac_ons + int(400/dt)
    check_ons      = sac_ons + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('stim_mod1', stim_mod1_locs)
    task.add('stim_mod2', stim1_mod2_locs, ons=stim1_mod2_ons, offs=stim1_mod2_offs)
    task.add('stim_mod2', stim2_mod2_locs, ons=stim2_mod2_ons, offs=stim2_mod2_offs)
    task.add('fix_out', offs=sac_ons)
    task.add('out', stim_mod1_locs, ons=sac_ons)
    task.add_c_mask(pre_offs=sac_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, stim1_mod2_ons),
                   'stim1'     : (stim1_mod2_ons, stim1_mod2_offs),
                   'delay1'   : (stim1_mod2_offs, stim2_mod2_ons),
                   'stim2'     : (stim2_mod2_ons, stim2_mod2_offs),
                   'delay2'   : (stim2_mod2_offs, sac_ons),
                   'go1'      : (sac_ons, None)}

    return task


rule_mapping = {TEST_INIT               : test_init,
                FIXATION                : fixation,
                REACTGO                 : go,
                DELAYGO                 : delaygo,
                CHOICE_MOD1             : choicego_mod1,
                CHOICE_MOD2             : choicego_mod2,
                CHOICEATTEND_MOD1       : choicego_attend_mod1,
                CHOICEATTEND_MOD2       : choicego_attend_mod2,
                CHOICE_INT              : choicego_int,
                CHOICEDELAY_MOD1        : choicedelaygo_mod1,
                CHOICEDELAY_MOD2        : choicedelaygo_mod2,
                CHOICEDELAY_MOD1_COPY   : choicedelaygo_mod1,
                CHOICEDELAYATTEND_MOD1  : choicegodelay_attend_mod1,
                CHOICEDELAYATTEND_MOD2  : choicegodelay_attend_mod2,
                CHOICEDELAY_INT         : choicegodelay_int,
                TIMEDGO                 : timedgo,
                REACTANTI               : remapgo,
                DELAYTIMEDGO            : delaytimedgo,
                DELAYANTI               : delayremapgo,
                DMSGO                   : delaymatchsamplego,
                DMSNOGO                 : delaymatchsamplenogo,
                DMCGO                   : delaymatchcategorygo,
                DMCNOGO                 : delaymatchcategorynogo,
                FDGO                    : inhgo,
                FDANTI                  : inhremapgo,
                INTREPRO                : intervalreproduction,
                OIC                     : oic,
                DMC                     : delaymatchcategory_original}

rule_name    = {FIXATION                : 'Fixation',
                REACTGO                 : 'RT Go',
                DELAYGO                 : 'Dly Go',
                FDGO                    : 'Go',
                CHOICE_MOD1             : 'DM 1',
                CHOICE_MOD2             : 'DM 2',
                CHOICEATTEND_MOD1       : 'Ctx DM 1',
                CHOICEATTEND_MOD2       : 'Ctx DM 2',
                CHOICE_INT              : 'MultSen DM',
                CHOICEDELAY_MOD1        : 'Dly DM 1',
                CHOICEDELAY_MOD2        : 'Dly DM 2',
                CHOICEDELAY_MOD1_COPY   : 'Dly DM 1*',
                CHOICEDELAYATTEND_MOD1  : 'Ctx Dly DM 1',
                CHOICEDELAYATTEND_MOD2  : 'Ctx Dly DM 2',
                CHOICEDELAY_INT         : 'MultSen Dly DM',
                TIMEDGO                 : 'Timed Go',
                DELAYTIMEDGO            : 'Timed Delay Go',
                REACTANTI               : 'RT Anti',
                DELAYANTI               : 'Dly Anti',
                FDANTI                  : 'Anti',
                DMSGO                   : 'DMS',
                DMSNOGO                 : 'DNMS',
                DMCGO                   : 'DMC',
                DMCNOGO                 : 'DNMC',
                INTREPRO                : 'Int repro',
                OIC                     : '1IC',
                DMC                     : 'DMC'}

#-----------------------------------------------------------------------------------------
# Rule features
#-----------------------------------------------------------------------------------------


Delay, Decision, InhControl, Anti, Match = features = range(5)

feature_names = {Delay : 'Delay',
                 Decision: 'Decision',
                 InhControl: 'Inh Ctrl',
                 Anti: 'Anti',
                 Match: 'Match'}

rule_features= {REACTGO             : [],
                FDGO               : [InhControl],
                DELAYGO             : [Delay],
                CHOICE_MOD1         : [Decision],
                CHOICE_MOD2         : [Decision],
                CHOICEATTEND_MOD1   : [Decision],
                CHOICEATTEND_MOD2   : [Decision],
                CHOICE_INT          : [Decision],
                CHOICEDELAY_MOD1    : [Delay, Decision],
                CHOICEDELAY_MOD2    : [Delay, Decision],
                REACTANTI               : [Anti],
                FDANTI            : [InhControl, Anti],
                DELAYANTI          : [Delay, Anti],
                DMSGO               : [Delay, Match],
                DMSNOGO             : [Delay, Match],
                DMCGO               : [Delay, Match],
                DMCNOGO             : [Delay, Match]}

def generate_onebatch(rule, config, mode, noise_on=True, **kwargs):
    '''
    Generate one batch of data
    :param rule: the rule for this batch
    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional:
    :param batch_size: Batch size (number of trials)
    :param param: a dictionary of parameters
    :return: dictionary of list of data.
    '''
    dt = config['dt']
    task = rule_mapping[rule](config, mode, **kwargs)
    if mode in ['random', 'sample', 'psychometric'] and noise_on:
        task.add_x_noise(dt)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else: # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else: # default behavior
        rule_off = None
        # rule_off = int(200/dt) #TODO: Study this

    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']

    if rule is TEST_INIT:
        # Add no rule
        return task

    if not hasattr(rule, '__iter__'):
        # rule is not iterable
        # Expand to list
        if 'rule_strength' in kwargs:
            rule_strength = [kwargs['rule_strength']]
        else:
            rule_strength = [1.]
        rule = [rule]

    else:
        if 'rule_strength' in kwargs:
            rule_strength = kwargs['rule_strength']
        else:
            rule_strength = [1.] * len(rule)

    for r, s in zip(rule, rule_strength):
        task.add_rule(r, on=rule_on, off=rule_off, strength=s)

    return task


def get_valid_saveaddons(save_type, save_type_end=None):
    # helper function to get all valid save_addons
    save_addons = list()

    _vars = range(0,1000)
    vars = list()

    for var in _vars:
        save_addon = save_type+'_'+str(var)
        if save_type_end is not None:
            save_addon = save_addon + save_type_end
        fname = 'data/config'+save_addon+'.pkl'
        if os.path.isfile(fname):
            save_addons.append(save_addon)
            vars.append(var)

    return save_addons, vars

