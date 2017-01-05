"""
2016/06/03 Restart, with Blocks

Collections of tasks
"""

from __future__ import division
import numpy as np

#-----------------------------------------------------------------------------------------
# Rules
#-----------------------------------------------------------------------------------------
N_RULE          = 17
# FIXATION, GO, INHGO, DELAYGO,\
# CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
# CHOICEDELAY_MOD1, CHOICEDELAY_MOD2, CHOICEDELAY_MOD1_COPY,\
# TIMEDGO, DELAYTIMEDGO,\
# REMAP, INHREMAP, DELAYREMAP,\
# DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO, INTREPRO = range(N_RULE)

GO, INHGO, DELAYGO,\
CHOICE_MOD1, CHOICE_MOD2, CHOICEATTEND_MOD1, CHOICEATTEND_MOD2, CHOICE_INT,\
CHOICEDELAY_MOD1, CHOICEDELAY_MOD2,\
REMAP, INHREMAP, DELAYREMAP,\
DELAYMATCHGO, DELAYMATCHNOGO, DMCGO, DMCNOGO = range(N_RULE)

CHOICEDELAY_MOD1_COPY = FIXATION = TIMEDGO = DELAYTIMEDGO = INTREPRO = -2 # dummy

TEST_INIT = -1

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

# Time constant
TAU                 = 100 # ms
# Noise level for training
SIGMA               = 0.01

def get_dist(original_dist): # Get the distance in periodic boundary conditions
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

#-----------------------------------------------------------------------------------------
# Tasks
#-----------------------------------------------------------------------------------------

# BS_EXPO = 12 # 2-exponent for the batch-size during testing
BS_EXPO = 6 # 2-exponent for the batch-size during testing

class Task(object):
    def __init__(self, config, tdim, batch_size):
        '''
        Initialize
        :param config: dictionary of configuration
        :param tdim:
        :param batch_size:
        :param info: TEMPORARILY info is provided here
        :return:
        '''
        self.float_type = 'float32' # This should be the default
        self.config     = config
        self.dt         = self.config['dt']
        N_RING          = self.config['N_RING']
        self.slices     = {'fix_in'     : 0,
                           'tar_mod1'   : slice(1, N_RING+1),
                           'tar_mod2'   : slice(N_RING+1, 2*N_RING+1),
                           'rule_start' : 2*N_RING+1,
                           'fix_out'    : 0,
                           'out'        : slice(1, N_RING+1)
                           }
        XDIM            = 1 + 2*N_RING + N_RULE
        YDIM            = 1 + N_RING
        self.pref       = np.arange(0,2*np.pi,2*np.pi/N_RING) # preferences
        self.N_RING     = N_RING

        self.batch_size = batch_size
        self.tdim       = tdim
        self.x          = np.zeros((tdim, batch_size, XDIM), dtype=self.float_type)
        self.y          = np.zeros((tdim, batch_size, YDIM), dtype=self.float_type)
        self.y[:,:,:]   = 0.05
        # y_loc is the target location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc      = -np.ones((tdim, batch_size)      , dtype=self.float_type)
        self.c_mask     = np.zeros((tdim, batch_size, YDIM), dtype=self.float_type)

    def expand(self, var):
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1):
        '''
        Add an input or target output
        locs not needed for fix_in or fix_out loc_type
        '''
        ons         = self.expand(ons)
        offs        = self.expand(offs)
        strengths   = self.expand(strengths)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]:offs[i],i,self.slices[loc_type]] = 1
            elif loc_type in ['tar_mod1','tar_mod2']:
                self.x[ons[i]:offs[i],i,self.slices[loc_type]] += self.add_x_loc(locs[i])*strengths[i]
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                self.y[ons[i]:offs[i],i,self.slices[loc_type]] = 0.8
            elif loc_type == 'out':
                self.y[ons[i]:offs[i],i,self.slices[loc_type]] += self.add_y_loc(locs[i])*strengths[i]
                self.y_loc[ons[i]:offs[i],i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self, dt):
        '''
        Add input noise
        :param sigma:
        :return:
        '''
        self.x += np.random.randn(*self.x.shape)*SIGMA*np.sqrt(2/dt*TAU)

    def add_c_mask_tempdisabled(self, pre_offs, post_ons):
        pre_on   = int(50/self.dt) # never check the first 50ms
        self.c_mask[pre_on:,:,:] = 100

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

        for i in range(self.batch_size):
            # Post response periods usually have the same length across tasks
            self.c_mask[post_ons[i]:, i, :] = 1.
            # Pre-response periods usually have different lengths across tasks
            # To keep cost comparable across tasks
            # Scale the cost mask of the pre-response period by a factor
            self.c_mask[pre_on:pre_offs[i], i, :] = (self.tdim-post_ons[i])/(pre_offs[i]-pre_on)

        #self.c_mask[:, :, 0] *= self.N_RING # Fixation is important
        self.c_mask[:, :, 0] *= 2 # Fixation is important

    def add_rule(self, rule, on=None, off=None):
        self.x[on:off,:,self.slices['rule_start']+rule] = 1. # Have rule input

    def add_x_loc(self, x_loc):
        dist = get_dist(x_loc-self.pref) # periodic boundary
        dist /= np.pi/8
        return 0.8*np.exp(-dist**2/2)

    def add_y_loc(self, y_loc):
        dist = get_dist(y_loc-self.pref) # periodic boundary
        dist /= np.pi/8
        y    = 0.8*np.exp(-dist**2/2)
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
    if mode == 'random':
        batch_size = kwargs['batch_size']
        tdim = int(np.random.uniform(1000,2000)/dt)

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


def go(config, mode, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    A target will be shown once the fixation is off
    And output should saccade to the target location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The target is shown between (fix_off,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim = int(np.random.uniform(1000,2000)/dt)

        # A list of locations of fixation points and fixation off time
        fix_offs = (0.8*np.ones(batch_size)*tdim).astype(int)

        # A list of locations of targets (they are always on)
        tar_locs = np.random.uniform(0, 2*np.pi, (batch_size,))

        tar_mod  = np.random.choice([1,2])

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(0.8*tdim)])
        tar_locs  = [0.2*np.pi]
        tar_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        fix_offs  = int(2000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_mod   = 1

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar_locs, ons=fix_offs)
    task.add('fix_out', offs=fix_offs)
    task.add('out', tar_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def inhgo(config, mode, **kwargs):
    '''
    Go with inhibitory control. Important difference with Go task is that
    the stimulus is presented from the beginning.

    Fixate whenever fixation point is shown,
    A target will be shown from the beginning
    And output should saccade to the target location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The target is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim = int(np.random.uniform(1000,2000)/dt)

        # A list of locations of fixation points and fixation off time
        fix_offs = (0.8*np.ones(batch_size)*tdim).astype(int)

        # A list of locations of targets (they are always on)
        tar_locs = np.random.rand(batch_size)*2*np.pi
        tar_mod  = np.random.choice([1,2])
        tar_ons  = (np.ones(batch_size)*np.random.uniform(100,300)/dt).astype(int)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = [int(0.8*tdim)]
        tar_locs  = [1.5*np.pi]
        tar_ons   = [int(0.1*tdim)]
        tar_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        fix_offs  = int(2000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_ons   = int(500/dt)
        tar_mod   = 1

    # time to check the saccade location
    check_ons  = (0.85*np.ones(batch_size)*tdim).astype(int)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar_locs, ons=tar_ons)
    task.add('fix_out', offs=fix_offs)
    task.add('out', tar_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def choicego_(config, mode, tar_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two targets are shown, saccade to the one with higher intensity
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two targets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger target

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of targets (they are always on)
        tar_dist = np.random.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*np.random.choice([-1,1],(batch_size,))
        tar1_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar2_locs = (tar1_locs+tar_dist)%(2*np.pi)

        # Target strengths
        tars_mean = np.random.uniform(0.8,1.2,(batch_size,))
        # tars_diff = np.random.uniform(0.01,0.2,(batch_size,))
        tars_diff = np.random.choice([0.02, 0.04, 0.08], (batch_size,))
        tars_sign = np.random.choice([1,-1], (batch_size,))

        tar1_strengths = tars_mean + tars_diff*tars_sign/2
        tar2_strengths = tars_mean - tars_diff*tars_sign/2

        # Time of targets on/off
        tar_on = int(np.random.uniform(100,400)/dt)
        tar_ons = (np.ones(batch_size)*tar_on).astype(int)
        # tar_dur = int(np.random.uniform(300,1500)/dt)
        tar_dur = int(np.random.uniform(900, 1500)/dt)
        fix_offs = (tar_ons+tar_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = tar_on+tar_dur+int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(0.75*tdim)])
        tar1_locs = [0.5*np.pi]
        tar2_locs = [1.5*np.pi]
        tar1_strengths = [0.9]
        tar2_strengths = [1.1]
        tar_ons  = np.array([int(0.15*tdim)])
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**(BS_EXPO//3)
        batch_size = 2**BS_EXPO
        ind_tar_loc, ind_tar1_strength, ind_tar2_strength = np.unravel_index(range(batch_size),(a,a,a))
        fix_offs  = int(2000/dt)

        tar1_locs = 2*np.pi*ind_tar_loc/a
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)
        tar1_strengths = 0.4*ind_tar1_strength/a+0.8
        tar2_strengths = 0.4*ind_tar2_strength/a+0.8
        tar_ons  = int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        tar1_locs = p['tar1_locs']
        tar2_locs = p['tar2_locs']
        tar1_strengths = p['tar1_strengths']
        tar2_strengths = p['tar2_strengths']
        tar_time = int(p['tar_time']/dt)
        batch_size = len(tar1_locs)

        # Time of targets on/off
        tar_ons = int(300/dt)
        fix_offs = int(300/dt) + tar_time
        tdim = int(400/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)


    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar1_locs, ons=tar_ons, offs=fix_offs, strengths=tar1_strengths)
    task.add('tar_mod'+str(tar_mod), tar2_locs, ons=tar_ons, offs=fix_offs, strengths=tar2_strengths)
    task.add('fix_out', offs=fix_offs)
    tar_locs = [tar1_locs[i] if (tar1_strengths[i]>tar2_strengths[i])
                else tar2_locs[i] for i in range(batch_size)]
    task.add('out', tar_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicego_mod1(config, mode, **kwargs):
    return choicego_(config, mode, 1, **kwargs)

def choicego_mod2(config, mode, **kwargs):
    return choicego_(config, mode, 2, **kwargs)


def choicego_attend_(config, mode, attend_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two targets are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two targets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger target

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
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of targets, same locations for both modalities
        tar_dist = np.random.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*np.random.choice([-1,1],(batch_size,))
        tar1_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar2_locs = (tar1_locs+tar_dist)%(2*np.pi)

        tars_mod1_mean = np.random.uniform(0.8,1.2,(batch_size,))
        # tars_mod1_diff = np.random.uniform(0.05,0.4,(batch_size,))
        # tars_mod1_diff = np.random.choice([0.02, 0.04, 0.08], (batch_size,))
        tars_mod1_diff = np.random.choice([0.04, 0.08, 0.16], (batch_size,))
        tars_mod1_sign = np.random.choice([1,-1], (batch_size,))

        tar1_mod1_strengths = tars_mod1_mean + tars_mod1_diff*tars_mod1_sign/2
        tar2_mod1_strengths = tars_mod1_mean - tars_mod1_diff*tars_mod1_sign/2

        tars_mod2_mean = np.random.uniform(0.8,1.2,(batch_size,))
        # tars_mod2_diff = np.random.uniform(0.05,0.4,(batch_size,))
        # tars_mod2_diff = np.random.choice([0.02, 0.04, 0.08], (batch_size,))
        tars_mod2_diff = np.random.choice([0.04, 0.08, 0.16], (batch_size,))
        tars_mod2_sign = np.random.choice([1,-1], (batch_size,))

        tar1_mod2_strengths = tars_mod2_mean + tars_mod2_diff*tars_mod2_sign/2
        tar2_mod2_strengths = tars_mod2_mean - tars_mod2_diff*tars_mod2_sign/2

        # Time of targets on/off
        tar_on = int(np.random.uniform(100,400)/dt)
        tar_ons = (np.ones(batch_size)*tar_on).astype(int)
        # tar_dur = int(np.random.uniform(300,1500)/dt)
        tar_dur = int(np.random.uniform(900, 1500)/dt)
        fix_offs = (tar_ons+tar_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = tar_on+tar_dur+int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(0.75*tdim)])
        tar1_locs = [0.5*np.pi]
        tar2_locs = [1.5*np.pi]
        tar1_mod1_strengths = [0.9]
        tar2_mod1_strengths = [1.1]
        tar1_mod2_strengths = [1.1]
        tar2_mod2_strengths = [0.9]
        tar_ons  = np.array([int(0.15*tdim)])
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**(BS_EXPO//3)
        batch_size = 2**BS_EXPO
        ind_tar_loc, ind_tar_mod1_strength, ind_tar_mod2_strength = np.unravel_index(range(batch_size),(a,a,a))
        fix_offs  = int(2000/dt)

        tar1_locs = 2*np.pi*ind_tar_loc/a
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)
        tar1_mod1_strengths = 0.4*ind_tar_mod1_strength/a+0.8
        tar2_mod1_strengths = 2 - tar1_mod1_strengths
        tar1_mod2_strengths = 0.4*ind_tar_mod2_strength/a+0.8
        tar2_mod2_strengths = 2 - tar1_mod2_strengths
        tar_ons  = int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        tar1_locs = p['tar1_locs']
        tar2_locs = p['tar2_locs']
        tar1_mod1_strengths = p['tar1_mod1_strengths']
        tar2_mod1_strengths = p['tar2_mod1_strengths']
        tar1_mod2_strengths = p['tar1_mod2_strengths']
        tar2_mod2_strengths = p['tar2_mod2_strengths']
        tar_time = int(p['tar_time']/dt)
        batch_size = len(tar1_locs)

        # Time of targets on/off
        tar_ons = int(400/dt)
        fix_offs = int(400/dt) + tar_time
        tdim = int(400/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        tar1_strengths, tar2_strengths = tar1_mod1_strengths, tar2_mod1_strengths
    elif attend_mod == 2:
        tar1_strengths, tar2_strengths = tar1_mod2_strengths, tar2_mod2_strengths

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod1', tar1_locs, ons=tar_ons, offs=fix_offs, strengths=tar1_mod1_strengths)
    task.add('tar_mod1', tar2_locs, ons=tar_ons, offs=fix_offs, strengths=tar2_mod1_strengths)
    task.add('tar_mod2', tar1_locs, ons=tar_ons, offs=fix_offs, strengths=tar1_mod2_strengths)
    task.add('tar_mod2', tar2_locs, ons=tar_ons, offs=fix_offs, strengths=tar2_mod2_strengths)
    task.add('fix_out', offs=fix_offs)
    tar_locs = [tar1_locs[i] if (tar1_strengths[i]>tar2_strengths[i])
                else tar2_locs[i] for i in range(batch_size)]
    task.add('out', tar_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicego_attend_mod1(config, mode, **kwargs):
    return choicego_attend_(config, mode, 1, **kwargs)

def choicego_attend_mod2(config, mode, **kwargs):
    return choicego_attend_(config, mode, 2, **kwargs)


def choicego_int(config, mode, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two targets are shown in each ring, pointing to the same direction
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two targets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger target

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of targets, same locations for both modalities
        tar_dist = np.random.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*np.random.choice([-1,1],(batch_size,))
        tar1_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar2_locs = (tar1_locs+tar_dist)%(2*np.pi)

        tars_mod1_mean = np.random.uniform(0.8,1.2,(batch_size,))
        # tars_mod1_diff = np.random.uniform(0.05,0.4,(batch_size,))
        tars_mod1_diff = np.random.choice([0.02, 0.04, 0.08], (batch_size,))
        tars_mod1_sign = np.random.choice([1,-1], (batch_size,))

        tar1_mod1_strengths = tars_mod1_mean + tars_mod1_diff*tars_mod1_sign/2
        tar2_mod1_strengths = tars_mod1_mean - tars_mod1_diff*tars_mod1_sign/2

        # Always the same as mod 1
        tar1_mod2_strengths, tar2_mod2_strengths = tar1_mod1_strengths, tar2_mod1_strengths

        # Time of targets on/off
        tar_on = int(np.random.uniform(100,400)/dt)
        tar_ons = (np.ones(batch_size)*tar_on).astype(int)
        # tar_dur = int(np.random.uniform(300,1500)/dt)
        tar_dur = int(np.random.uniform(900, 1500)/dt)
        fix_offs = (tar_ons+tar_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = tar_on+tar_dur+int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = np.array([int(0.75*tdim)])
        tar1_locs = [0.5*np.pi]
        tar2_locs = [1.5*np.pi]
        tar1_mod1_strengths = [0.95]
        tar2_mod1_strengths = [1.05]
        tar1_mod2_strengths, tar2_mod2_strengths = tar1_mod1_strengths, tar2_mod1_strengths
        tar_ons  = [int(0.15*tdim)]
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**(BS_EXPO//3)
        batch_size = 2**BS_EXPO
        ind_tar_loc, ind_tar1_strength, ind_tar2_strength = np.unravel_index(range(batch_size),(a,a,a))
        fix_offs  = int(2000/dt)

        tar1_locs = 2*np.pi*ind_tar_loc/a
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)
        tar1_mod1_strengths = 0.4*ind_tar1_strength/a+0.8
        tar2_mod1_strengths = 0.4*ind_tar2_strength/a+0.8
        tar1_mod2_strengths, tar2_mod2_strengths = tar1_mod1_strengths, tar2_mod1_strengths
        tar_ons  = int(500/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        tar1_locs = p['tar1_locs']
        tar2_locs = p['tar2_locs']
        tar1_mod1_strengths = p['tar1_mod1_strengths']
        tar2_mod1_strengths = p['tar2_mod1_strengths']
        tar1_mod2_strengths = p['tar1_mod2_strengths']
        tar2_mod2_strengths = p['tar2_mod2_strengths']
        # tar1_mod2_strengths, tar2_mod2_strengths = tar1_mod1_strengths, tar2_mod1_strengths
        tar_time = int(p['tar_time']/dt)
        batch_size = len(tar1_locs)

        # Time of targets on/off
        tar_ons = int(400/dt)
        fix_offs = int(400/dt) + tar_time
        tdim = int(400/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod1', tar1_locs, ons=tar_ons, offs=fix_offs, strengths=tar1_mod1_strengths)
    task.add('tar_mod1', tar2_locs, ons=tar_ons, offs=fix_offs, strengths=tar2_mod1_strengths)
    task.add('tar_mod2', tar1_locs, ons=tar_ons, offs=fix_offs, strengths=tar1_mod2_strengths)
    task.add('tar_mod2', tar2_locs, ons=tar_ons, offs=fix_offs, strengths=tar2_mod2_strengths)
    task.add('fix_out', offs=fix_offs)
    tar_locs = [tar1_locs[i] if (tar1_mod1_strengths[i]>tar2_mod1_strengths[i])
                else tar2_locs[i] for i in range(batch_size)]
    task.add('out', tar_locs, ons=fix_offs)

    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def choicedelaygo_(config, mode, tar_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two targets are shown at different time, with different intensities

    The fixation is shown between (0, fix_off)
    The two targets is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger target

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # A list of locations of targets (they are always on)
        tar_dist = np.random.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*np.random.choice([-1,1],(batch_size,))
        tar1_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar2_locs = (tar1_locs+tar_dist)%(2*np.pi)

        tars_mean = np.random.uniform(0.8,1.2,(batch_size,))
        tars_diff = np.random.uniform(0.3,0.5,(batch_size,))
        tars_sign = np.random.choice([1,-1], (batch_size,))

        tar1_strengths = tars_mean + tars_diff*tars_sign/2
        tar2_strengths = tars_mean - tars_diff*tars_sign/2

        # tar1_strengths = np.random.uniform(0.25,1.75,(batch_size,))
        # tar2_strengths = np.random.uniform(0.25,1.75,(batch_size,))

        # Time of targets on/off
        tar1_ons  = (np.ones(batch_size)*np.random.uniform(100,300)/dt).astype(int)
        tar1_offs = tar1_ons + int(300/dt)
        # tar2_ons  = (np.ones(batch_size)*np.random.choice([400,500,600,700,1400])/dt).astype(int)
        tar2_ons  = (np.ones(batch_size)*np.random.choice([400,600,1000,1400,2000])/dt).astype(int)
        # tar2_ons  = (np.ones(batch_size)*np.random.uniform(2800,3200)/dt).astype(int)
        tar2_offs = tar2_ons + int(300/dt)

        fix_offs  = tar2_offs + int(np.random.uniform(100,300)/dt)

        # each batch consists of sequences of equal length
        tdim = max(fix_offs) + int(300/dt) # longest trial

    elif mode == 'sample':
        tar1_locs = [0.5*np.pi]
        tar2_locs = [1.5*np.pi]
        tar1_strengths = [2.0] # always make tar1 stronger
        tar2_strengths = [0.75]
        tar1_ons = [int(100/dt)]
        tar1_offs = [int(300/dt)]

        tdim = int(3300/dt)
        fix_offs  = np.array([int(2800/dt)])
        tar2_ons = [int(2500/dt)]
        tar2_offs = [int(2700/dt)]
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**(BS_EXPO//3)
        batch_size = 2**BS_EXPO
        ind_tar_loc, ind_tar1_strength, ind_tar2_strength = np.unravel_index(range(batch_size),(a,a,a))
        fix_offs  = int(2000/dt)
        tar1_locs = 2*np.pi*ind_tar_loc/a
        tar2_locs = (tar1_locs+np.pi)%(2*np.pi)
        tar1_strengths = 1.0*ind_tar1_strength/a+0.5
        tar2_strengths = 1.0*ind_tar2_strength/a+0.5
        tar1_ons = int(500/dt)
        tar1_offs = int(800/dt)
        tar2_ons = int(1600/dt)
        tar2_offs = int(1900/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        tar1_locs       = p['tar1_locs']
        tar2_locs       = p['tar2_locs']
        tar1_strengths  = p['tar1_strengths']
        tar2_strengths  = p['tar2_strengths']
        tar1_ons        = int(p['tar1_ons']/dt)
        tar1_offs       = int(p['tar1_offs']/dt)
        tar2_ons        = int(p['tar2_ons']/dt)
        tar2_offs       = int(p['tar2_offs']/dt)
        batch_size = len(tar1_locs)

        fix_offs = int(200/dt) + tar2_offs
        tdim = int(300/dt) + fix_offs

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar1_locs, ons=tar1_ons, offs=tar1_offs, strengths=tar1_strengths)
    task.add('tar_mod'+str(tar_mod), tar2_locs, ons=tar2_ons, offs=tar2_offs, strengths=tar2_strengths)
    task.add('fix_out', offs=fix_offs)
    tar_locs = [tar1_locs[i] if (tar1_strengths[i]>tar2_strengths[i])
                else tar2_locs[i] for i in range(batch_size)]
    task.add('out', tar_locs, ons=fix_offs)


    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar1_ons),
                   'tar1'     : (tar1_ons, tar1_offs),
                   'delay1'   : (tar1_offs, tar2_ons),
                   'tar2'     : (tar2_ons, tar2_offs),
                   'delay2'   : (tar2_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task

def choicedelaygo_mod1(config, mode, **kwargs):
    return choicedelaygo_(config, mode, 1, **kwargs)

def choicedelaygo_mod2(config, mode, **kwargs):
    return choicedelaygo_(config, mode, 2, **kwargs)


def delaygo(config, mode, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    saccade to the location of the previously shown target
    whenever the fixation point is off
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The target is shown between (tar_on, tar_off)

    The output should be fixation location for (0, fix_off)
    and the target location for (fix_off, T)

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim = int(np.random.choice([1600,1800,2000,3000])/dt)

        # A list of locations of fixation points and fixation off time
        fix_offs = tdim - (np.ones(batch_size)*500/dt).astype(int)

        # A list of locations of targets and on/off time
        tar_locs = np.random.rand(batch_size)*2*np.pi
        tar_ons  = (np.random.uniform(100,300,(batch_size,))/dt).astype(int)
        tar_offs = tar_ons + int(200/dt)
        tar_mod  = np.random.choice([1,2])

    elif mode == 'sample':
        tdim = int(2000/dt)
        fix_offs  = np.array([int(1800/dt)])
        tar_locs  = [1.5*np.pi]
        tar_ons   = [int(200/dt)]
        tar_offs  = [int(400/dt)]
        tar_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        fix_offs  = int(2000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_ons   = int(500/dt)
        tar_mod   = 1
        tar_offs  = int(700/dt)

    check_ons= fix_offs + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar_locs, ons=tar_ons, offs=tar_offs)
    task.add('fix_out', offs=fix_offs)
    task.add('out', tar_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, tar_offs),
                   'delay1'   : (tar_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def delaytimedgo(config, mode, **kwargs):
    '''
    Fixation point is always on
    Saccade to the target location after a fixed interval following the target offset
    Generate one batch of trials

    The fixation is shown between (0, T)
    The target is shown between (tar_on, tar_off)
    The time difference between tar_off and the supposed saccade time sac_on is fixed:
    sac_on = tar_off + 500

    The output should be fixation location for (0, sac_on)
    Otherwise should be the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    timed_interval = 1000

    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # Fixation and target locations
        tar_locs = np.random.uniform(0, 2*np.pi, (batch_size,))

        # Target onset and fixation offset
        tar_ons  = (np.random.uniform(200,600)*np.ones(batch_size)/dt).astype(int)
        tar_offs = tar_ons + (np.random.uniform(200,400)*np.ones(batch_size)/dt).astype(int)
        sac_ons  = tar_offs + int(timed_interval/dt)

        # each batch consists of sequences of equal length
        tdim = max(sac_ons) + int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        tar_ons   = [int(200/dt)]
        tar_offs  = [int(300/dt)]
        sac_ons   = np.array([int((300+timed_interval)/dt)])
        tar_locs  = [1.5*np.pi]
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        sac_ons   = int(2000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_ons   = int(500/dt)
        tar_offs  = int(1000/dt)

    # time to check the saccade location
    check_ons  = sac_ons + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('tar', tar_locs, ons=tar_ons, offs=tar_offs)
    task.add('fix_out', offs=sac_ons)
    task.add('out', tar_locs, ons=sac_ons)
    task.add_c_mask(pre_offs=sac_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, tar_offs),
                   'delay1'   : (tar_offs, sac_ons),
                   'go1'      : (sac_ons, None)}

    return task


def delayremapgo(config, mode, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    A target is shown before the fixation is off
    And output should move away from the target location (remap)
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The target is shown between (tar_on, tar_off)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the anti location of the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim = int(np.random.uniform(1000,2000)/dt)

        # A list of locations of fixation points and fixation off time
        fix_offs = (0.8*np.ones(batch_size)*tdim).astype(int)

        # A list of locations of target on and off time
        tar_ons  = (0.1*np.ones(batch_size)*tdim).astype(int)
        tar_offs = (0.3*np.ones(batch_size)*tdim).astype(int)

        # A list of locations of targets (they are always on)
        tar_locs      = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar_mod  = np.random.choice([1,2])

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = [int(0.8*tdim)]
        tar_ons   = [int(0.1*tdim)]
        tar_offs  = [int(0.3*tdim)]
        tar_locs  = np.array([1.5*np.pi])
        tar_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        fix_offs  = int(2000/dt)
        tar_ons   = int(500/dt)
        tar_offs  = int(800/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_mod   = 1

    # time to check the saccade location
    check_ons  = (0.85*np.ones(batch_size)*tdim).astype(int)

    tar_anti_locs = (tar_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar_locs, ons=tar_ons, offs=tar_offs)
    task.add('fix_out', offs=fix_offs)
    task.add('out', tar_anti_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, tar_offs),
                   'delay1'   : (tar_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def timedgo(config, mode, **kwargs):
    '''
    Fixation point is always on
    Saccade to the target location after a fixed interval following the target onset
    Generate one batch of trials

    The fixation is shown between (0, T)
    The target is shown between (tar_on, T)
    The time difference between tar_on and the supposed saccade time sac_on is fixed:
    sac_on = tar_on + 500

    The output should be fixation location for (0, sac_on)
    Otherwise should be the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    timed_interval = 1000

    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']

        # Fixation and target locations
        tar_locs = np.random.uniform(0, 2*np.pi, (batch_size,))

        # Target onset and fixation offset
        tar_ons  = (np.random.uniform(200,600)*np.ones(batch_size)/dt).astype(int)
        sac_ons  = tar_ons + int(timed_interval/dt)

        # each batch consists of sequences of equal length
        tdim = max(sac_ons) + int(500/dt)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        tar_ons   = [int(200/dt)]
        sac_ons   = np.array([int((200+timed_interval)/dt)])
        tar_locs  = [1.5*np.pi]
        batch_size = 1

    elif mode == 'test':
        tdim = int(2000/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        tar_ons   = int(500/dt)
        sac_ons   = int(1000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a

    # time to check the saccade location
    check_ons  = sac_ons + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('tar', tar_locs, ons=tar_ons)
    task.add('fix_out', offs=sac_ons)
    task.add('out', tar_locs, ons=sac_ons)
    task.add_c_mask(pre_offs=sac_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, sac_ons),
                   'go1'      : (sac_ons, None)}

    return task


def remapgo(config, mode, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    A target will be shown once the fixation is off
    And output should saccade away from the target location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The target is shown between (fix_off,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the anti location of the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim = int(np.random.uniform(1000,2000)/dt)

        # A list of locations of fixation points and fixation off time
        fix_offs = (0.8*np.ones(batch_size)*tdim).astype(int)

        # A list of locations of targets (they are always on)
        tar_locs      = np.random.uniform(0, 2*np.pi, (batch_size,))

        tar_mod  = np.random.choice([1,2])

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = [int(0.8*tdim)]
        tar_locs  = np.array([1.5*np.pi])
        tar_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        fix_offs  = int(2000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_mod   = 1

    # time to check the saccade location
    check_ons  = (0.85*np.ones(batch_size)*tdim).astype(int)

    tar_anti_locs = (tar_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar_locs, ons=fix_offs)
    task.add('fix_out', offs=fix_offs)
    task.add('out', tar_anti_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def inhremapgo(config, mode, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    A target will be shown from the beginning
    And output should saccade away from the target location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The target is shown between (0, T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the anti location of the target location

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim = int(np.random.uniform(1000,2000)/dt)

        # A list of locations of fixation points and fixation off time
        fix_offs = (0.8*np.ones(batch_size)*tdim).astype(int)

        # A list of locations of targets (they are always on)
        tar_locs      = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar_mod  = np.random.choice([1,2])

        tar_ons  = (np.ones(batch_size)*np.random.uniform(100,300)/dt).astype(int)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        fix_offs  = [int(0.8*tdim)]
        tar_locs  = np.array([1.5*np.pi])
        tar_ons   = [int(0.1*tdim)]
        tar_mod   = 1
        batch_size = 1

    elif mode == 'test':
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        fix_offs  = int(2000/dt)
        tar_locs  = 2*np.pi*np.arange(a)/a
        tar_ons   = int(500/dt)
        tar_mod   = 1

    # time to check the saccade location
    check_ons  = (0.85*np.ones(batch_size)*tdim).astype(int)

    tar_anti_locs = (tar_locs+np.pi)%(2*np.pi)

    task = Task(config, tdim, batch_size)
    task.add('fix_in', offs=fix_offs)
    task.add('tar_mod'+str(tar_mod), tar_locs, ons=tar_ons)
    task.add('fix_out', offs=fix_offs)
    task.add('out', tar_anti_locs, ons=fix_offs)
    task.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar_ons),
                   'tar1'     : (tar_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return task


def delaymatch_(config, mode, matchnogo, **kwargs):
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

    The first target is shown between (tar1_on, tar1_off)
    The second target is shown between (tar2_on, T)

    The output should be fixation location for (0, tar2_on)
    If two stimuli the different location, then for (tar2_on, T) go to tar2_loc
    Otherwise keep fixation

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        tdim      = int(np.random.choice([1400, 1500, 1600, 2200])/dt)

        tar1_mod  = np.random.choice([1,2])
        tar2_mod  = np.random.choice([1,2])
        # A list of locations of targets
        # Since tar1 is always shown first, it's important that we completely randomize their relative positions
        matchs    = np.random.choice([0,1],(batch_size,)) # match or not?
        tar_dist  = np.random.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*np.random.choice([-1,1],(batch_size,))
        tar1_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar2_locs = (tar1_locs+tar_dist*(1-matchs))%(2*np.pi)

        # Time of targets on/off
        tar1_ons  = (np.ones(batch_size)*np.random.uniform(100,300)/dt).astype(int)
        tar1_offs = tar1_ons + int(200/dt)
        tar2_ons  = tdim - (np.ones(batch_size)*400/dt).astype(int)

    elif mode == 'sample':
        tdim = int(kwargs['t_tot']/dt)
        tar1_mod = 1
        tar2_mod = 1
        matchs    = np.array([0])
        tar_dist = 0.5*np.pi
        tar1_locs = [0.5*np.pi]
        tar2_locs = [(0.5*np.pi+tar_dist*(1-matchs))%(2*np.pi)]
        tar1_ons = np.array([int(100/dt)])
        tar1_offs = [int(200/dt)]
        tar2_ons = [int(700/dt)]
        batch_size = 1

    elif mode == 'test':
        # Set this test so the model always respond
        tdim = int(2500/dt)
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        tar1_locs = 2*np.pi*np.arange(a)/a
        matchs = (1 - matchnogo)*np.ones(batch_size) # make sure the response is Go
        tar2_locs = (tar1_locs+np.pi*(1-matchs))%(2*np.pi)
        tar1_ons  = int(500/dt)
        tar1_offs = int(800/dt)
        tar2_ons  = int(2000/dt)
        tar1_mod = 1
        tar2_mod = 1

    # time to check the saccade location
    check_ons = tar2_ons + int(100/dt)

    task = Task(config, tdim, batch_size)

    task.add('fix_in')
    task.add('tar_mod'+str(tar1_mod), tar1_locs, ons=tar1_ons, offs=tar1_offs)
    task.add('tar_mod'+str(tar2_mod), tar2_locs, ons=tar2_ons)

    if hasattr(tar2_ons, '__iter__'):
        fix_out_offs = list(tar2_ons)
    else:
        fix_out_offs = [tar2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to target location


    task.add('fix_out', offs=fix_out_offs)
    task.add('out', tar2_locs, ons=tar2_ons, offs=out_offs)

    task.add_c_mask(pre_offs=tar2_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar1_ons),
                   'tar1'     : (tar1_ons, tar1_offs),
                   'delay1'   : (tar1_offs, tar2_ons),
                   'go1'      : (tar2_ons, None)}

    return task


def delaymatchgo(config, mode, **kwargs):
    return delaymatch_(config, mode, 0, **kwargs)

def delaymatchnogo(config, mode, **kwargs):
    return delaymatch_(config, mode, 1, **kwargs)

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
    tar1_mod2_ons = int(200/dt)
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length
        interval  = int(np.random.uniform(500,1000)/dt)

        # Location of target
        tar_mod1_locs  = np.random.rand(batch_size)*2*np.pi

        tar1_mod2_ons  = int(np.random.uniform(100,300)/dt)

    elif mode == 'sample':
        batch_size     = 1
        tar_mod1_locs  = np.array([1.0*np.pi])
        interval  = int(750/dt)

    elif mode == 'test':
        a = 2**BS_EXPO
        batch_size = 2**BS_EXPO
        tar_mod1_locs  = 2*np.pi*np.arange(a)/a
        interval  = int(750/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        interval      = int(p['interval']/dt)
        tar_mod1_locs = p['tar_mod1_locs']
        batch_size    = len(tar_mod1_locs)


    tar1_mod2_locs  = (tar_mod1_locs+np.pi)%(2*np.pi)
    tar2_mod2_locs  = (tar_mod1_locs+np.pi/2)%(2*np.pi)
    tar1_mod2_offs = tar1_mod2_ons + int(100/dt)
    tar2_mod2_ons  = tar1_mod2_ons + interval
    tar2_mod2_offs = tar2_mod2_ons + int(100/dt)
    sac_ons        = tar2_mod2_ons + interval
    tdim           = sac_ons + int(400/dt)
    check_ons      = sac_ons + int(100/dt)

    task = Task(config, tdim, batch_size)
    task.add('fix_in')
    task.add('tar_mod1', tar_mod1_locs)
    task.add('tar_mod2', tar1_mod2_locs, ons=tar1_mod2_ons, offs=tar1_mod2_offs)
    task.add('tar_mod2', tar2_mod2_locs, ons=tar2_mod2_ons, offs=tar2_mod2_offs)
    task.add('fix_out', offs=sac_ons)
    task.add('out', tar_mod1_locs, ons=sac_ons)
    task.add_c_mask(pre_offs=sac_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar1_mod2_ons),
                   'tar1'     : (tar1_mod2_ons, tar1_mod2_offs),
                   'delay1'   : (tar1_mod2_offs, tar2_mod2_ons),
                   'tar2'     : (tar2_mod2_ons, tar2_mod2_offs),
                   'delay2'   : (tar2_mod2_offs, sac_ons),
                   'go1'      : (sac_ons, None)}

    return task


def delaymatchcategory_(config, mode, matchnogo, **kwargs):
    '''
    Delay-match-to-category
    Tailored to the Freedman experiment. Notably every interval is fixed during training

    Two stimuli are shown, separated in time, either at the locations of the same category or not
    Fixate before the second stimulus is shown

    If matchnogo is one, then:
    If the two stimuli are the same, then keep fixation.
    If the two stimuli are different, then saccade to the location of the stimulus

    If matchnogo is zero, then:
    If the two stimuli are different, then keep fixation.
    If the two stimuli are the same, then saccade to the location of the stimulus

    The first target is shown between (tar1_on, tar1_off)
    The second target is shown between (tar2_on, T)

    The output should be fixation location for (0, tar2_on)
    If two stimuli the different location, then for (tar2_on, T) go to tar2_loc
    Otherwise keep fixation

    :param mode: the mode of generating. Options: 'random', 'sample', 'explicit'...
    Optional parameters:
    :param batch_size: Batch size (required for mode=='random')
    :param tdim: dimension of time (required for mode=='sample')
    :param param: a dictionary of parameters (required for mode=='explicit')
    :return: 2 Tensor3 data array (Time, Batchsize, Units)
    '''
    dt = config['dt']
    if mode == 'random': # Randomly generate parameters
        batch_size = kwargs['batch_size']
        # each batch consists of sequences of equal length

        # Use only mod 1 for input
        tar1_mod  = 1
        tar2_mod  = 1
        # A list of locations of targets
        # Since tar1 is always shown first, it's important that we completely randomize their relative positions
        # tar1_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        # tar2_locs = np.random.uniform(0, 2*np.pi, (batch_size,))
        tar1_locs = np.random.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))
        tar2_locs = np.random.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))
        # Time of targets on/off
        tar1_ons  = int(np.random.uniform(100,600)/dt)
        tar1_offs = tar1_ons + int(np.random.uniform(100,600)/dt)
        tar2_ons  = tar1_offs + int(np.random.uniform(500,1500)/dt)

        tdim = tar2_ons + int(500/dt)

    elif mode == 'sample':
        tar1_mod = 1
        tar2_mod = 1
        tar1_locs = np.array([1.25*np.pi])
        tar2_locs = np.array([0.25*np.pi])
        tar1_ons = np.array([int(300/dt)])
        tar1_offs = tar1_ons + int(200/dt)
        tar2_ons = tar1_offs + int(1000/dt)
        batch_size = 1
        tdim = tar2_ons[0] + int(500/dt)

    elif mode == 'psychometric':
        tar1_mod = 1
        tar2_mod = 1

        p = kwargs['params']
        tar1_locs = p['tar1_locs']
        tar2_locs = p['tar2_locs']
        tar1_ons  = int(300/dt) #TODO: Temporarily setting this high
        tar1_offs = tar1_ons + int(300/dt)
        tar2_ons  = tar1_offs + int(1000/dt)
        batch_size = len(tar1_locs)

        tdim = tar2_ons + int(500/dt)

    elif mode == 'test':
        # Set this test so the model always respond
        a = 2**(BS_EXPO-1)
        batch_size = 2**BS_EXPO
        #tar1_locs = 2*np.pi*np.arange(a)/a ##TODO: Change it so it doesn't include boundary case!
        tar1_locs = np.concatenate(((0.1+0.8*np.arange(a)/a),(1.1+0.8*np.arange(a)/a)))*np.pi
        matchs = (1 - matchnogo)*np.ones(batch_size) # make sure the response is Go
        tar2_locs = (tar1_locs+np.pi*(1-matchs))%(2*np.pi)
        tar1_ons  = int(500/dt)
        tar1_offs = tar1_ons + int(300/dt)
        tar2_ons  = tar1_offs + int(1200/dt)
        tar1_mod = 1
        tar2_mod = 1
        tdim = tar2_ons + int(500/dt)

    # time to check the saccade location
    check_ons = tar2_ons + int(100/dt)

    tar1_cats = tar1_locs<np.pi # Category of target 1
    tar2_cats = tar2_locs<np.pi # Category of target 2
    matchs    = tar1_cats==tar2_cats

    task = Task(config, tdim, batch_size)

    task.add('fix_in')
    task.add('tar_mod'+str(tar1_mod), tar1_locs, ons=tar1_ons, offs=tar1_offs)
    task.add('tar_mod'+str(tar2_mod), tar2_locs, ons=tar2_ons)

    if hasattr(tar2_ons, '__iter__'):
        fix_out_offs = list(tar2_ons)
    else:
        fix_out_offs = [tar2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to target location


    task.add('fix_out', offs=fix_out_offs)
    task.add('out', tar2_locs, ons=tar2_ons, offs=out_offs)

    task.add_c_mask(pre_offs=tar2_ons, post_ons=check_ons)

    task.epochs = {'fix1'     : (None, tar1_ons),
                   'tar1'     : (tar1_ons, tar1_offs),
                   'delay1'   : (tar1_offs, tar2_ons),
                   'go1'      : (tar2_ons, None)}

    return task

def delaymatchcategorygo(config, mode, **kwargs):
    return delaymatchcategory_(config, mode, 0, **kwargs)

def delaymatchcategorynogo(config, mode, **kwargs):
    return delaymatchcategory_(config, mode, 1, **kwargs)


rule_mapping = {TEST_INIT               : test_init,
                FIXATION                : fixation,
                GO                      : go,
                DELAYGO                 : delaygo,
                CHOICE_MOD1             : choicego_mod1,
                CHOICE_MOD2             : choicego_mod2,
                CHOICEDELAY_MOD1        : choicedelaygo_mod1,
                CHOICEDELAY_MOD2        : choicedelaygo_mod2,
                CHOICEDELAY_MOD1_COPY   : choicedelaygo_mod1,
                CHOICEATTEND_MOD1       : choicego_attend_mod1,
                CHOICEATTEND_MOD2       : choicego_attend_mod2,
                CHOICE_INT              : choicego_int,
                TIMEDGO                 : timedgo,
                REMAP                   : remapgo,
                DELAYTIMEDGO            : delaytimedgo,
                DELAYREMAP              : delayremapgo,
                DELAYMATCHGO            : delaymatchgo,
                DELAYMATCHNOGO          : delaymatchnogo,
                DMCGO                   : delaymatchcategorygo,
                DMCNOGO                 : delaymatchcategorynogo,
                INHGO                   : inhgo,
                INHREMAP                : inhremapgo,
                INTREPRO                : intervalreproduction}

rule_name    = {FIXATION                : 'Fixation',
                GO                      : 'Go',
                DELAYGO                 : 'Del Go',
                CHOICE_MOD1             : 'Choice 1',
                CHOICE_MOD2             : 'Choice 2',
                CHOICEATTEND_MOD1       : 'Choice Att 1',
                CHOICEATTEND_MOD2       : 'Choice Att 2',
                CHOICE_INT              : 'Choice Int',
                CHOICEDELAY_MOD1        : 'Del Choice 1',
                CHOICEDELAY_MOD2        : 'Del Choice 2',
                CHOICEDELAY_MOD1_COPY   : 'Del Choice 1*',
                TIMEDGO                 : 'Timed Go',
                DELAYTIMEDGO            : 'Timed Delay Go',
                INHGO                   : 'Inh Go',
                REMAP                   : 'Remap',
                DELAYREMAP              : 'Del Remap',
                INHREMAP                : 'Inh Remap',
                DELAYMATCHGO            : 'Del Match Go',
                DELAYMATCHNOGO          : 'Del Match NoGo',
                DMCGO                   : 'DMC Go',
                DMCNOGO                 : 'DMC NoGo',
                INTREPRO                : 'Int repro'}

#-----------------------------------------------------------------------------------------
# Rule features
#-----------------------------------------------------------------------------------------

# Mod1, Mod2, Inh, Choice, Delay, Timed, Remap, Match = range(8)
#
# features = [Mod1, Mod2, Inh, Choice, Delay, Timed, Remap, Match]
#
# feature_names = {Mod1   : 'Mod 1',
#                  Mod2   : 'Mod 2',
#                  Inh    : 'Inh. Control',
#                  Choice : 'Choice',
#                  Delay  : 'Delay',
#                  Timed  : 'Timed',
#                  Remap  : 'Remap',
#                  Match  : 'Match'}
#
# rule_features= {FIXATION            : [],
#                 GO                  : [Mod1, Mod2],
#                 DELAYGO             : [Mod1, Mod2, Inh, Delay],
#                 CHOICE_MOD1         : [Mod1, Inh, Choice],
#                 CHOICE_MOD2         : [Mod2, Inh, Choice],
#                 CHOICEATTEND_MOD1   : [Mod1, Mod2, Inh, Choice],
#                 CHOICEATTEND_MOD2   : [Mod1, Mod2, Inh, Choice],
#                 CHOICE_INT          : [Mod1, Mod2, Inh, Choice],
#                 CHOICEDELAY_MOD1    : [Mod1, Inh, Choice, Delay],
#                 CHOICEDELAY_MOD1_COPY : [Mod1, Inh, Choice, Delay],
#                 CHOICEDELAY_MOD2    : [Mod2, Inh, Choice, Delay],
#                 TIMEDGO             : [Mod1, Mod2, Inh, Timed],
#                 DELAYTIMEDGO        : [Mod1, Mod2, Inh, Timed, Delay],
#                 INHGO               : [Mod1, Mod2, Inh],
#                 REMAP               : [Mod1, Mod2, Remap],
#                 DELAYREMAP          : [Mod1, Mod2, Inh, Delay, Remap],
#                 INHREMAP            : [Mod1, Mod2, Inh, Remap],
#                 DELAYMATCHGO        : [Mod1, Mod2, Inh, Match, Delay],
#                 DELAYMATCHNOGO      : [Mod1, Mod2, Inh, Match, Delay],
#                 INTREPRO            : [Mod1, Mod2, Inh, Timed, Delay]}

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

    if 'add_rule' in kwargs:
        rule = kwargs['add_rule'] # overwrite current rule for input

    if rule is not TEST_INIT:
        task.add_rule(rule, on=rule_on, off=rule_off)

    return task

