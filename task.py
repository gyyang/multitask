"""Collections of tasks."""

from __future__ import division
import six
import numpy as np


# all rules
rules_dict = \
    {'all' : ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
              'dm1', 'dm2', 'contextdm1', 'contextdm2', 'multidm',
              'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
              'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo'],

    'mante' : ['contextdm1', 'contextdm2'],

    'oicdmc' : ['oic', 'dmc']}

# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind


def get_num_ring(ruleset):
    '''get number of stimulus rings'''
    return 3 if ruleset=='oicdmc' else 2


def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])


def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule]+config['rule_start']


def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))


class Trial(object):
    """Class representing a batch of trials."""

    def __init__(self, config, tdim, batch_size):
        """A batch of trials.

        Args:
            config: dictionary of configurations
            tdim: int, number of time steps
            batch_size: int, batch size
        """
        self.float_type = 'float32' # This should be the default
        self.config = config
        self.dt = self.config['dt']

        self.n_eachring = self.config['n_eachring']
        self.n_input = self.config['n_input']
        self.n_output = self.config['n_output']
        self.pref  = np.arange(0,2*np.pi,2*np.pi/self.n_eachring) # preferences

        self.batch_size = batch_size
        self.tdim = tdim
        self.x = np.zeros((tdim, batch_size, self.n_input), dtype=self.float_type)
        self.y = np.zeros((tdim, batch_size, self.n_output), dtype=self.float_type)
        if self.config['loss_type'] == 'lsq':
            self.y[:,:,:] = 0.05
        # y_loc is the stimulus location of the output, -1 for fixation, (0,2 pi) for response
        self.y_loc = -np.ones((tdim, batch_size)      , dtype=self.float_type)

        self._sigma_x = config['sigma_x']*np.sqrt(2/config['alpha'])

    def expand(self, var):
        """Expand an int/float to list."""
        if not hasattr(var, '__iter__'):
            var = [var] * self.batch_size
        return var

    def add(self, loc_type, locs=None, ons=None, offs=None, strengths=1, mods=None):
        """Add an input or stimulus output.

        Args:
            loc_type: str (fix_in, stim, fix_out, out), type of information to be added
            locs: array of list of float (batch_size,), locations to be added, only for loc_type=stim or out
            ons: int or list, index of onset time
            offs: int or list, index of offset time
            strengths: float or list, strength of input or target output
            mods: int or list, modalities of input or target output
        """

        ons = self.expand(ons)
        offs = self.expand(offs)
        strengths = self.expand(strengths)
        mods = self.expand(mods)

        for i in range(self.batch_size):
            if loc_type == 'fix_in':
                self.x[ons[i]: offs[i], i, 0] = 1
            elif loc_type == 'stim':
                # Assuming that mods[i] starts from 1
                self.x[ons[i]: offs[i], i, 1+(mods[i]-1)*self.n_eachring:1+mods[i]*self.n_eachring] \
                    += self.add_x_loc(locs[i])*strengths[i]
            elif loc_type == 'fix_out':
                # Notice this shouldn't be set at 1, because the output is logistic and saturates at 1
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 0] = 0.8
                else:
                    self.y[ons[i]: offs[i], i, 0] = 1.0
            elif loc_type == 'out':
                if self.config['loss_type'] == 'lsq':
                    self.y[ons[i]: offs[i], i, 1:] += self.add_y_loc(locs[i])*strengths[i]
                else:
                    y_tmp = self.add_y_loc(locs[i])
                    y_tmp /= np.sum(y_tmp)
                    self.y[ons[i]: offs[i], i, 1:] += y_tmp
                self.y_loc[ons[i]: offs[i], i] = locs[i]
            else:
                raise ValueError('Unknown loc_type')

    def add_x_noise(self):
        """Add input noise."""
        self.x += self.config['rng'].randn(*self.x.shape)*self._sigma_x

    def add_c_mask(self, pre_offs, post_ons):
        """Add a cost mask.

        Usually there are two periods, pre and post response
        Scale the mask weight for the post period so in total it's as important
        as the pre period
        """

        pre_on   = int(100/self.dt) # never check the first 100ms
        pre_offs = self.expand(pre_offs)
        post_ons = self.expand(post_ons)

        if self.config['loss_type'] == 'lsq':
            c_mask = np.zeros((self.tdim, self.batch_size, self.n_output), dtype=self.float_type)
            for i in range(self.batch_size):
                # Post response periods usually have the same length across tasks
                c_mask[post_ons[i]:, i, :] = 5.
                # Pre-response periods usually have different lengths across tasks
                # To keep cost comparable across tasks
                # Scale the cost mask of the pre-response period by a factor
                c_mask[pre_on:pre_offs[i], i, :] = 1.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2. # Fixation is important

            self.c_mask = c_mask.reshape((self.tdim*self.batch_size, self.n_output))
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
        """Add rule input."""
        if isinstance(rule, int):
            self.x[on:off, :, self.config['rule_start']+rule] = strength
        else:
            ind_rule = get_rule_index(rule, self.config)
            self.x[on:off, :, ind_rule] = strength

    def add_x_loc(self, x_loc):
        """Input activity given location."""
        dist = get_dist(x_loc-self.pref)  # periodic boundary
        dist /= np.pi/8
        return 0.8*np.exp(-dist**2/2)

    def add_y_loc(self, y_loc):
        """Target response given location."""
        dist = get_dist(y_loc-self.pref)  # periodic boundary
        if self.config['loss_type'] == 'lsq':
            dist /= np.pi/8
            y = 0.8*np.exp(-dist**2/2)
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

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)

    return trial


def delaygo_(config, mode, anti_response, **kwargs):
    '''
    Fixate whenever fixation point is shown,
    saccade to the location of the previously shown stimulus
    whenever the fixation point is off
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (stim_on, stim_off)

    The output should be fixation location for (0, fix_off)
    and the stimulus location for (fix_off, T)

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # A list of locations of stimuluss and on/off time
        stim_locs = rng.rand(batch_size)*2*np.pi
        # stim_ons  = int(500/dt)
        stim_ons  = int(rng.choice([300, 500, 700])/dt)
        # stim_offs = stim_ons + int(200/dt)
        stim_offs = stim_ons + int(rng.choice([200, 400, 600])/dt)
        fix_offs = stim_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        # fix_offs = stim_offs + int(rng.choice([1600])/dt)
        tdim     = fix_offs + int(500/dt)
        stim_mod  = rng.choice([1,2])

    elif mode == 'test':
        tdim = int(2500/dt)
        n_stim_loc, n_stim_mod = batch_shape = 20, 2
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod = np.unravel_index(range(batch_size),batch_shape)

        fix_offs  = int(2000/dt)
        stim_locs  = 2*np.pi*ind_stim_loc/n_stim_loc
        stim_ons   = int(500/dt)
        stim_mod   = ind_stim_mod + 1
        stim_offs  = int(1000/dt)

    elif mode == 'psychometric':
        p = kwargs['params']
        stim_locs = p['stim_locs']
        # Time of stimuluss on/off
        stim_ons    = int(p['stim_ons']/dt)
        stim_offs   = int(p['stim_offs']/dt)
        delay_time = int(p['delay_time']/dt)
        fix_offs   = stim_offs + delay_time
        tdim       = int(400/dt) + fix_offs
        stim_mod    = 1

        batch_size = len(stim_locs)

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    check_ons= fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, offs=stim_offs, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, stim_offs),
                   'delay1'   : (stim_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def delaygo(config, mode, **kwargs):
    return delaygo_(config, mode, False, **kwargs)


def contextdm_genstim(batch_size, rng, stim_coh_range=None):
    stim_mean = rng.uniform(0.8, 1.2, (batch_size,))
    if stim_coh_range is None:
        stim_coh_range = np.array([0.16, 0.32, 0.64])*1.0
    stim_coh  = rng.choice(stim_coh_range, (batch_size,))
    stim_sign = rng.choice([+1, -1], (batch_size,))
    stim1_strengths = stim_mean + stim_coh*stim_sign
    stim2_strengths = stim_mean - stim_coh*stim_sign
    return stim1_strengths, stim2_strengths


def _contextdm(config, mode, attend_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimuluss are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    In this task, if the model's strategy is to ignore context, and integrate both,
    then the maximum performance is 75%. So we need to make the highest correct performance
    much higher than that.

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # A list of locations of stimuluss, same locations for both modalities
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stim_coh_range = np.array([0.01, 0.02, 0.04, 0.08])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])
            stim_coh_range *= 10

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = contextdm_genstim(batch_size, rng, stim_coh_range)

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

        # Time of stimuluss on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        # stim_dur = rng.choice((np.array([200, 400, 800, 1600])/dt).astype(int)) # Current setting
        # stim_dur = int(rng.uniform(500, 1000)/dt) # Current setting
        # stim_dur = int(800/dt)
        stim_offs = stim_ons+stim_dur

        # delay_dur = rng.choice((np.array([200, 400, 800])/dt).astype(int)) # Current setting
        delay_dur = 0
        fix_offs  = stim_offs + delay_dur

        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+delay_dur+int(500/dt)

    elif mode == 'test':
        tdim = int(2000/dt)
        n_stim_loc, n_stim_mod1_strength, n_stim_mod2_strength = batch_shape = 20, 5, 5
        batch_size = np.prod(batch_shape)
        ind_stim_loc, ind_stim_mod1_strength, ind_stim_mod2_strength = np.unravel_index(range(batch_size),batch_shape)
        fix_offs  = int(1500/dt)

        stim1_locs = 2*np.pi*ind_stim_loc/n_stim_loc
        stim2_locs = (stim1_locs+np.pi)%(2*np.pi)
        stim1_mod1_strengths = 0.4*ind_stim_mod1_strength/n_stim_mod1_strength+0.8
        stim2_mod1_strengths = 2 - stim1_mod1_strengths
        stim1_mod2_strengths = 0.4*ind_stim_mod2_strength/n_stim_mod2_strength+0.8
        stim2_mod2_strengths = 2 - stim1_mod2_strengths
        stim_ons  = int(500/dt)
        stim_offs = int(1500/dt)

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

        # Time of stimuluss on/off
        stim_ons = int(500/dt)
        stim_offs = stim_ons + stim_time
        fix_offs = stim_offs
        tdim = int(500/dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=stim_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=stim_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'    : (stim_ons, stim_offs),
                   # 'delay1'   : (stim_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def contextdm1(config, mode, **kwargs):
    return _contextdm(config, mode, 1, **kwargs)


def contextdm2(config, mode, **kwargs):
    return _contextdm(config, mode, 2, **kwargs)


def multidm(config, mode, **kwargs):
    return _contextdm(config, mode, 'both', **kwargs)


def reactgo_(config, mode, anti_response, **kwargs):
    '''
    Fixate when fixation point is shown,
    A stimulus will be shown, and the output should saccade to the stimulus location
    Generate one batch of trials

    The fixation is shown between (0, T)
    The stimulus is shown between (fix_off,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimulus location

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # A list of locations of stimuluss (they are always on)
        stim_locs = rng.uniform(0, 2*np.pi, (batch_size,))

        stim_mod  = rng.choice([1,2])

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

        # Time of stimuluss on/off
        stim_ons = int(1000/dt)
        tdim = int(400/dt) + stim_ons
        stim_mod   = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = stim_ons + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in')
    trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    trial.add('fix_out', offs=stim_ons)
    trial.add('out', response_locs, ons=stim_ons)
    trial.add_c_mask(pre_offs=stim_ons, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'go1'      : (stim_ons, None)}

    return trial


def reactgo(config, mode, **kwargs):
    return reactgo_(config, mode, False, **kwargs)


def reactanti(config, mode, **kwargs):
    return reactgo_(config, mode, True, **kwargs)


def fdgo_(config, mode, anti_response, **kwargs):
    '''
    Go with inhibitory control. Important difference with Go task is that
    the stimulus is presented from the beginning.

    Fixate whenever fixation point is shown,
    A stimulus will be shown from the beginning
    And output should saccade to the stimulus location
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The stimulus is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise should be the stimulus location

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # Time of stimuluss on/off
        stim_ons   = int(300/dt)
        fix_offs  = stim_ons + stim_time
        tdim      = int(400/dt) + fix_offs
        stim_mod   = 1

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    # Response locations
    stim_locs = np.array(stim_locs)
    if not anti_response:
        response_locs = stim_locs
    else:
        response_locs = (stim_locs+np.pi)%(2*np.pi)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim_locs, ons=stim_ons, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    trial.add('out', response_locs, ons=fix_offs)
    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def fdgo(config, mode, **kwargs):
    return fdgo_(config, mode, False, **kwargs)


def fdanti(config, mode, **kwargs):
    return fdgo_(config, mode, True, **kwargs)


def delayanti(config, mode, **kwargs):
    return delaygo_(config, mode, True, **kwargs)


def _dm(config, mode, stim_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimuluss are shown, saccade to the one with higher intensity
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # A list of locations of stimuluss (they are always on)
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
            # stim_coh_range = np.array([0.1, 0.2, 0.4, 0.8])
            stim_coh_range *= 10

        stims_coh  = rng.choice(stim_coh_range, (batch_size,))
        stims_sign = rng.choice([1,-1], (batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign

        # Time of stimuluss on/off
        stim_on = int(rng.uniform(100,400)/dt)
        stim_ons = (np.ones(batch_size)*stim_on).astype(int)
        # stim_dur = int(rng.uniform(300,1500)/dt)
        stim_dur = int(rng.choice([400, 800, 1600])/dt)
        fix_offs = (stim_ons+stim_dur).astype(int)
        # each batch consists of sequences of equal length
        tdim = stim_on+stim_dur+int(500/dt)

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

        # Time of stimuluss on/off
        stim_ons = int(300/dt)
        fix_offs = int(300/dt) + stim_time
        tdim = int(400/dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)


    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim_ons, offs=fix_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim_ons, offs=fix_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim_ons),
                   'stim1'     : (stim_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def dm1(config, mode, **kwargs):
    return _dm(config, mode, 1, **kwargs)


def dm2(config, mode, **kwargs):
    return _dm(config, mode, 2, **kwargs)


def _delaydm(config, mode, stim_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimuluss are shown at different time, with different intensities

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # A list of locations of stimuluss (they are always on)
        stim_dist = rng.uniform(0.5*np.pi, 1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stims_mean = rng.uniform(0.8,1.2,(batch_size,))
        # stims_diff = rng.choice([0.32,0.64,1.28],(batch_size,))

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.16,0.32,0.64])
            stim_coh_range *= 2

        stims_coh  = rng.choice(stim_coh_range,(batch_size,))
        stims_sign = rng.choice([1,-1], (batch_size,))

        stim1_strengths = stims_mean + stims_coh*stims_sign
        stim2_strengths = stims_mean - stims_coh*stims_sign

        # stim1_strengths = rng.uniform(0.25,1.75,(batch_size,))
        # stim2_strengths = rng.uniform(0.25,1.75,(batch_size,))

        # Time of stimuluss on/off
        stim1_ons  = int(rng.choice([200, 400, 600])/dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600])/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        stim2_offs = stim2_ons + int(rng.choice([200, 400, 600])/dt)
        fix_offs  = stim2_offs + int(rng.uniform(100,300)/dt)

        # stim2_ons  = (np.ones(batch_size)*rng.choice([400,500,600,700,1400])/dt).astype(int)
        # stim2_ons  = (np.ones(batch_size)*rng.choice([400,600,1000,1400,2000])/dt).astype(int)
        # stim2_ons  = (np.ones(batch_size)*rng.uniform(2800,3200)/dt).astype(int)

        # each batch consists of sequences of equal length
        tdim = fix_offs + int(500/dt) # longest trial

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

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_strengths, mods=stim_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_strengths, mods=stim_mod)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)


    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'stim2'     : (stim2_ons, stim2_offs),
                   'delay2'   : (stim2_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def delaydm1(config, mode, **kwargs):
    return _delaydm(config, mode, 1, **kwargs)


def delaydm2(config, mode, **kwargs):
    return _delaydm(config, mode, 2, **kwargs)


def _contextdelaydm(config, mode, attend_mod, **kwargs):
    '''
    Fixate whenever fixation point is shown.
    Two stimuluss are shown in each ring,
    Saccade to the one with higher intensity for the attended ring
    Generate one batch of trials

    The fixation is shown between (0, fix_off)
    The two stimuluss is shown between (0,T)

    The output should be fixation location for (0, fix_off)
    Otherwise the location of the stronger stimulus

    In this task, if the model's strategy is to ignore context, and integrate both,
    then the maximum performance is 75%. So we need to make the highest correct performance
    much higher than that.

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # A list of locations of stimuluss, same locations for both modalities
        stim_dist = rng.uniform(0.5*np.pi,1.5*np.pi,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist)%(2*np.pi)

        stim_coh_range = np.array([0.08,0.16,0.32])
        if ('easy_task' in config) and config['easy_task']:
            # stim_coh_range = np.array([0.16, 0.32, 0.64])
            stim_coh_range *= 2

        if (attend_mod == 1) or (attend_mod == 2):
            stim1_mod1_strengths, stim2_mod1_strengths = \
                contextdm_genstim(batch_size, rng, stim_coh_range)
            stim1_mod2_strengths, stim2_mod2_strengths = \
                contextdm_genstim(batch_size, rng, stim_coh_range)
            if attend_mod == 1:
                stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
            else:
                stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
        else:
            stim1_strengths, stim2_strengths = \
                contextdm_genstim(batch_size, rng, stim_coh_range)

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

        # Time of stimuluss on/off
        stim1_ons  = int(rng.choice([200, 400, 600])/dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600])/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        stim2_offs = stim2_ons + int(rng.choice([200, 400, 600])/dt)
        fix_offs  = stim2_offs + int(rng.uniform(100,300)/dt)

        # each batch consists of sequences of equal length
        tdim = fix_offs + int(500/dt) # longest trial

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
        stim2_offs = int(2500/dt)
        fix_offs  = int(3000/dt)
        tdim = int(3500/dt)

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

        # Time of stimuluss on/off
        fix_offs = int(200/dt) + stim2_offs
        tdim = int(300/dt) + fix_offs

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons  = fix_offs + int(100/dt)

    if attend_mod == 1:
        stim1_strengths, stim2_strengths = stim1_mod1_strengths, stim2_mod1_strengths
    elif attend_mod == 2:
        stim1_strengths, stim2_strengths = stim1_mod2_strengths, stim2_mod2_strengths
    elif attend_mod == 'both':
        stim1_strengths = stim1_mod1_strengths + stim1_mod2_strengths
        stim2_strengths = stim2_mod1_strengths + stim2_mod2_strengths

    trial = Trial(config, tdim, batch_size)
    trial.add('fix_in', offs=fix_offs)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod1_strengths, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod1_strengths, mods=1)
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, strengths=stim1_mod2_strengths, mods=2)
    trial.add('stim', stim2_locs, ons=stim2_ons, offs=stim2_offs, strengths=stim2_mod2_strengths, mods=2)
    trial.add('fix_out', offs=fix_offs)
    stim_locs = [stim1_locs[i] if (stim1_strengths[i]>stim2_strengths[i])
                else stim2_locs[i] for i in range(batch_size)]
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'stim2'     : (stim2_ons, stim2_offs),
                   'delay2'   : (stim2_offs, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def contextdelaydm1(config, mode, **kwargs):
    return _contextdelaydm(config, mode, 1, **kwargs)


def contextdelaydm2(config, mode, **kwargs):
    return _contextdelaydm(config, mode, 2, **kwargs)


def multidelaydm(config, mode, **kwargs):
    return _contextdelaydm(config, mode, 'both', **kwargs)


def dms_(config, mode, matchnogo, **kwargs):
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

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stim2_on, T)

    The output should be fixation location for (0, stim2_on)
    If two stimuli the different location, then for (stim2_on, T) go to stim2_loc
    Otherwise keep fixation

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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
        # A list of locations of stimuluss
        # Since stim1 is always shown first, it's important that we completely randomize their relative positions
        matchs    = rng.choice([0,1],(batch_size,)) # match or not?
        # stim_dist range between 1/18*pi and (2-1/18*pi), corresponding to 10 degree to 350 degree
        stim_dist  = rng.uniform(np.pi/9,np.pi*17./9.,(batch_size,))*rng.choice([-1,1],(batch_size,))
        stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim2_locs = (stim1_locs+stim_dist*(1-matchs))%(2*np.pi)

        # Time of stimuluss on/off
        stim1_ons  = int(rng.choice([200, 400, 600])/dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600])/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        tdim       = stim2_ons + int(500/dt)

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

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimulus location


    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return trial


def dmsgo(config, mode, **kwargs):
    return dms_(config, mode, 0, **kwargs)


def dmsnogo(config, mode, **kwargs):
    return dms_(config, mode, 1, **kwargs)


def dmc_(config, mode, matchnogo, **kwargs):
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

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stim2_on, T)

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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
        # A list of locations of stimuluss
        # Since stim1 is always shown first, it's important that we completely randomize their relative positions
        # stim1_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        # stim2_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim1_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))
        stim2_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))

        # Time of stimuluss on/off
        stim1_ons  = int(rng.choice([200, 400, 600])/dt)
        stim1_offs = stim1_ons + int(rng.choice([200, 400, 600])/dt)
        stim2_ons  = stim1_offs + int(rng.choice([200, 400, 800, 1600])/dt)
        tdim       = stim2_ons + int(rng.choice([200, 400, 600])/dt)

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

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    stim2_cats = stim2_locs<np.pi # Category of stimulus 2
    matchs    = stim1_cats==stim2_cats

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=stim1_mod)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=stim2_mod)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == matchnogo: # If match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimulus location


    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return trial


def dmcgo(config, mode, **kwargs):
    return dmc_(config, mode, 0, **kwargs)


def dmcnogo(config, mode, **kwargs):
    return dmc_(config, mode, 1, **kwargs)


def oic(config, mode, **kwargs):
    '''
    One-interval categorization

    One stimuli is shown in ring 1 for 1000ms,
    then two stimuluss are shown in rings 2 and 3.
    If the stimulus is category 1, then go to the location of ring 2, otherwise ring 3

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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
        # A list of locations of stimuluss
        stim1_locs = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9])*np.pi,size=(batch_size,))

        # Color stimulus
        stim2_locs = rng.uniform(0, 2*np.pi, (batch_size,))
        stim3_locs = (stim2_locs+np.pi)%(2*np.pi)

        # Time of stimuluss on/off
        stim1_ons  = int(rng.uniform(100,600)/dt)
        fix_offs  = stim1_ons + int(1000/dt)

        tdim = fix_offs + int(500/dt)

    elif mode == 'test':
        batch_size = a = 128
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

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = fix_offs + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, mods=1)
    trial.add('stim', stim2_locs, ons=fix_offs, mods=2)
    trial.add('stim', stim3_locs, ons=fix_offs, mods=3)

    # Target location
    stim_locs = list()
    for i in range(batch_size):
        if stim1_cats[i] == 0:
            stim_locs.append(stim2_locs[i])
        else:
            stim_locs.append(stim3_locs[i])

    trial.add('fix_out', offs=fix_offs)
    trial.add('out', stim_locs, ons=fix_offs)

    trial.add_c_mask(pre_offs=fix_offs, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, fix_offs),
                   'go1'      : (fix_offs, None)}

    return trial


def delaymatchcategory_original(config, mode, **kwargs):
    '''
    Delay-match-to-category.
    Tailored to the Freedman experiment. Notably some intervals are fixed during training

    Two or three stimuli are shown in ring 1, separated in time, either at the locations of the same category or not
    Fixate before the second stimulus is shown

    If the two stimuli are different, then keep fixation.
    If the two stimuli are match, then saccade to the location of the stimulus

    The first stimulus is shown between (stim1_on, stim1_off)
    The second stimulus is shown between (stim2_on, T)

    :param mode: the mode of generating. Options: 'random', 'explicit'...
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

        # Time of stimuluss on/off
        stim1_ons  = int(rng.uniform(100,600)/dt)
        stim1_offs = stim1_ons + int(1000/dt)
        stim2_ons  = stim1_offs + int(1000/dt)
        tdim = stim2_ons + int(500/dt)

    elif mode == 'test':
        # Set this test so the model always respond
        batch_size = a = 128
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

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    # time to check the saccade location
    check_ons = stim2_ons + int(100/dt)

    stim1_cats = stim1_locs<np.pi # Category of stimulus 1
    stim2_cats = stim2_locs<np.pi # Category of stimulus 2
    matchs    = stim1_cats==stim2_cats

    trial = Trial(config, tdim, batch_size)

    trial.add('fix_in')
    trial.add('stim', stim1_locs, ons=stim1_ons, offs=stim1_offs, mods=1)
    trial.add('stim', stim2_locs, ons=stim2_ons, mods=1)

    if hasattr(stim2_ons, '__iter__'):
        fix_out_offs = list(stim2_ons)
    else:
        fix_out_offs = [stim2_ons]*batch_size
    out_offs = [None]*batch_size

    for i in range(batch_size):
        if matchs[i] == 0: # If non-match
            fix_out_offs[i] = None # Keep fixation
            out_offs[i] = 0 # And don't go to stimulus location


    trial.add('fix_out', offs=fix_out_offs)
    trial.add('out', stim2_locs, ons=stim2_ons, offs=out_offs)

    trial.add_c_mask(pre_offs=stim2_ons, post_ons=check_ons)

    trial.epochs = {'fix1'     : (None, stim1_ons),
                   'stim1'     : (stim1_ons, stim1_offs),
                   'delay1'   : (stim1_offs, stim2_ons),
                   'go1'      : (stim2_ons, None)}

    return trial


rule_mapping = {'testinit': test_init,
                'fdgo': fdgo,
                'reactgo': reactgo,
                'delaygo': delaygo,
                'fdanti': fdanti,
                'reactanti': reactanti,
                'delayanti': delayanti,
                'dm1': dm1,
                'dm2': dm2,
                'contextdm1': contextdm1,
                'contextdm2': contextdm2,
                'multidm': multidm,
                'delaydm1': delaydm1,
                'delaydm2': delaydm2,
                'contextdelaydm1': contextdelaydm1,
                'contextdelaydm2': contextdelaydm2,
                'multidelaydm': multidelaydm,
                'dmsgo': dmsgo,
                'dmsnogo': dmsnogo,
                'dmcgo': dmcgo,
                'dmcnogo': dmcnogo,
                'oic': oic,
                'dmc': delaymatchcategory_original}

rule_name    = {'reactgo': 'RT Go',
                'delaygo': 'Dly Go',
                'fdgo': 'Go',
                'dm1': 'DM 1',
                'dm2': 'DM 2',
                'contextdm1': 'Ctx DM 1',
                'contextdm2': 'Ctx DM 2',
                'multidm': 'MultSen DM',
                'delaydm1': 'Dly DM 1',
                'delaydm2': 'Dly DM 2',
                'contextdelaydm1': 'Ctx Dly DM 1',
                'contextdelaydm2': 'Ctx Dly DM 2',
                'multidelaydm': 'MultSen Dly DM',
                'reactanti': 'RT Anti',
                'delayanti': 'Dly Anti',
                'fdanti': 'Anti',
                'dmsgo': 'DMS',
                'dmsnogo': 'DNMS',
                'dmcgo': 'DMC',
                'dmcnogo': 'DNMC',
                'oic': '1IC',
                'dmc': 'DMC'
                }


def generate_trials(rule, hp, mode, noise_on=True, **kwargs):
    """Generate one batch of data.

    Args:
        rule: str, the rule for this batch
        hp: dictionary of hyperparameters
        mode: str, the mode of generating. Options: random, test, psychometric
        noise_on: bool, whether input noise is given

    Return:
        trial: Trial class instance, containing input and target output
    """
    config = hp
    trial = rule_mapping[rule](config, mode, **kwargs)

    # Add rule input to every task
    if 'rule_on' in kwargs:
        rule_on = kwargs['rule_on']
    else: # default behavior
        rule_on = None
    if 'rule_off' in kwargs:
        rule_off = kwargs['rule_off']
    else: # default behavior
        rule_off = None

    # overwrite current rule for input
    if 'replace_rule' in kwargs:
        rule = kwargs['replace_rule']

    if rule is 'testinit':
        # Add no rule
        return trial

    if isinstance(rule, six.string_types):
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
        trial.add_rule(r, on=rule_on, off=rule_off, strength=s)

    if noise_on:
        trial.add_x_noise()

    return trial
