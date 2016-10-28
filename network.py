"""
2016/06/03 Restart, with Blocks

Main training loop and network structure
"""

from __future__ import division

import numpy as np
import theano
import theano.tensor as T

from blocks.bricks import Linear, Logistic, Tanh, Rectifier, Initializable, Softplus, Feedforward
from blocks.bricks.base import application, lazy, Brick
from blocks.bricks.interfaces import Activation
from blocks.bricks.recurrent import (BaseRecurrent, SimpleRecurrent,
                                     LSTM, GatedRecurrent, recurrent)
from blocks.bricks.cost import Cost, CostMatrix
from blocks.initialization import Constant, IsotropicGaussian, Identity
from blocks.roles import add_role, WEIGHT, BIAS, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from task import *

class MaskedSquaredError(CostMatrix):
    """Base class for costs which can be calculated element-wise.

    Assumes that the data has format (time, batch, units).
    """
    @application(outputs=["cost"])
    def apply(self, *args, **kwargs):
        return self.cost_matrix(*args, **kwargs).sum(axis=2).mean()

    @application
    def cost_matrix(self, y, y_hat, c_mask=None):
        if c_mask is None:
            return T.sqr(y - y_hat)
        else:
            return T.sqr((y - y_hat)*c_mask)

class MaskedPerformance(Cost):
    def __init__(self, thres, **kwargs):
        super(MaskedPerformance, self).__init__(**kwargs)
        self.thres = theano.shared(thres)

    @application(outputs=["cost"])
    def apply(self, y_loc, y_hat_loc, c_mask):
        #return T.any((abs(y_loc-y_hat_loc)>self.thres)*c_mask[:,:,0], axis=0).mean()
        return ((abs(y_loc-y_hat_loc)>self.thres)*c_mask[:,:,0]).mean()

class SimplePerformance(Cost):
    def __init__(self, thres, **kwargs):
        super(SimplePerformance, self).__init__(**kwargs)
        self.thres = theano.shared(thres)

    @application(outputs=["cost"])
    def apply(self, y_loc, y_fix_out, y_hat_loc):
        '''
        :param y_loc: Location of output ring
        :param y_fix_out: Value of fixation output
        :param y_hat_loc: Intended location (-1 if should fixate)
        :return:
        '''
        original_dist = y_loc-y_hat_loc
        dist = T.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
        loc_correct = dist<self.thres
        fix_correct = y_fix_out > 0.5 # True if fixating
        should_fix = y_hat_loc < -0.5 # if y_hat_loc == -1 should fixate
        performance = T.switch(should_fix, fix_correct, loc_correct)
        return performance.mean()

class PopVec(Brick):
    def __init__(self, input_dim, **kwargs):
        super(PopVec, self).__init__(**kwargs)
        pref = np.arange(0, 2*np.pi, 2*np.pi/input_dim) # preferences
        self.vec_cos = theano.shared(np.cos(pref))
        self.vec_sin = theano.shared(np.sin(pref))

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        # Population vector read out
        y_sumunit = input_.sum(axis=-1)
        temp_cos = T.sum(input_*self.vec_cos, axis=-1)/y_sumunit
        temp_sin = T.sum(input_*self.vec_sin, axis=-1)/y_sumunit
        y_loc0 = T.arctan2(temp_sin, temp_cos)
        y_loc = T.mod(y_loc0, theano.shared(2*np.pi))
        return y_loc

class ChanceAbbott(Activation):
    r""" ChanceAbbott brick.

    The chance & abbott function is defined as :math:`\zeta(x) = (x-x0)/(1-exp(-g(x-x0)))`.
    The chosen parameters here makes it very similar to the one used for E neurons
    in Wong & Wang 2006 J Neurosci

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.nnet.softplus(40.*(input_-0.4))


class LeakyRecurrent(BaseRecurrent, Initializable):
    """The leaky version of Simple Recurrent

    Input is a matrix multiplication, optionally followed by a non-linearity.
    State is updated through leaky integration

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, alpha=1, **kwargs):
        self.dim = dim
        self.alpha = alpha
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(LeakyRecurrent, self).__init__(**kwargs)


    @property
    def W(self):
        return self.parameters[0]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (LeakyRecurrent.apply.sequences +
                    LeakyRecurrent.apply.states):
            return self.dim
        return super(LeakyRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name="W"))
        add_role(self.parameters[0], WEIGHT)
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                                   name="initial_state"))
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs, states, mask=None):
        """Apply the leaky transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        next_states = ((1-self.alpha) * states + self.alpha *
                       self.children[0].apply(inputs + T.dot(states, self.W)))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return T.repeat(self.parameters[1][None, :], batch_size, 0)


class LeakyEIRecurrent(BaseRecurrent, Initializable):
    """The leaky version of EI Recurrent Network

    Input is a matrix multiplication, optionally followed by a non-linearity.
    State is updated through leaky integration
    Network has EI structure. Imposed by relu on weight and sign matrix on states

    Parameters
    ----------
    dim : int
        The dimension of the hidden state
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation, alpha=1, **kwargs):
        self.dim = dim
        self.dimE = int(dim*0.8) # Excitatory neurons always 80%
        self.dimI = self.dim-self.dimE # Inhibitory neurons
        self.signs = theano.shared(np.array([1.]*self.dimE+[-1.]*self.dimI).astype(theano.config.floatX))
        self.eps = theano.shared(1e-7).astype(theano.config.floatX) # small constant
        self.alpha = alpha
        children = [activation]
        kwargs.setdefault('children', []).extend(children)
        super(LeakyEIRecurrent, self).__init__(**kwargs)

    @property
    def W(self):
        return self.parameters[0]

    def W_ei(self):
        return self.signs*abs(self.W+self.eps)

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in (LeakyEIRecurrent.apply.sequences +
                    LeakyEIRecurrent.apply.states):
            return self.dim
        return super(LeakyEIRecurrent, self).get_dim(name)

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.dim, self.dim),
                                                  name="W"))
        add_role(self.parameters[0], WEIGHT)
        self.parameters.append(shared_floatx_zeros((self.dim,),
                                                   name="initial_state"))
        add_role(self.parameters[1], INITIAL_STATE)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @recurrent(sequences=['inputs', 'mask'], states=['states'],
               outputs=['states'], contexts=[])
    def apply(self, inputs, states, mask=None):
        """Apply the leaky transition.

        Parameters
        ----------
        inputs : :class:`~tensor.TensorVariable`
            The 2D inputs, in the shape (batch, features).
        states : :class:`~tensor.TensorVariable`
            The 2D states, in the shape (batch, features).
        mask : :class:`~tensor.TensorVariable`
            A 1D binary array in the shape (batch,) which is 1 if
            there is data available, 0 if not. Assumed to be 1-s
            only if not given.

        """
        next_states = ((1-self.alpha) * states + self.alpha *
                       self.children[0].apply(
                        inputs + T.dot(states*self.signs, abs(self.W+self.eps))))
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
        return next_states

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return T.repeat(self.parameters[1][None, :], batch_size, 0)

class MyNetwork(Initializable):
    """The top brick.

    It is often convenient to gather all bricks of the model under the
    roof of a single top brick.

    """
    def __init__(self, config, **kwargs):
        super(MyNetwork, self).__init__(**kwargs)

        x_dim, h_dim, y_dim = config['shape']
        self.config = config

        # Std is chosen to be small to not saturate the output units
        h_to_o = Linear(h_dim, y_dim, name='h_to_o',
                        weights_init=IsotropicGaussian(std=0.4/np.sqrt(h_dim)),
                        biases_init=Constant(0.0))

        if config['h_type'] == 'lstm':
            x_to_h = Linear(x_dim, h_dim * 4, name='x_to_h',
                            weights_init=IsotropicGaussian(),biases_init=Constant(0.0))
            h_layer = LSTM(h_dim, name='h_rec',
                           weights_init=IsotropicGaussian(),biases_init=Constant(0.0))

        elif config['h_type'] == 'simple_rec':
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(),biases_init=Constant(0.0))
            h_layer = SimpleRecurrent(dim=h_dim, name='h_rec', activation=Softplus(),
                                      weights_init=Identity())

        elif config['h_type'] == 'leaky_rec':
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(std=1.0/np.sqrt(x_dim)),biases_init=Constant(0.0))
            h_layer = LeakyRecurrent(dim=h_dim, name='h_rec', alpha=config['alpha'],
                                     activation=Softplus(), weights_init=Identity(mult=0.54))
            # Weight initialization is set so r=1 is a fixed point for the self-recurrent connection

        elif config['h_type'] == 'leaky_rec_ca': # Chance & Abbott
            '''
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(std=0.02/np.sqrt(x_dim)),biases_init=Constant(0.4))
            h_layer = LeakyRecurrent(dim=h_dim, name='h_rec', alpha=config['alpha'],
                                     activation=ChanceAbbott(), weights_init=Identity(mult=0.54/50.))
            # Weight initialization is set so r=1 is a fixed point for the self-recurrent connection

            h_to_o = Linear(h_dim, y_dim, name='h_to_o',
                        weights_init=IsotropicGaussian(std=0.4/np.sqrt(h_dim)),
                        biases_init=Constant(0.0))
            '''
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(std=1.0/40./np.sqrt(x_dim)),biases_init=Constant(0.4))
            h_layer = LeakyRecurrent(dim=h_dim, name='h_rec', alpha=config['alpha'],
                                     activation=Softplus(), weights_init=Identity(mult=0.54/40.))

        elif config['h_type'] == 'leaky_rec_ei':
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(std=1.0/np.sqrt(x_dim)),biases_init=Constant(0.0))
            h_layer = LeakyEIRecurrent(dim=h_dim, name='h_rec', alpha=config['alpha'],
                                     activation=Softplus(), weights_init=Identity(mult=0.54)) #TODO: Still need to be tuned

        elif config['h_type'] == 'gru':
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(),biases_init=Constant(0.0))
            x_to_g = Linear(x_dim, 2*h_dim, name='x_to_g',
                            weights_init=IsotropicGaussian(),biases_init=Constant(0.0))
            h_layer = GatedRecurrent(dim=h_dim, name='h_rec', activation=Tanh(),
                                     gate_activation=Logistic(),
                                     weights_init=IsotropicGaussian())

        elif config['h_type'] == 'mlp':
            x_to_h = Linear(x_dim, h_dim, name='x_to_h',
                            weights_init=IsotropicGaussian(),biases_init=Constant(0.0))
            h_layer = Softplus(name='h')




        y_nonlin = Logistic(name='y_nonlin')

        err_func = MaskedSquaredError(name='err_func') # self-defined

        pop_vec = PopVec(input_dim=config['N_RING'], name='pop_vec')
        perf_func = SimplePerformance(thres=0.3*np.pi, name='perf_func')

        self.x_to_h = x_to_h
        self.h_layer = h_layer
        self.h_to_o = h_to_o
        self.y_nonlin = y_nonlin
        self.err_func = err_func
        self.popvec = pop_vec
        self.perf_func = perf_func
        self.children = [x_to_h, h_layer, h_to_o, y_nonlin, err_func, pop_vec, perf_func]
        #self.children = [x_to_h, h_layer, h_to_o, err_func, pop_vec, perf_func]
        if config['h_type'] == 'gru':
            self.x_to_g = x_to_g
            self.children += [x_to_g]

    @application
    def get_h(self, x):
        if self.config['h_type'] == 'lstm':
            x_transform = self.x_to_h.apply(x)
            h, c = self.h_layer.apply(x_transform)

        elif self.config['h_type'] == 'simple_rec':
            x_transform = self.x_to_h.apply(x)
            h = self.h_layer.apply(x_transform)

        elif self.config['h_type'] in ['leaky_rec', 'leaky_rec_ei', 'leaky_rec_ca']:
            x_transform = self.x_to_h.apply(x)
            h = self.h_layer.apply(x_transform)

        elif self.config['h_type'] == 'gru':
            x_transform = self.x_to_h.apply(x)
            g = self.x_to_g.apply(x)
            h = self.h_layer.apply(x_transform,g)

        elif self.config['h_type'] == 'mlp':
            x_transform = self.x_to_h.apply(x)
            h = self.h_layer.apply(x_transform)

        return h

    @application
    def get_y_from_h(self, h):
        y = self.h_to_o.apply(h)
        y = self.y_nonlin.apply(y)
        return y

    @application
    def get_y_deprecated(self, x):
        h = self.get_h(x)
        y = self.h_to_o.apply(h)
        y = self.y_nonlin.apply(y)
        return y

    @application
    def get_cost_from_y(self, y, y_hat, y_hat_loc, c_mask=None):
        # Loss
        loss = self.err_func.apply(y_hat, y, c_mask)

        # Disable this for now
        # Performance
        y_loc = self.popvec.apply(y[-1,:,-self.config['N_RING']:])
        performance = self.perf_func.apply(y_loc, y[-1,:,0], y_hat_loc)

        return [loss, performance]

    @application
    def cost(self, x, y_hat, y_hat_loc, c_mask=None):
        h = self.get_h(x)
        y = self.get_y_from_h(h)
        return self.get_cost_from_y(y, y_hat, y_hat_loc, c_mask)





