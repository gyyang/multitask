"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn_cell_impl import RNNCell, _linear

import tools

def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H


def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf


class LeakyRNNCell(RNNCell):
    """The most basic Leaky RNN cell."""

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 sigma_rec=0,
                 input_size=None,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse)

        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._bias_start = 0.
            self._w_in_start = 1.0
            if self._w_rec_init == 'diag':
                self._w_rec_start = 0.54
            elif self._w_rec_init == 'randortho':
                self._w_rec_start = 1.0
            elif self._w_rec_init == 'randgauss':
                self._w_rec_start = 1.0
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._bias_start = 0.
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._bias_start = 0.5
            self._w_in_start = 1.0
            if self._w_rec_init == 'diag':
                self._w_rec_start = 0.54
            elif self._w_rec_init == 'randortho':
                self._w_rec_start = 0.5
            elif self._w_rec_init == 'randgauss':
                self._w_rec_start = 1.0
        elif activation == 'suplin':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._bias_start = 0.5
            self._w_in_start = 1.0
            if self._w_rec_init == 'diag':
                self._w_rec_start = 0.01  # Only using this now
            elif self._w_rec_init == 'randortho':
                self._w_rec_start = 1.0
            elif self._w_rec_init == 'randgauss':
                self._w_rec_start = 1.0
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2*alpha) * sigma_rec
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        # Generate initialization matrix
        n_hidden = self._num_units
        w_in0 = (self.rng.randn(n_input, n_hidden) /
                 np.sqrt(n_input) * self._w_in_start)

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(n_hidden)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*gen_ortho_matrix(n_hidden, rng=self.rng)
        elif self._w_rec_init == 'randgauss':
            w_rec0 = (self._w_rec_start * 
                      self.rng.randn(n_hidden, n_hidden)/np.sqrt(n_hidden))

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0
        self._initializer = tf.constant_initializer(matrix0, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Leaky RNN.

        output = new_state = 
        (1-alpha)*state + alpha*activation(W * input + U * state + B).
        """
        with vs.variable_scope("leaky_rnn_cell",
                               initializer=self._initializer,
                               reuse=self._reuse):

            output = (
                (1-self._alpha)*state + self._alpha*self._activation(
                _linear([inputs, state], self._num_units, True) +
                tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)))

        return output, output


class LeakyGRUCell(RNNCell):
    """Leaky Gated Recurrent Unit cell.
    
    See for example Song, Yang, Wang eLife 2017 for reference
    """

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 input_size=None,
                 activation=tf.tanh,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(LeakyGRUCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

        self._alpha = alpha
        self._sigma = np.sqrt(2*alpha) * sigma_rec

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                        self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(
                _linear([inputs, r * state], self._num_units, True,
                        self._bias_initializer, self._kernel_initializer))

        # leaky version of GRU
        new_h = (1-self._alpha*u) * state + self._alpha * u * c + \
            tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        return new_h, new_h


class EILeakyGRUCell(RNNCell):
    """Excitatory-inhibitory Leaky GRU cell."""

    def __init__(self,
                 num_units,
                 alpha,
                 sigma_rec=0,
                 input_size=None,
                 activation=tf.tanh,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(EILeakyGRUCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

        self._alpha = alpha
        self._sigma = np.sqrt(2*alpha) * sigma_rec
        nE = int(num_units/2)
        nI = num_units - nE
        self._signs = tf.constant([1.]*nE+[-1.]*nI, dtype=tf.float32)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

        Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
        (default is all zeros).
        kernel_initializer: starting value to initialize the weight.

        Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

        Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:
            weights = vs.get_variable(
                'kernel', [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
            # Weights have to be positive
            if len(args) == 1:
                res = math_ops.matmul(args[0], tf.abs(weights))
            else:
                res = math_ops.matmul(array_ops.concat(args, 1), tf.abs(weights))
            if not bias:
                return res
            with vs.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                if bias_initializer is None:
                    bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
                biases = vs.get_variable(
                'bias', [output_size],
                dtype=dtype,
                initializer=bias_initializer)
            return nn_ops.bias_add(res, biases)

    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        # include positive and negative signs in state
        signed_state = state * self._signs

        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, signed_state]][0]
                bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
            value = math_ops.sigmoid(
                _linear([inputs, signed_state], 2 * self._num_units, True, bias_ones,
                        self._kernel_initializer))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(
                _linear([inputs, r * signed_state], self._num_units, True,
                        self._bias_initializer, self._kernel_initializer))

        # leaky version of GRU
        new_h = (1-self._alpha*u) * state + self._alpha * u * c + \
        tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        return new_h, new_h


class Model(object):
    """The model."""

    def __init__(self,
                 save_dir,
                 hparams=None,
                 sigma_rec=None,
                 dt=None):
        """
        Initializing the model with information from hparams

        Args:
            save_dir: string, directory of the model
            hparams: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hparams
        """

        # Reset tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hparams is None:
            hparams = tools.load_hparams(save_dir)

        tf.set_random_seed(hparams['seed'])
        rng = np.random.RandomState(hparams['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hparams['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hparams['dt'] = dt

        hparams['alpha'] = 1.0*hparams['dt']/hparams['tau']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        if hparams['in_type'] == 'normal':
            n_input = hparams['n_input']
        elif hparams['in_type'] == 'multi':
            n_input = hparams['rule_start'] * hparams['n_rule']
        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, hparams['n_output']])
        if hparams['loss_type'] == 'lsq':
            self.c_mask = tf.placeholder("float", [None, hparams['n_output']])
        else:
            # Mask on time
            self.c_mask = tf.placeholder("float", [None])

        # Activation functions
        f_act = getattr(tf.nn, hparams['activation'])

        # Recurrent activity
        if hparams['rnn_type'] == 'LeakyRNN':
            n_in_rnn = self.x.get_shape().as_list()[-1]
            cell = LeakyRNNCell(hparams['n_rnn'], n_in_rnn,
                                hparams['alpha'],
                                sigma_rec=hparams['sigma_rec'],
                                activation=hparams['activation'],
                                w_rec_init=hparams['w_rec_init'],
                                rng=rng)
        elif hparams['rnn_type'] == 'LeakyGRU':
            cell = LeakyGRUCell(
                    hparams['n_rnn'], hparams['alpha'],
                    sigma_rec=hparams['sigma_rec'], activation=f_act)
        elif hparams['rnn_type'] == 'EILeakyGRU':
            cell = EILeakyGRUCell(
                    hparams['n_rnn'], hparams['alpha'],
                    sigma_rec=hparams['sigma_rec'], activation=f_act)
        elif hparams['rnn_type'] == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(hparams['n_rnn'], activation=f_act)

        elif hparams['rnn_type'] == 'GRU':
            cell = tf.contrib.rnn.GRUCell(hparams['n_rnn'], activation=f_act)
        else:
            raise NotImplementedError("""rnn_type must be one of LeakyRNN,
            LeakyGRU, EILeakyGRU, LSTM, GRU
            """)

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
                cell, self.x, dtype=tf.float32, time_major=True)

        # Output
        with tf.variable_scope("output"):
            # Using default initialization `glorot_uniform_initializer`
            w_out = tf.get_variable(
                'weights',
                [hparams['n_rnn'], hparams['n_output']],
                dtype=tf.float32
            )
            b_out = tf.get_variable(
                'biases',
                [hparams['n_output']],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32)
            )

        h_shaped = tf.reshape(self.h, (-1, hparams['n_rnn']))
        y_shaped = tf.reshape(self.y, (-1, hparams['n_output']))
        # y_hat_ shape (n_time*n_batch, n_unit)
        y_hat_ = tf.matmul(h_shaped, w_out) + b_out
        if hparams['loss_type'] == 'lsq':
            # Least-square loss
            y_hat = tf.sigmoid(y_hat_)
            self.cost_lsq = tf.reduce_mean(
                    tf.square((y_shaped-y_hat)*self.c_mask))
        else:
            y_hat = tf.nn.softmax(y_hat_)
            # Cross-entropy loss
            self.cost_lsq = tf.reduce_mean(
                    self.c_mask * tf.nn.softmax_cross_entropy_with_logits(
                            labels=y_shaped, logits=y_hat_))

        self.y_hat = tf.reshape(y_hat,
                                (-1, tf.shape(self.h)[1], hparams['n_output']))

        self.var_list = tf.trainable_variables()

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hparams['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h))*hparams['l1_h']
        if hparams['l2_h'] > 0:
            self.cost_reg += tf.sqrt(
                    tf.reduce_mean(tf.square(self.h)))*hparams['l2_h']

        if hparams['l1_weight'] > 0:
            self.cost_reg += hparams['l1_weight']*tf.reduce_mean(
                [tf.reduce_mean(tf.abs(v)) for v in self.var_list if ('kernel' in v.name or 'weight' in v.name) ])
            #hparams['l1_weight']*tf.add_n([tf.reduce_mean(tf.abs(v)) for v in self.var_list if ('kernel' in v.name or 'weight' in v.name) ])
        if hparams['l2_weight'] > 0: #maddy added check
            self.cost_reg += hparams['l2_weight']*tf.reduce_mean(
              [tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.var_list if ('kernel' in v.name or 'weight' in v.name) ])          

        # Create an optimizer.
        self.opt = tf.train.AdamOptimizer(
                learning_rate=hparams['learning_rate'])
        # Set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver(self.var_list)

        self.save_dir = save_dir
        self.hparams = hparams

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self):
        """restore the model"""
        sess = tf.get_default_session()
        save_path = os.path.join(self.save_dir, 'model.ckpt')
        self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.save_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def get_h(self, x):
        """get the recurrent unit activities"""
        return self.sess.run(self.h, feed_dict={self.x: x})

    def get_y_from_h(self, h):
        """get the output from recurrent activities"""
        return self.sess.run(
                self.y_hat,
                feed_dict={self.h: h}).reshape((h.shape[0], h.shape[1], -1))

    def get_y(self, x):
        """get the output from input"""
        return self.get_y_from_h(self.get_h(x))

    def get_y_loc(self, y):
        """get the response location from the output"""
        return popvec(y[..., 1:])

    def set_optimizer(self, extra_cost=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : tensorflow variable, 
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        self.grads_and_vars = self.opt.compute_gradients(cost, self.var_list)
        # gradient clipping
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grads_and_vars]
        self.train_step = self.opt.apply_gradients(capped_gvs)

    def lesion_units(self, sess, units, verbose=False):
        """Lesion units given by units

        Args:
            units : can be None, an integer index, or a list of integer indices
        """
        if self.hparams['rnn_type'] != 'LeakyRNN':
            raise ValueError('Only supporting LearkyRNN for now')

        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        n_input, n_rnn, n_output = self.hparams['shape']

        w_out = sess.run(self.var_list[0])
        w_rec = sess.run(self.var_list[2])

        # check if the recurrent and output connection has the correct shape
        assert w_out.shape == (n_rnn, n_output)
        assert w_rec.shape == (n_input+n_rnn, n_rnn)

        # Set output projections from these units to zero
        w_out[units, :] = 0
        w_rec[n_input+units, :] = 0

        # Apply the lesioning
        sess.run(self.var_list[0].assign(w_out))
        sess.run(self.var_list[2].assign(w_rec))

        if verbose:
            print('Lesioned units:')
            print(units)