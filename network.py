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
from tensorflow.python.ops.rnn_cell_impl import RNNCell

import tools


def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

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


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = y.get_shape().as_list()[-1]
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / num_units)  # preferences
    cos_pref = np.cos(pref)
    sin_pref = np.sin(pref)
    temp_sum = tf.reduce_sum(y, axis=-1)
    temp_cos = tf.reduce_sum(y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.mod(loc, 2*np.pi)


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
    """The most basic RNN cell.

    Args:
        num_units: int, The number of units in the RNN cell.
        activation: Nonlinearity to use.    Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
         in an existing scope.    If not `True`, and the existing scope already has
         the given variables, an error is raised.
        name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
    """

    def __init__(self,
                 num_units,
                 n_input,
                 alpha,
                 sigma_rec=0,
                 activation='softplus',
                 w_rec_init='diag',
                 rng=None,
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        # self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._w_rec_init = w_rec_init
        self._reuse = reuse

        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._bias_start = 0.
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._bias_start = 0.
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._bias_start = 0.5
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
            self._bias_start = 0.5
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
            self._bias_start = 0.5
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
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
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(n_hidden,
                                                              rng=self.rng)
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

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError(
                "Expected inputs.shape[-1] to be known, saw shape: %s"
                                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
                'kernel',
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._initializer)
        self._bias = self.add_variable(
                'bias',
                shape=[self._num_units],
                initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1-self._alpha) * state + self._alpha * output

        return output, output


class LeakyGRUCell(RNNCell):
  """Leaky Gated Recurrent Unit cell (cf. https://elifesciences.org/articles/21492).

  Args:
    num_units: int, The number of units in the GRU cell.
    alpha: dt/T, simulation time step over time constant
    sigma_rec: recurrent noise
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               alpha,
               sigma_rec=0,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(LeakyGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.
    # self.input_spec = base_layer.InputSpec(ndim=2)

    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

    self._alpha = alpha
    self._sigma = np.sqrt(2 * alpha) * sigma_rec

  @property
  def state_size(self):
      return self._num_units

  @property
  def output_size(self):
      return self._num_units

  def build(self, inputs_shape):
      if inputs_shape[1].value is None:
        raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                         % inputs_shape)

      input_depth = inputs_shape[1].value
      self._gate_kernel = self.add_variable(
          "gates/%s" % 'kernel',
          shape=[input_depth + self._num_units, 2 * self._num_units],
          initializer=self._kernel_initializer)
      self._gate_bias = self.add_variable(
          "gates/%s" % 'bias',
          shape=[2 * self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.constant_initializer(1.0, dtype=self.dtype)))
      self._candidate_kernel = self.add_variable(
          "candidate/%s" % 'kernel',
          shape=[input_depth + self._num_units, self._num_units],
          initializer=self._kernel_initializer)
      self._candidate_bias = self.add_variable(
          "candidate/%s" % 'bias',
          shape=[self._num_units],
          initializer=(
              self._bias_initializer
              if self._bias_initializer is not None
              else init_ops.zeros_initializer(dtype=self.dtype)))

      self.built = True

  def call(self, inputs, state):
      """Gated recurrent unit (GRU) with nunits cells."""

      gate_inputs = math_ops.matmul(
          array_ops.concat([inputs, state], 1), self._gate_kernel)
      gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

      value = math_ops.sigmoid(gate_inputs)
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

      r_state = r * state

      candidate = math_ops.matmul(
          array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
      candidate = nn_ops.bias_add(candidate, self._candidate_bias)

      c = self._activation(candidate)
      # new_h = u * state + (1 - u) * c  # original GRU
      new_h = (1 - self._alpha * u) * state + (self._alpha * u) * c
      new_h += tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
      return new_h, new_h


class LeakyGRUCell_obsolete(RNNCell):
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
            if hparams is None:
                raise ValueError(
                    'No hparams found for save_dir {:s}'.format(save_dir))

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
        if hparams['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hparams['in_type'])
        n_input = hparams['n_input']
        n_rnn = hparams['n_rnn']
        n_output = hparams['n_output']

        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        if hparams['loss_type'] == 'lsq':
            self.c_mask = tf.placeholder("float", [None, n_output])
        else:
            # Mask on time
            self.c_mask = tf.placeholder("float", [None])

        # Activation functions
        if hparams['activation'] == 'power':
            f_act = lambda x: tf.square(tf.nn.relu(x))
        elif hparams['activation'] == 'retanh':
            f_act = lambda x: tf.tanh(tf.nn.relu(x))
        else:
            f_act = getattr(tf.nn, hparams['activation'])

        # Recurrent activity
        if hparams['rnn_type'] == 'LeakyRNN':
            n_in_rnn = self.x.get_shape().as_list()[-1]
            cell = LeakyRNNCell(n_rnn, n_in_rnn,
                                hparams['alpha'],
                                sigma_rec=hparams['sigma_rec'],
                                activation=hparams['activation'],
                                w_rec_init=hparams['w_rec_init'],
                                rng=rng)
        elif hparams['rnn_type'] == 'LeakyGRU':
            cell = LeakyGRUCell(
                    n_rnn, hparams['alpha'],
                    sigma_rec=hparams['sigma_rec'], activation=f_act)
        elif hparams['rnn_type'] == 'EILeakyGRU':
            cell = EILeakyGRUCell(
                    n_rnn, hparams['alpha'],
                    sigma_rec=hparams['sigma_rec'], activation=f_act)
        elif hparams['rnn_type'] == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(n_rnn, activation=f_act)

        elif hparams['rnn_type'] == 'GRU':
            cell = tf.contrib.rnn.GRUCell(n_rnn, activation=f_act)
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
                [n_rnn, n_output],
                dtype=tf.float32
            )
            b_out = tf.get_variable(
                'biases',
                [n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, dtype=tf.float32)
            )

        h_shaped = tf.reshape(self.h, (-1, n_rnn))
        y_shaped = tf.reshape(self.y, (-1, n_output))
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
                                (-1, tf.shape(self.h)[1], n_output))
        y_hat_fix, y_hat_ring = tf.split(
            self.y_hat, [1, n_output-1], axis=-1)
        self.y_hat_loc = tf_popvec(y_hat_ring)

        self.var_list = tf.trainable_variables()
        self.weight_list = [v for v in self.var_list if is_weight(v)]

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hparams['l1_h'] > 0:
            self.cost_reg += tf.reduce_mean(tf.abs(self.h))*hparams['l1_h']
        if hparams['l2_h'] > 0:
            self.cost_reg += tf.sqrt(
                    tf.reduce_mean(tf.square(self.h)))*hparams['l2_h']

        if hparams['l1_weight'] > 0:
            self.cost_reg += hparams['l1_weight']*tf.reduce_mean(
                [tf.reduce_mean(tf.abs(v)) for v in self.weight_list])
            #hparams['l1_weight']*tf.add_n([tf.reduce_mean(tf.abs(v)) for v in self.var_list if ('kernel' in v.name or 'weight' in v.name) ])
        if hparams['l2_weight'] > 0: #maddy added check
            # TODO(gryang): Check if this is correct
            self.cost_reg += hparams['l2_weight']*tf.reduce_mean(
              [tf.sqrt(tf.reduce_mean(tf.square(v))) for v in self.weight_list])

        # Create an optimizer.
        self.opt = tf.train.AdamOptimizer(
            learning_rate=hparams['learning_rate'])
        # Set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver(self.var_list)

        self.save_dir = save_dir
        self.hparams = hparams
        for v in self.var_list:
            if 'rnn' in v.name:
                if 'kernel' in v.name or 'weight' in v.name:
                    # TODO(gryang): For GRU, fix
                    self.w_rec = v[n_input:, :]
                    self.w_in = v[:n_input, :]
                else:
                    self.b_rec = v
            else:
                assert 'output' in v.name
                if 'kernel' in v.name or 'weight' in v.name:
                    self.w_out = v
                else:
                    self.b_out = v

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                             str((n_rnn, n_output)) + ', but found ' +
                             str(w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                             str((n_rnn, n_rnn)) + ', but found ' +
                             str(self.w_rec.shape))
        if self.w_in.shape != (n_input, n_rnn):
            raise ValueError('Shape of w_in should be ' +
                             str((n_input, n_rnn)) + ', but found ' +
                             str(self.w_in.shape))

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

    def get_h_obsolete(self, x):
        """get the recurrent unit activities"""
        return self.sess.run(self.h, feed_dict={self.x: x})

    def get_y_from_h_obsolete(self, h):
        """get the output from recurrent activities"""
        return self.sess.run(
                self.y_hat,
                feed_dict={self.h: h}).reshape((h.shape[0], h.shape[1], -1))

    def get_y_obsolete(self, x):
        """get the output from input"""
        return self.get_y_from_h(self.get_h(x))

    def get_y_loc_obsolete(self, y):
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
            sess: tensorflow session
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hparams['n_input']
        for v in self.var_list:
            if 'kernel' in v.name or 'weight' in v.name:
                # Connection weights
                v_val = sess.run(v)
                if 'output' in v.name:
                    # output weights
                    v_val[units, :] = 0
                elif 'rnn' in v.name:
                    # recurrent weights
                    v_val[n_input + units, :] = 0
                sess.run(v.assign(v_val))

        if verbose:
            print('Lesioned units:')
            print(units)
