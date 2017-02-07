"""
2016/06/03 Restart, with Blocks

Main training loop and network structure
"""

from __future__ import division

import numpy as np

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest

class LeakyRNNCell(RNNCell):
    """The most basic RNN cell."""

    def __init__(self, num_units, alpha, sigma_rec=0, input_size=None, activation='softplus'):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        if activation == 'softplus':
            self._activation = tf.nn.softplus
        elif activation == 'tanh':
            self._activation = tf.tanh
        elif activation == 'relu':
            self._activation == tf.nn.relu
        else:
            raise ValueError('Unknown activation')
        self._alpha = alpha
        self._sigma = np.sqrt(2*alpha) * sigma_rec


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Leaky RNN: output = new_state = (1-alpha)*state + alpha*activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):    # "LeakyRNNCell"
            output = (1-self._alpha)*state + \
                     self._alpha * self._activation(_linear([inputs, state], self._num_units, True)) + \
                tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma, dtype = tf.float32)

        return output, output

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

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
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    input_size = total_arg_size - output_size
    matrix0 = np.random.randn(input_size, output_size)/np.sqrt(input_size)
    matrix0 = np.concatenate((matrix0, 0.54*np.eye(output_size)), axis=0)

    # Now the computation.
    with vs.variable_scope(scope or "Linear"):
        matrix = vs.get_variable(
                "Matrix", [total_arg_size, output_size], dtype=dtype,
            initializer=tf.constant_initializer(matrix0, dtype=dtype))
        if len(args) == 1:
            res = math_ops.matmul(args[0], matrix)
        else:
            res = math_ops.matmul(array_ops.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = vs.get_variable(
                "Bias", [output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(
                        bias_start, dtype=dtype))
    return res + bias_term


def popvec(y):
    # Population vector read out
    # Assuming the last dimension is the dimension to be collapsed
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1]) # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)

def get_perf(y_hat, y_loc):
    '''
    Get performance
    :param y_hat: Actual output. Time, Batch, Unit
    :param y_loc: Target output location (-1 for fixation). Time, Batch
    :return:
    '''
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[...,0]
    y_hat_loc = popvec(y_hat[...,1:])

    # Fixating? Correctly saccading?
    corr_fix = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * corr_fix + (1-should_fix) * corr_loc
    return perf