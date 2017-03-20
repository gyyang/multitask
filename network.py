"""
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

def gen_ortho_matrix(dim):
    # Generate random orthogonal matrix
    # Taken from scipy.stats.ortho_group
    # Copied here from compatibilty with older versions of scipy
    H = np.eye(dim)
    for n in range(1, dim):
        x = np.random.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1)
                 - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H

class LeakyRNNCell(RNNCell):
    """The most basic RNN cell."""

    def __init__(self, num_units, alpha, sigma_rec=0, input_size=None,
                 activation='softplus', w_rec_init='diag'):
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._w_rec_init = w_rec_init
        if activation == 'softplus':
            self._activation = tf.nn.softplus
            self._bias_start = 0.
            self._w_in_start = 1.0
            if self._w_rec_init == 'diag':
                self._w_rec_start= 0.54
            elif self._w_rec_init == 'randortho':
                self._w_rec_start= 1.0
        elif activation == 'tanh':
            self._activation = tf.tanh
            self._bias_start = 0.
            self._w_in_start = 1.0
            self._w_rec_start= 0.54
        elif activation == 'relu':
            self._activation = tf.nn.relu
            self._bias_start = 0.5
            self._w_in_start = 1.0
            self._w_rec_start= 0.54
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

    def _linear(self, args, output_size, bias, scope=None):
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

        w_in0 = np.random.randn(input_size, output_size)/np.sqrt(input_size)*self._w_in_start

        if self._w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(output_size)
        elif self._w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*gen_ortho_matrix(output_size)

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.w_rnn0 = matrix0

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
                            self._bias_start, dtype=dtype))
        return res + bias_term



    def __call__(self, inputs, state, scope=None):
        """Leaky RNN: output = new_state = (1-alpha)*state + alpha*activation(W * input + U * state + B)."""
        with vs.variable_scope(scope or type(self).__name__):    # "LeakyRNNCell"
            output = \
    (1-self._alpha)*state + \
    self._alpha*self._activation(self._linear([inputs, state], self._num_units, True) + \
    tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma, dtype=tf.float32))

        return output, output

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

if __name__ == '__main__':
    from scipy.stats import ortho_group
    H = ortho_group.rvs(dim=100)
    H = gen_ortho_matrix(100)
    w, v = np.linalg.eig(H)
    import matplotlib.pyplot as plt
    plt.scatter(np.real(w), np.imag(w))