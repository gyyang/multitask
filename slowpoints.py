"""
Find fixed points of a RNN
Directly using numpy. Numpy is faster in this case.
"""

from __future__ import division
import time
from scipy.optimize import curve_fit, minimize
import tensorflow as tf

from task import *
from run import Run

def find_slowpoints(save_addon, rule, x0):

    with Run(save_addon) as R:
        w_rec = R.w_rec
        w_in  = R.w_in
        b_rec = R.b_rec
        config = R.config

    N_RING = config['N_RING']
    _, nh, _ = config['shape']
    # Add the constant rule input to the baseline
    b_rec = b_rec + w_in[:, 2*N_RING+1+rule]

    def dgdx(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        dfdx = 1/(1+1/expy)
        return -F + np.dot(w_rec.T, F*dfdx)

    def g(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        return np.sum(F**2)/2

    # It's 10% faster to provide ejac and fun separately
    # Default seeting Newton-CG is 100% slower but more accurate than L-BFGS-B
    # But much much better than SLSQP
    res = minimize(g, x0, method='Newton-CG', jac=dgdx)
    return res


if __name__ == '__main__':
    save_addon = 'tf_latest_300'
    rule = CHOICEATTEND_MOD1
    find_slowpoints(save_addon, rule, x0 = np.zeros(300))

# Test
# start = time.time()
# x0 = np.random.rand(nh)*2
# g0 = g(x0)
# dgdx0 = dgdx(x0)
# print(g0)
# print(dgdx0[:10])
# print(time.time()-start)


# Tensorflow equivalence
# b = b_rec[:,np.newaxis]
# x = tf.placeholder("float", [nh, 1])
# F = -x + tf.nn.softplus(tf.matmul(w_rec, x) + b)
# g = tf.reduce_sum(tf.square(F))/2
#
# dgdx = tf.gradients(g, x)
#
# with tf.Session() as sess:
#     start = time.time()
#
#     g0 = sess.run(g, feed_dict={x: x0[:,np.newaxis]})
#     dgdx0 = sess.run(dgdx, feed_dict={x: x0[:,np.newaxis]})[0]
#     print(g0)
#     print(dgdx0.flatten()[:10])
#     print(time.time()-start)