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

def find_slowpoints(save_addon, input, x0=None):
    print('Findings slow points...')
    with Run(save_addon) as R:
        w_rec = R.w_rec
        w_in  = R.w_in
        b_rec = R.b_rec

    nh = len(b_rec)
    # Add constant input to baseline
    b_rec = b_rec + np.dot(w_in, input)

    def dgdx(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        dfdx = 1/(1+1/expy)
        return (-F + np.dot(w_rec.T, F*dfdx))/nh

    def g(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        return np.mean(F**2)/2

    if x0 is None:
        x0 = np.ones(nh)
    # res = minimize(g, x0, method='Newton-CG', jac=dgdx)
    res = minimize(g, x0, method='L-BFGS-B', jac=dgdx,
                   bounds=[(0,100)]*nh, options={'ftol':1e-20, 'gtol': 1e-7})
    # ftol may be important for how slow points are
    # If I pick gtol=1e-7, ftol=1e-20. Then regardless of starting points
    # I find only one fixed point, which depends on the input to the network
    return res

if __name__ == '__main__':
    save_addon = 'tf_latest_300'
    rule = CHOICEATTEND_MOD1
    find_slowpoints(save_addon, rule, x0_list = [np.zeros(300)])

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