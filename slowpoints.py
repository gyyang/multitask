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

def find_slowpoints(save_addon, input, start_points=None, find_fixedpoints=True, dtype='float64'):
    if find_fixedpoints:
        # Finding fixed points require less tolerange
        print('Findings fixed points...')
        options = {'ftol':1e-10, 'gtol': 1e-10} # for L-BFGS-B
        # options = {'ftol':1e-7, 'gtol': 1e-7} # for L-BFGS-B
        # options = {'xtol': 1e-10}
    else:
        # Finding slow points allow more tolerange
        print('Findings slow points...')
        options = {'ftol':1e-4, 'gtol': 1e-4} # for L-BFGS-B
        # options = {'xtol': 1e-5}
    with Run(save_addon) as R:
        w_rec = R.w_rec.astype(dtype)
        w_in  = R.w_in.astype(dtype)
        b_rec = R.b_rec.astype(dtype)

    nh = len(b_rec)
    # Add constant input to baseline
    input = input.astype(dtype)
    b_rec = b_rec + np.dot(w_in, input)

    def dgdx(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        dfdx = 1/(1+1/expy)
        return (-F + np.dot(w_rec.T, F*dfdx))

    def g(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        return np.sum(F**2)/2

    if start_points is None:
        start_points = [np.ones(nh)]

    res_list = list()
    for start_point in start_points:
        start_point = start_point.astype(dtype)
        # res = minimize(g, start_point, method='Newton-CG', jac=dgdx, options=options)
        res = minimize(g, start_point, method='L-BFGS-B', jac=dgdx,
                       bounds=[(0,100)]*nh, options=options)
        # ftol may be important for how slow points are
        # If I pick gtol=1e-7, ftol=1e-20. Then regardless of starting points
        # I find only one fixed point, which depends on the input to the network
        res_list.append(res)
    return res_list

if __name__ == '__main__':
    save_addon = 'tf_latest_300'
    rule = CHOICEATTEND_MOD1
    # find_slowpoints(save_addon, rule, x0_list = [np.zeros(300)])

    with Run(save_addon) as R:
        w_rec = R.w_rec.astype('float64')
        w_in  = R.w_in.astype('float64')
        b_rec = R.b_rec.astype('float64')

    nh = len(b_rec)

    def dgdx(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        dfdx = 1/(1+1/expy)
        return (-F + np.dot(w_rec.T, F*dfdx))

    def g(x):
        expy = np.exp(np.dot(w_rec, x) + b_rec)
        F = -x + np.log(1.+expy) # Assume standard softplus nonlinearity
        return np.sum(F**2)/2.

    x0 = np.ones(nh)
    dx = np.ones(nh)*0.0000001
    x1 = x0 + dx

    print(g(x1) - g(x0))
    print(np.sum(dgdx(x0)*dx))

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