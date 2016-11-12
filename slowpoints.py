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

def find_slowpoints(save_addon, rule, x0_list):
    print('Findings slow points...')
    # Generate coherence 0 inputs
    # This has to be different from Mante et al. 2013,
    # because our inputs are always positive, and can appear at different locations
    params = {'tar1_locs' : [0],
              'tar2_locs' : [np.pi],
              'tar1_mod1_strengths' : [1],
              'tar2_mod1_strengths' : [1],
              'tar1_mod2_strengths' : [1],
              'tar2_mod2_strengths' : [1],
              'tar_time'    : 1600}

    with Run(save_addon) as R:
        w_rec = R.w_rec
        w_in  = R.w_in
        b_rec = R.b_rec
        task  = generate_onebatch(rule, R.config, 'psychometric', noise_on=False, params=params)

    # Add constant input to baseline
    b_rec = b_rec + np.dot(w_in, task.x[1000,0,:])

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
    res_list = list()
    for x0 in x0_list:
        res = minimize(g, x0, method='Newton-CG', jac=dgdx)
        res_list.append(res)
    return res_list


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