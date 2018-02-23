"""Tests for network."""

from __future__ import division

import unittest
import numpy as np
import tensorflow as tf
import network


class NetworkTest(unittest.TestCase):

  def testTFPopVec(self):
    batch_size = 15
    n_units = 36
    y = np.random.rand(batch_size, n_units)
    theta = network.popvec(y)

    y2 = tf.placeholder('float', [batch_size, n_units])
    theta2 = network.tf_popvec(y2)
    with tf.Session() as sess:
      theta2_val = sess.run(theta2, feed_dict={y2: y})

    self.assertTrue(np.allclose(theta, theta2_val))


if __name__ == '__main__':
  unittest.main()
