"""Tests for Tools."""

from __future__ import division

import unittest

import time
import numpy as np
import tensorflow as tf

import tools


class ToolsTest(unittest.TestCase):

    def testLoadSaveLog(self):
        train_dir = 'data/tmp'
        tools.mkdir_p(train_dir)

        log = dict()
        log['train_dir'] = train_dir
        log['step'] = range(10000)
        log['time'] = range(1000)
        for i in range(20):
            log['perf' + str(i)] = list(np.random.rand(1000))
        for i in range(40):
            log['perf' + str(i)] = np.random.rand(1000).astype(np.float32).tolist()

        start_time = time.time()
        tools.save_log(log)
        print(time.time() - start_time)
        log2 = tools.load_log(train_dir)
        print(time.time() - start_time)

        for key, val in log.items():
            self.assertListEqual(list(log2[key]), list(val))


if __name__ == '__main__':
    unittest.main()