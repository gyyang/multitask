# -*- coding: utf-8 -*-
"""
Chance Abbott unit
@author: guangyuyang
"""

from __future__ import division
import numpy
import matplotlib.pyplot as plt
import time

import theano
from theano import tensor as T
from theano import scalar
from theano.tensor import elemwise
from theano.tensor.nnet import scalar_sigmoid

from blocks.bricks.base import application
from blocks.bricks.interfaces import Activation

from mysoftplus import Softplus



class ChanceAbbott(Activation):
    r""" ChanceAbbott brick.

    The chance & abbott function is defined as :math:`\zeta(x) = (x-x0)/(1-exp(-g(x-x0)))`.

    .. See Wong & Wang 2006 for example

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        # TODO: Make this numerically stable, and perhaps efficient
        input_shift = input_ - 0.4
        return input_shift/(1-T.exp(-50.*input_shift))


x = T.vector('x')
#act = Softplus()
act = ChanceAbbott()
y = act.apply(x)
f_y = theano.function([x], y)
grad_y = theano.grad(y.sum(), [x])
f_grad_y = theano.function([x], grad_y)



start = time.time()
x_ = numpy.linspace(0,0.6,601).astype(theano.config.floatX)
for i in range(100):    
    grad_y_ = f_grad_y(x_)
print(time.time()-start)

plt.plot(x_, grad_y_[0])
plt.show()
