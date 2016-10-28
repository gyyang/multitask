# -*- coding: utf-8 -*-
"""
Chance Abbott unit
@author: guangyuyang
"""

from __future__ import division
import numpy
import theano
from theano import tensor as T
from theano import scalar
from theano.tensor import elemwise
from theano.tensor.nnet import scalar_sigmoid
from blocks.bricks.base import application
from blocks.bricks.interfaces import Activation


class ScalarSoftplus(scalar.UnaryScalarOp):
    """
    This helps numerical stability.
    """
    @staticmethod
    def static_impl(x):
        if x < -30.0:
            return 0.0
        if x > 30.0:
            return x
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.log1p(numpy.exp(x, sig='f'))
        return numpy.log1p(numpy.exp(x))

    def impl(self, x):
        return ScalarSoftplus.static_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * scalar_sigmoid(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        # These constants were obtained by looking at the output of
        # python commands like:
        #  for i in xrange(750):
        #      print i, repr(numpy.log1p(numpy.exp(theano._asarray([i,-i], dtype=dt))))
        # the boundary checks prevent us from generating inf

        # float16 limits: -17.0, 6.0
        # We use the float32 limits for float16 for now as the
        # computation will happend in float32 anyway.
        if (node.inputs[0].type == scalar.float32 or
                node.inputs[0].type == scalar.float16):
            return """%(z)s = %(x)s < -103.0f ? 0.0 : %(x)s > 14.0f ? %(x)s : log1p(exp(%(x)s));""" % locals()
        elif node.inputs[0].type == scalar.float64:
            return """%(z)s = %(x)s < -745.0 ? 0.0 : %(x)s > 16.0 ? %(x)s : log1p(exp(%(x)s));""" % locals()
        else:
            raise NotImplementedError('only floatingpoint is implemented')

    def c_code_cache_version(self):
        v = super(ScalarSoftplus, self).c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v
scalar_softplus = ScalarSoftplus(scalar.upgrade_to_float,
                                 name='scalar_softplus')
softplus = elemwise.Elemwise(scalar_softplus, name='softplus')



class Softplus(Activation):
    r""" Softplus brick.

    The softplus is defined as :math:`\zeta(x) = \log(1+e^x)`.

    .. Dugas, C., Bengio, Y., Belisle, F., Nadeau, C., and Garcia,
       R. (2001). Incorporating second-order functional knowledge
       for better option pricing. In NIPS 13 . MIT Press.

    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return softplus(input_)
