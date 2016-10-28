# -*- coding: utf-8 -*-
"""
Chance Abbott unit
@author: guangyuyang
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,0.6,100)
x2 = (x-0.4)*50

#plt.plot(x, x2/(1-np.exp(-x2)))
plt.plot(x, np.log(1+np.exp(x2)))
plt.show()
