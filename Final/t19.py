# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:16:32 2013

@author: pramodh
"""

import matplotlib.pyplot as plt
import numpy as np


x = np.random.uniform(size=100)
y = x**2
plt.plot(x, y)
plt.show()
