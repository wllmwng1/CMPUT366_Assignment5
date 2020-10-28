#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   V = np.load('ValueFunction.npy')
   G = np.load('ValueFunction1.npy')
   T = np.load('ValueFunction2.npy')
   print V.shape
   print(V)
   x = []
   y = []
   for i in V:
       print(i)
   plt.plot(V, label = 'n = 0')
   plt.plot(G, label = 'n = 5')
   plt.plot(T, label = 'n = 50')
   plt.legend()
   plt.show()
