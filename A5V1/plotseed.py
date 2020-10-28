#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   V = np.load('SeedFunction.npy')
   print V.shape
   print(V)
   for i in V:
       print(i)
   plt.plot(V[0], label = 'n = 0')
   plt.plot(V[1], label = 'n = 5')
   plt.plot(V[2], label = 'n = 50')
   plt.legend()
   plt.show()
