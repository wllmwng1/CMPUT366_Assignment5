#!/usr/bin/env python

"""
 Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian, Zach Holland
 Purpose: for use of Rienforcement learning course University of Alberta Fall 2017
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   V = np.load('AlphaFunction.npy')
   print V.shape
   print(V)
   A = [0.03125,0.0625,0.125,0.25,0.5,1]
   for i in range(len(V)):
       print(i)
       plt.plot(V[i], label = str(A[i]))
   plt.legend()
   plt.show()
