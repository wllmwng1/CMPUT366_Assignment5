#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Andrew
  Jacobsen, Victor Silva, Sina Ghiassian
  Purpose: Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlon agent using RL_glue.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta

"""

from rl_glue import *  # Required for RL-Glue
RLGlue("gridworld_env", "dynaq_agent")

import numpy as np
import random
import pickle

if __name__ == "__main__":
    num_episodes = 100
    max_steps = 1000

    num_runs = 10
    data = [[] for i in range(num_episodes)]
    n_types = [0,5,50]
    for run in range(num_runs):
      RL_init()
      print "run number: ", run
      print "\n"
      for episode in range(num_episodes):
          steps = 0
          is_terminal = False
          RL_start()
          while (not is_terminal) and ((max_steps == 0) or (steps < max_steps)):
              rl_step_result = RL_step()
              is_terminal = rl_step_result['isTerminal']
              steps += 1
          data[episode].append(steps)
    average = []
    for d in data:
        average.append(np.mean(d))
    RL_cleanup()

    np.save("ValueFunction2", average)
