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
    data = [[[] for i in range(num_episodes)] for x in range(3)]
    print(data)
    n_types = [0,5,50]

    for run in range(num_runs):
      print "run number: ", run
      print "\n"
      RL_init()
      RL_agent_message('n = 0')
      steps = 0
      is_terminal = False
      RL_start()
      while (not is_terminal) and ((max_steps == 0) or (steps < max_steps)):
          rl_step_result = RL_step()
          is_terminal = rl_step_result['isTerminal']
          steps += 1
      Q = RL_agent_message("get Q")
      M = RL_agent_message("get M")
      for n in range(len(n_types)):
          data[n][0].append(steps)
      for n in range(len(n_types)):
          RL_init()
          RL_agent_message('n = ' + str(n_types[n]))
          RL_agent_message(['give Q',Q])
          RL_agent_message(['give M',M])
          for episode in range(1,num_episodes):
              steps = 0
              is_terminal = False
              RL_start()
              while (not is_terminal) and ((max_steps == 0) or (steps < max_steps)):
                  rl_step_result = RL_step()
                  is_terminal = rl_step_result['isTerminal']
                  steps += 1
              data[n][episode].append(steps)
    print(data)
    average = [[] for n in range(3)]
    for d in range(len(data)):
        for r in data[d]:
            average[d].append(np.mean(r))
    print(average)
    RL_cleanup()

    np.save("SeedFunction", average)
