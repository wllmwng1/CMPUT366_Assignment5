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
    num_episodes = 50
    max_steps = 1000

    num_runs = 10
    data = [[[] for i in range(num_episodes)] for x in range(6)]
    print(data)
    alpha_types = [0.03125,0.0625,0.125,0.25,0.5,1]

    for a in range(len(alpha_types)):
        RL_agent_message('n = 5')
        RL_agent_message('a = ' + str(alpha_types[a]))
        print('a = ' + str(alpha_types[a]))
        for run in range(num_runs):
            RL_init()
            for e in range(num_episodes):
                steps = 0
                is_terminal = False
                RL_start()
                while (not is_terminal) and ((max_steps == 0) or (steps < max_steps)):
                    rl_step_result = RL_step()
                    is_terminal = rl_step_result['isTerminal']
                    steps += 1
                data[a][e].append(steps)

    print(data)
    average = [[] for n in range(6)]
    for d in range(len(data)):
        for r in data[d]:
            average[d].append(np.mean(r))
    print(average)
    RL_cleanup()

    np.save("AlphaFunction", average)
