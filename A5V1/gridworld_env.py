#!/usr/bin/env python

"""
  Author: Adam White, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Code for the Gambler's problem environment from the Sutton and Barto
  Reinforcement Learning: An Introduction Chapter 4.
  For use in the Reinforcement Learning course, Fall 2017, University of Alberta
"""

from utils import rand_norm, rand_in_range, rand_un
import numpy as np

gridworld = [
[0,0,0,0,0,0,0,1,0],
[0,0,1,0,0,0,0,1,0],
[0,0,1,0,0,0,0,1,0],
[0,0,1,0,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,0,0]]

start_state = (0,4)
end_state = (8,5)

current_state = None

def env_init():
    global current_state
    current_state = (0,0)


def env_start():
    """ returns numpy array """
    global current_state
    global start_state
    current_state = start_state
    return current_state

def env_step(action):
    """
    Arguments
    ---------
    action : int
        the action taken by the agent in the current state

    Returns
    -------
    result : dict
        dictionary with keys {reward, state, isTerminal} containing the results
        of the action taken
    """
    global current_state
    global end_state

    (x,y) = current_state
    if action == "up" and y-1 > 0:
        if gridworld[y-1][x] == 0:
            y = y-1
    if action == "down" and y+1 < 6:
        if gridworld[y+1][x] == 0:
            y = y+1
    if action == "right" and x+1 < 9:
        if gridworld[y][x+1] == 0:
            x = x+1
    if action == "left" and x-1 >= 0:
        if gridworld[y][x-1] == 0:
            x = x-1
    current_state = (x,y)

    reward = 0
    is_terminal = False
    if (x,y) == end_state:
        reward = 1
        is_terminal = True
    result = {"reward": reward, "state": current_state, "isTerminal": is_terminal}

    return result

def env_cleanup():
    #
    return

def env_message(in_message): # returns string, in_message: string
    """
    Arguments
    ---------
    inMessage : string
        the message being passed

    Returns
    -------
    string : the response to the message
    """
    return ""
