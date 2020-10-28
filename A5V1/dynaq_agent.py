#!/usr/bin/env python

"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017

"""

from utils import rand_in_range, rand_un
import numpy as np
import random
import pickle
rows = 6
columns = 9
last_state = None
last_action = None
n = 50
alpha = 1
epsilon = 0.1
step = 0.1
'''
from page 135 of Reinforcement Learning: An Introduction by Richard Sutton 2nd Edition
Tabular Dyna-Q
Initialize Q(s,a) and Model(s,a) for all s that is element of S and a that is element of A(s)
Do forever:
    a) S <- current(nonterminal) state_prime
    b) A <- epsilon greedy(S,Q)
    c) Execute action A; observe resultant reward, R, and state, S'
    d) Q(S,A) <- Q(S,A) + alpha[R + stepsize(max(a)Q(S',a) - Q(S,A))]
    e) Model(S,A) <- R,S' (assuming deterministic environment)
    f) Repeat n times:
        S <- random previously observed state
        A <- random action previously taken in S
        R,S' <- Model(S,A)
        Q(S,A) <- Q(S,A) + alpha[R + stepsize(max(a)Q(S',a) - Q(S,A))]
'''

def agent_init():
    """
    Hint: Initialize the variables that need to be reset before each run begins
    Returns: nothing
    """
    #initialize the policy array in a smart way
    global Q
    global Model
    Q = [[[0 for a in range(4)] for y in range(columns)] for a in range(rows)]
    Model = dict()

def agent_start(state):
    """
    Hint: Initialize the variavbles that you want to reset before starting a new episode
    Arguments: state: numpy array
    Returns: action: integer
    """
    # pick the first action, don't forget about exploring starts
    global last_state
    global last_action
    (x,y) = state
    prob = np.random.rand()
    if prob < epsilon:
        prob = np.random.randint(4)
        if prob == 0:
            action = "up"
        if prob == 1:
            action = "down"
        if prob == 2:
            action = "right"
        if prob == 3:
            action = "left"
        last_action = prob
    else:
        max_indices = [i for i in range(len(Q[y][x])) if Q[y][x][i] == max(Q[y][x])]
        actionindex = random.choice(max_indices)
        if actionindex == 0:
            action = "up"
        if actionindex == 1:
            action = "down"
        if actionindex == 2:
            action = "right"
        if actionindex == 3:
            action = "left"
        last_action = actionindex
    last_state = state
    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: integer
    Returns: action: integer
    """
    # select an action, based on Q
    global Model
    global Q
    global last_state
    global last_action
    global n

    (x,y) = state
    (last_x,last_y) = last_state
    a = last_action
    #Q(S,A) <- Q(S,A) + alpha[R + stepsize(max(a)Q(S',a) - Q(S,A))]
    Q[last_y][last_x][a] = Q[last_y][last_x][a] + alpha*(reward + step*(max(Q[y][x]) - Q[last_y][last_x][a]))
    Model[(last_state,last_action)] = (reward,state)
    for i in range(n):
        (((last_x,last_y),a), (r,(x,y))) = random.choice(list(Model.items()))
        #print(r,x,y)
        Q[last_y][last_x][a] = Q[last_y][last_x][a] + alpha*(reward + step*(max(Q[y][x]) - Q[last_y][last_x][a]))

    prob = np.random.rand()
    if prob < epsilon:
        prob = np.random.randint(4)
        if prob == 0:
            action = "up"
        if prob == 1:
            action = "down"
        if prob == 2:
            action = "right"
        if prob == 3:
            action = "left"
        last_action = prob
    else:
        max_indices = [i for i in range(len(Q[y][x])) if Q[y][x][i] == max(Q[y][x])]
        actionindex = random.choice(max_indices)
        if actionindex == 0:
            action = "up"
        if actionindex == 1:
            action = "down"
        if actionindex == 2:
            action = "right"
        if actionindex == 3:
            action = "left"
        last_action = actionindex
    last_state = state
    return action

def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    # do learning and update pi
    # Q(S,A) <- Q(S,A) + alpha[R + stepsize(max(a)Q(S',a) - Q(S,A))]
    (last_x,last_y) = last_state
    (x,y) = last_state
    a = last_action
    if a == 0:
        y = last_y - 1
    if a == 1:
        y = last_y + 1
    if a == 2:
        x = last_x - 1
    if a == 3:
        x = last_y + 1
    state = (last_x,last_y)
    #Q(S,A) <- Q(S,A) + alpha[R + stepsize(max(a)Q(S',a) - Q(S,A))]
    Q[last_y][last_x][a] = Q[last_y][last_x][a] + alpha*(reward + step*(max(Q[y][x]) - Q[last_y][last_x][a]))
    Model[(last_state,last_action)] = (reward,state)
    for i in range(n):
        (((last_x,last_y),a), (r,(x,y))) = random.choice(list(Model.items()))
        Q[last_y][last_x][a] = Q[last_y][last_x][a] + alpha*(r + step*(max(Q[y][x]) - Q[last_y][last_x][a]))

    return

def agent_cleanup():
    """
    This function is not used
    """
    # clean up
    return

def agent_message(in_message): # returns string, in_message: string
    #not needed right now
    global n
    global alpha
    global Q
    global Model
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'n = 0'):
        n = 0
    elif (in_message == 'n = 5'):
        n = 5
    elif (in_message == 'n = 50'):
        n = 50
    elif (in_message == 'a = 0.03125'):
        alpha = 0.03125
    elif (in_message == 'a = 0.0625'):
        alpha = 0.0625
    elif (in_message == 'a = 0.125'):
        alpha = 0.125
    elif (in_message == 'a = 0.25'):
        alpha = 0.25
    elif (in_message == 'a = 0.5'):
        alpha = 0.5
    elif (in_message == 'a = 1'):
        alpha = 1
    elif (in_message == 'get Q'):
        return Q
    elif (in_message[0] == 'give Q'):
        Q = in_message[1]
    elif (in_message == 'get M'):
        return Model
    elif (in_message[0] == 'give M'):
        Model = in_message[1]
    else:
        return "I don't know what to return!!"
