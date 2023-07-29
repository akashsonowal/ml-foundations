# approximate q values at a state
# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
# https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb

import random
import math
import gym
import numpy as np  

import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.autograd as autograd 
import torch.nn.functional as F 

from IPython.display import clear_output
import matplotlib.pyplot as plt 
%matplotlib inline 

class ReplayBuffer(object):
    pass 

class DQN(nn.Module):
    def __init__(self):
        pass 
    
    def forward(self):
        pass 

    def act(self):
        pass 

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    q_values = model(state)
    next_q_values = model(next_state)

    expected_q_value = reward * gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss 

if __name__ == "__main__":
    # cartpole environment
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    #

