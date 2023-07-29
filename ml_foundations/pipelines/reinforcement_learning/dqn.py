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

def compute_td_loss():
    pass 

if __name__ == "__main__":
    # cartpole environment
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    #

