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

from collections import dequeue

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = dequeue(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
    
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
    
    def forward(self, x):
        return self.layers(x)

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

    USE_CUDA = torch.cuda.is_available()
    Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
    # cartpole environment
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx * epsilon_decay)

    plt.plot([epsilon_by_frame(i) for i in range(10000)])



