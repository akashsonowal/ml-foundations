import torch 
from torch import nn 
import numpy as np 

class RNNCell(nn.Module):
    def __init__(self):
        pass 
    def forward(self, input, hx=None):
        if hx is None:
            pass 