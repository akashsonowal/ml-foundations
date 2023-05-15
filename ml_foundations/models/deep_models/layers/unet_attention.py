from typing import Optional
import torch
from torch.nn.functional as F
from torch import nn

class SpatialTransformer(nn.Module):
  def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
    super().__init__()
    self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    self.transformer_blocks = nn.ModuleList()
    self.proj_out = nn.Conv2d()
    
  def forward(self, x: torch.Tensor, cond: torch.Tensor):
    # x is (b, c, h, w) and cond is (b, n_cond, d_cond)
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    
    
    
