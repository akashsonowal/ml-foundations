import torch
from torch import nn

class GraphAttentionLayer(nn.Module):
  def __init__(self, in_features: int, out_features: int, n_heads: int, is_concat: bool = True, dropout: float = 0.6, leaky_relu_negative_slope: float = 0.2):
    super().__init__()
    self.n_heads = n_heads
    self.is_concat = is_concat
    
    if is_concat:
      assert out_features % n_heads == 0
    