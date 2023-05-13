import torch
from torch import nn

class GraphAttentionLayer(nn.Module):
  def __init__(self, in_features: int, out_features: int, n_heads: int, is_concat: bool = True, dropout: float = 0.6, leaky_relu_negative_slope: float = 0.2):
    super().__init__()
    self.n_heads = n_heads
    self.is_concat = is_concat
    
    if is_concat:
      assert out_features % n_heads == 0
      self.n_hidden = out_features // n_heads
    else:
      self.n_hidden = out_features 
    
    self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False) 
    self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
    self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
    self.softmax = nn.Softmax(dim=1)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
    n_nodes = h.shape[0]
    g = self.linear(h.view(n_nodes, self.n_heads, self.n_hidden))
    g_repeat = g.repeat(n_nodes, 1, 1)
    

      
