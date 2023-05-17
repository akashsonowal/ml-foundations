import typing import Dict

import numpy as np
import torch
from torch import nn

from ml_foundations.models.deep_models.layers import GraphAttentionLayer

class CoraDataset:

  labels: torch.Tensor
  classes: Dict[str, int]
  features: torch.Tensor
  adj_mat: torch.Tensor

  @staticmethod
  def _download():
    pass

  def __init__(self, include_edges: bool = True):
    pass

class GAT():
  def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
    super().__init__()
    self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
    self.activation = nn.ELU()
    self.dropout = nn.Dropout(dropout)

  def forward(self):
    pass

def main():
  pass

if __name__ == "__main__":
  main()