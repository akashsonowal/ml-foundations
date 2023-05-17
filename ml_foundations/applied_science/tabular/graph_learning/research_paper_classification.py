import typing import Dict

import numpy as np
import torch
from torch import nn

from .....models.deep_models.layers import GraphAttentionLayer

class CoraDataset:
  labels: torch.Tensor
  classes: Dict[str, int]
  features: torch.Tensor
  adj_mat: torch.Tensor
        
  @staticmethod
  def _download():
    pass
