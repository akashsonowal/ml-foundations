from torch import Tensor
from torch import nn

class Attention(nn.Module):
    def __init__(
        self.
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False, 
        proj_bias: bool = True, 
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
    
    def forward(self, x: Tensor) -> Tensor:
        pass