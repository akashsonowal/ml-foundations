import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
    ):
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
        # h (n_nodes, in_features), adj_mat (n_heads, n_heads, n_heads) or (n_heads, n_heads, 1) when adj_mat is  same for all heads
        n_nodes = h.shape[0]
        g = self.linear(h).view(
            n_nodes, self.n_heads, self.n_hidden
        )  # for each head we do linear transformation
        g_repeat = g.repeat(
            n_nodes, 1, 1
        )  # repeats (stack) along the dim 0 [g1, g2, .....gn, g1, g2, ...] (n_nodes*n_nodes, n_heads, n_hodden)
        g_repeat_interleave = g.repeat_interleave(
            g_repeat, dim=0
        )  # the same nodes gets together [g1, g1, ......gn, gn]  (n_nodes*n_nodes, n_heads, n_hodden)
        g_concat = torch.cat(
            [g_repeat_interleave, g_repeat], dim=-1
        )  # concats to make pairs (g1, g1), (g1, g2).....(n_nodes*n_nodes, n_heads, 2*n_hodden)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)
        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)  # (n_nodes, n_nodes, n_heads)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat == 0, float("-inf"))
        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum("ijh,jhf->ihf", a, g)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)
