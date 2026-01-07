import torch.nn as nn


class SpatialBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, drop=0.3):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):  # (B,T,J,D)
        B, T, J, D = x.shape
        x_res = x
        q = x.reshape(B * T, J, D)
        q = self.ln1(q)
        q, _ = self.attn(q, q, q)
        q = q + x_res.reshape(B * T, J, D)
        q = q + self.mlp(self.ln2(q))
        q = q.reshape(B, T, J, D)
        return q
