import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, dim, max_len=30):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):  # (B,T,*,D)
        T = x.size(1)
        return x + self.pe[:T].transpose(0, 1)