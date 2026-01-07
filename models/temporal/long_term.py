import torch.nn as nn


class LongTermTemporalBlock(nn.Module):
    def __init__(self, dim, heads=8, layers=2, drop=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x, key_padding_mask=None):  # (B,L,D)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)
