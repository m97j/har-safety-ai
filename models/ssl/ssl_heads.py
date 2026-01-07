import torch.nn as nn
import torch.nn.functional as F


class SSLHeads(nn.Module):
    def __init__(self, dim, joints, out_ch=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim))
        self.recon = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, out_ch))
        self.order_cls = nn.Linear(dim, 1)

    def contrastive_proj(self, pooled):  # (B,D)->(B,D)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)

    def reconstruct(self, tokens):  # (B,T,J,D)->(B,T,J,C)
        B, T, J, D = tokens.shape
        return self.recon(tokens.reshape(B * T * J, D)).reshape(B, T, J, -1)

    def order_logits(self, pooled):
        return self.order_cls(pooled)
