import torch
import torch.nn as nn


class MMFusionConcatLN(nn.Module):
    def __init__(self, pose_dim, img_dim, out_dim=128, num_classes=6):
        super().__init__()
        in_dim = pose_dim + img_dim
        self.ln = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, pose_feat, img_feat, return_feat=False):  # (B,Dp),(B,Di)
        fused = torch.cat([pose_feat, img_feat], dim=-1)
        fused = self.mlp(self.ln(fused))
        logits = self.classifier(fused)
        return (logits, fused) if return_feat else logits
