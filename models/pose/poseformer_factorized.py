import torch
import torch.nn as nn

from models.pose.sinusoidal_pe import SinusoidalPE
from models.pose.spatial_block import SpatialBlock
from models.pose.temporal_block import TemporalBlock


class PoseFormerFactorized(nn.Module):
    def __init__(self, joints=17, in_ch=3, dim=128, layers=4, heads=8,
                 num_classes=6, max_T=30, drop=0.3, use_sinusoidal_pe=False,
                 enable_temporal=True, return_tokens=True):
        super().__init__()
        self.joints, self.in_ch, self.dim = joints, in_ch, dim
        self.enable_temporal, self.return_tokens = enable_temporal, return_tokens

        self.embed = nn.Linear(in_ch, dim)
        self.joint_type_emb = nn.Embedding(joints, dim)
        if use_sinusoidal_pe:
            self.temporal_pe = SinusoidalPE(dim, max_len=max_T)
            self.use_pe = True
        else:
            self.temporal_pe_param = nn.Parameter(torch.zeros(1, max_T, 1, dim))
            self.use_pe = False

        self.blocks = nn.ModuleList([
            nn.ModuleList([TemporalBlock(dim, heads=heads, drop=drop),
                           SpatialBlock(dim, heads=heads, drop=drop)])
            for _ in range(layers)
        ])
        self.head = nn.Linear(dim, num_classes)

    def _ensure_3ch(self, x):  # (B,T,J,C)
        if x.shape[-1] == 2:
            B, T, J, _ = x.shape
            z = torch.zeros((B, T, J, 1), device=x.device, dtype=x.dtype)
            return torch.cat([x, z], dim=-1)
        if x.shape[-1] > 3:
            return x[..., :3]
        return x

    def forward_tokens(self, x):  # (B,T,J,C)
        x = self._ensure_3ch(x)
        B, T, J, C = x.shape
        z = x[..., 2:3]
        x = torch.cat([x[..., :2], torch.tanh(z)], dim=-1)

        joint_ids = torch.arange(J, device=x.device)
        x = self.embed(x) + self.joint_type_emb(joint_ids)[None, None, :, :]
        x = self.temporal_pe(x) if self.use_pe else (x + self.temporal_pe_param[:, :T])

        for tblk, sblk in self.blocks:
            if self.enable_temporal:
                x = tblk(x)
            x = sblk(x)
        return x  # (B,T,J,D)

    def forward_features(self, x):
        tokens = self.forward_tokens(x)
        pooled = tokens.mean(dim=(1, 2))
        return (pooled, tokens) if self.return_tokens else pooled

    def forward(self, pose_x, return_feat=False):
        f = self.forward_features(pose_x)
        pooled = f[0] if isinstance(f, tuple) else f
        logits = self.head(pooled)
        return (logits, pooled) if return_feat else logits
