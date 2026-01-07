import torch
import torch.nn as nn


class MMFusionCrossAttnShallow(nn.Module):
    def __init__(self, pose_dim, img_dim, num_classes=6, heads=4, layers=1, drop=0.1):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, pose_dim)
        self.img_proj = nn.Linear(img_dim, pose_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=pose_dim, nhead=heads, dropout=drop, batch_first=True)
        self.cross = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.ln = nn.LayerNorm(pose_dim)
        self.mlp = nn.Sequential(nn.Linear(pose_dim, pose_dim), nn.GELU(), nn.Linear(pose_dim, pose_dim))
        self.classifier = nn.Linear(pose_dim, num_classes)

    def forward(self, pose_tokens, img_pooled, return_feat=False):  # (B,T,J,D), (B,Di)
        B, T, J, D = pose_tokens.shape
        pose_seq = pose_tokens.mean(dim=2)  # (B,T,D)
        pose_seq = self.pose_proj(pose_seq)
        img_tok = self.img_proj(img_pooled).unsqueeze(1)  # (B,1,D)

        x = torch.cat([pose_seq, img_tok], dim=1)  # (B,T+1,D)
        x = self.cross(x)
        pose_x = x[:, :T]  # drop image token
        fused = self.mlp(self.ln(pose_x.mean(dim=1)))  # (B,D)
        logits = self.classifier(fused)
        return (logits, fused) if return_feat else logits
