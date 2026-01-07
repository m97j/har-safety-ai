import torch
import torch.nn as nn

from fusion.concat_ln import MMFusionConcatLN
from fusion.cross_attn import MMFusionCrossAttnShallow
from pose.poseformer_factorized import PoseFormerFactorized
from temporal.long_term import LongTermTemporalBlock
from vision.image_encoder import ImageEncoder


class MultiScaleTemporalModel(nn.Module):
    def __init__(self, short_seq_model: PoseFormerFactorized, num_classes: int,
                 enable_long_term=True, long_heads=8, long_layers=2, drop=0.1,
                 multimodal=True, img_feature_dim=128, img_backbone="resnet18",
                 fusion_mode="concat", fusion_out_dim=128):
        super().__init__()
        self.short_seq_model = short_seq_model
        self.enable_long_term = enable_long_term
        self.pose_dim = short_seq_model.dim
        self.multimodal = multimodal

        # long-term pose
        if enable_long_term:
            self.long_term = LongTermTemporalBlock(self.pose_dim, heads=long_heads, layers=long_layers, drop=drop)
        self.classifier_pose = nn.Linear(self.pose_dim, num_classes)  # pose-only classifier

        # image encoder & fusion
        if multimodal:
            self.image_encoder = ImageEncoder(output_dim=img_feature_dim, pretrained=True, backbone=img_backbone)
            if fusion_mode == "concat":
                self.fusion = MMFusionConcatLN(self.pose_dim, img_feature_dim, out_dim=fusion_out_dim, num_classes=num_classes)
                self.fusion_mode = "concat"
            elif fusion_mode == "xattn":
                self.fusion = MMFusionCrossAttnShallow(self.pose_dim, img_feature_dim, num_classes=num_classes, heads=4, layers=1, drop=0.1)
                self.fusion_mode = "xattn"
            else:
                raise ValueError("fusion_mode must be 'concat' or 'xattn'")
        else:
            self.image_encoder = None
            self.fusion = None
            self.fusion_mode = None

    def forward(self, pose_seq_batch, img_seq_batch=None, long_key_padding_mask=None,
                return_feat=False, return_tokens=False):
        # pose_seq_batch: (B,L,T,J,C)
        B, L, T, J, C = pose_seq_batch.shape

        pooled_list, tokens_list = [], []
        for l in range(L):
            pooled, tokens = self.short_seq_model.forward_features(pose_seq_batch[:, l])
            pooled_list.append(pooled)
            if return_tokens:
                tokens_list.append(tokens)
        feats = torch.stack(pooled_list, 1)  # (B,L,Dp)

        if self.enable_long_term:
            long_out = self.long_term(feats, key_padding_mask=long_key_padding_mask)  # (B,L,Dp)
            pose_pooled = long_out.mean(1)  # (B,Dp)
        else:
            pose_pooled = feats.mean(1)  # (B,Dp)

        # unimodal pose-only
        if not self.multimodal or img_seq_batch is None or self.fusion is None:
            logits = self.classifier_pose(pose_pooled)
            return (logits, pose_pooled) if return_feat else logits

        # Pooling images by segment (processing time axis/group axis + reflecting mask)
        # img_seq_batch: (B, L, T, C, H, W)
        assert img_seq_batch is not None, "img_seq_batch is required if multimodal=True."
        B, L, T, C, H, W = img_seq_batch.shape

        img_feats = []
        for l in range(L):
            x = img_seq_batch[:, l]                  # (B, T, C, H, W)
            Bt, Ct, Ht, Wt = B*T, C, H, W
            x4d = x.reshape(Bt, Ct, Ht, Wt)
            feat_bt = self.image_encoder(x4d)        # (B*T, Di)  ← Conv2d-safe
            feat_b_t = feat_bt.view(B, T, -1)        # (B, T, Di)
            feat_b = feat_b_t.mean(dim=1)            # (B, Di)  ← Time-axis aggregation (average).
            img_feats.append(feat_b)

        img_feats = torch.stack(img_feats, dim=1)    # (B, L, Di)

        # Group axis aggregation with mask reflection (ignore padding)
        if long_key_padding_mask is not None:
            # long_key_padding_mask: (B, L), True = pad
            valid = (~long_key_padding_mask).float()           # (B, L)
            num_valid = valid.sum(dim=1).clamp(min=1.0)        # (B,)
            img_pooled = (img_feats * valid.unsqueeze(-1)).sum(dim=1) / num_valid.unsqueeze(-1)
        else:
            img_pooled = img_feats.mean(dim=1)                 # (B, Di)


        # fusion
        if self.fusion_mode == "concat":
            return self.fusion(pose_pooled, img_pooled, return_feat=return_feat)
        else:  # xattn
            assert return_tokens, "Set return_tokens=True to use cross-attention fusion."
            pose_tokens = torch.cat(tokens_list, dim=1)  # (B, sumL*T, J, D)
            return self.fusion(pose_tokens, img_pooled, return_feat=return_feat)
