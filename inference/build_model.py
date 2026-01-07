from models.multiscale_model import MultiScaleTemporalModel
from models.pose.poseformer_factorized import PoseFormerFactorized


def build_model_from_config(cfg):
    pose_cfg = cfg["pose_model"]
    cls = cfg["classifier"]["num_classes"]

    pose = PoseFormerFactorized(
        joints=pose_cfg["joints"],
        in_ch=pose_cfg["in_channels"],
        dim=pose_cfg["hidden_dim"],
        layers=pose_cfg["layers"],
        heads=pose_cfg["heads"],
        max_T=pose_cfg["max_seq_len"],
        drop=cfg["regularization"]["short_term_dropout"],
        use_sinusoidal_pe=pose_cfg.get("use_sinusoidal_pe", False),
        enable_temporal=pose_cfg.get("enable_temporal_attention", True),
        return_tokens=True
    )

    model = MultiScaleTemporalModel(
        short_seq_model=pose,
        num_classes=cls,
        enable_long_term=cfg["long_term_model"]["enabled"],
        long_heads=cfg["long_term_model"]["heads"],
        long_layers=cfg["long_term_model"]["layers"],
        drop=cfg["long_term_model"]["dropout"],
        multimodal=cfg["image_model"]["enabled"],
        img_feature_dim=cfg["image_model"]["feature_dim"],
        img_backbone=cfg["image_model"]["backbone"],
        fusion_mode=cfg["fusion_model"]["mode"],
        fusion_out_dim=cfg["fusion_model"]["output_dim"]
    )
    return model
