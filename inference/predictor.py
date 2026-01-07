import torch
from inference.build_model import build_model_from_config


class HARPredictor:
    def __init__(self, weight_path, config, device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = build_model_from_config(config)
        state = torch.load(weight_path, map_location=self.device)
        if "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()

    @torch.no_grad()
    def forward_logits(self, pose_seq, img_seq):
        pose_seq = pose_seq.to(self.device)

        if img_seq is not None:
            img_seq = img_seq.to(self.device)
            
        return self.model(pose_seq, img_seq)

    def predict_distribution(self, pose_seq, img_seq):
        logits = self.forward_logits(pose_seq, img_seq)
        return logits.softmax(dim=-1)

    def topk(self, pose_seq, img_seq, k=5):
        probs = self.predict_distribution(pose_seq, img_seq)
        return probs.topk(k, dim=-1)

    def threshold(self, pose_seq, img_seq, thr=0.1):
        probs = self.predict_distribution(pose_seq, img_seq)
        return probs * (probs > thr)

