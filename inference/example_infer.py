import torch
from inference.predictor import HARPredictor
from inference.utils.download_model import download_model
from inference.utils.load_config import load_config
from inference.utils.load_labels import load_labels

cfg = load_config()
labels = load_labels()

predictor = HARPredictor(
    weight_path=download_model(),
    config=cfg
)

# dummy inputs
pose = torch.randn(1, 4, 30, 17, 3)
img  = torch.randn(1, 4, 30, 3, 224, 224)

topk_ids, topk_probs = predictor.topk(pose, img)
decoded = [(labels[str(i)], p) for i, p in zip(topk_ids, topk_probs)]

print("Top-k Predictions: ")
for label, prob in decoded:
    print(f"  {label}: {prob}")