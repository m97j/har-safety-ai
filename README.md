# 🖇 HAR-Safety-AI

[![Hugging Face Model](https://img.shields.io/badge/Hugging_Face-model_card-ff69b4)](https://huggingface.co/m97j/har-safety-model)

**Multimodal Pose + Image Fusion-Based Action Recognition Model**

---

# 📌 Project Overview

HAR-Safety-AI is a **real-time human action recognition model** designed for safety-critical environments such as industrial sites and public facilities.

This model uses:

* **Pose sequence (skeleton)**
* **RGB image sequence**

and fuses them using a multi-scale transformer architecture for robust behavior classification.

---

# 🎯 Key Features

### 1) **Multimodal Fusion**

* **Pose**: Motion structure, invariant to illumination/background
* **Image**: Context such as objects, environment
* → Combining both yields higher robustness and accuracy

### 2) **Efficient Transformer-Based Architecture**

* Temporal/Spatial Factorized Attention (PoseFormerFactorized)
* Reduced complexity from $(O(T^2J^2)) → (O(T^2J + J^2T))$

### 3) **Real-time & Scalable**

* Low latency inference
* Modular: Easily extend to IMU, depth sensors, or other modalities

---

# 📁 Repository Structure

```
har-safety-ai/
│── models/
│   ├── pose/
│   │   └── poseformer_factorized.py
│   ├── image/
│   │   └── image_encoder.py
│   ├── fusion/
│   │   └── fusion_model.py
│   └── multiscale_model.py
│
│── inference/
│   ├── utils/
│   │   ├── download_model.py
│   │   ├── load_config.py
│   │   └── load_labels.py
│   ├── build_model.py
│   ├── predictor.py
│   └── example_infer.py
│
│── README.md
```

---

# 📦 Installation

```bash
pip install torch torchvision
pip install timm
pip install huggingface_hub
```

(ResNet-18 is loaded from torchvision; timm optional if you switch backbones.)

HuggingFace dependency:

```
huggingface_hub >= 0.23
```

---

# ⚡ Device Auto Selection

The predictor automatically selects:

* **CUDA (GPU)** if available
* **CPU** otherwise

You can override manually:

```python
predictor = HARPredictor(weight_path, config, device="cpu")
```

---

# 🧩 Input Format

The model consumes **two parallel sequences**:

---

## 1) Pose Sequence

**Tensor shape:**

```
(B, C, T, J, 3)
```

Where:

| Dimension | Meaning                          |
| --------- | -------------------------------- |
| B         | batch size                       |
| C         | channels (pose encoding streams) |
| T         | 30 frames                        |
| J         | 17 joints                        |
| 3         | (x, y, confidence)               |

---

## 2) RGB Image Sequence

**Tensor shape:**

```
(B, C, T, 3, 224, 224)
```

* Each frame is resized to **224 × 224**
* Uses ResNet-18 backbone (modifiable)

---

# 🔍 Running Inference

Below is the minimal working example:

```python
import torch

from inference.utils.load_config import load_config
from inference.utils.load_labels import load_labels
from inference.utils.download_model import download_model
from inference.predictor import HARPredictor

cfg = load_config()
labels = load_labels()

predictor = HARPredictor(
    weight_path=download_model(),
    config=cfg
)

# Dummy inputs
pose = torch.randn(1, 4, 30, 17, 3)
img  = torch.randn(1, 4, 30, 3, 224, 224)

topk_ids, topk_probs = predictor.topk(pose, img, k=5)

print("Top-k Predictions:")
for idx, score in zip(topk_ids[0], topk_probs[0]):
    print(labels[str(idx.item())], f"{score.item():.4f}")
```

---

# 📈 Output Format — Probability Distribution (Not Argmax!)

Unlike standard classifiers, this model produces a **500-dimensional probability vector**.  
This design supports:

---

## ✔ Top-K Semantic Ranking

```python
topk_ids, topk_probs = predictor.topk(pose, img, k=5)
```

---

## ✔ Threshold-based Filtering

Useful in ambiguous actions:

```python
filtered = predictor.threshold(pose, img, thr=0.1)
```

---

# 📊 Dataset

Training is based on:

### **MPOSE**

* BODY_25 format
* 30-frame pose sequences

### **HAA500**

* RGB videos + synchronized OpenPose skeleton

---

# 🏗 Model Architecture

* **PoseFormerFactorized**  
  Temporal/Spatial attention separation

* **ImageEncoder (ResNet-18)**  
  Extracts contextual features

* **MultiModalFusionModel**  
  Late fusion + probability output

---

# 🚀 Training Strategy (Overview)

1. **Stage 1 — Pose-only pretraining**  
   Train PoseFormerFactorized on MPOSE

2. **Stage 2 — Multimodal fine-tuning**  
   Add RGB encoder and fusion module  
   Train on HAA500  

(Training code not included in this repository.)

---

# 📜 License

Apache License 2.0

---

