# 🖇 HAR-Safety-AI
[![HF Model](https://img.shields.io/badge/HF%20Model-npc_LoRA--fps-ff69b4)](https://huggingface.co/m97j/har-safety-model)
[![Colab](https://img.shields.io/badge/Colab-Notebook-yellow)](https://colab.research.google.com/drive/1Nv46aBuSGtsPjjckHdpfFWRMAqbwj5Bh?usp=sharing)

**Multimodal Pose-Image Fusion-Based Action Recognition Model**

---

## 📌 Project Overview
- **Goal**: **Real-time hazardous action recognition in public safety and industrial settings**

- **Approach**:

  - OpenPose-based **pose sequence** + RGB **image sequence** fusion
  - **2-Step Learning Strategy**: Pose-Specific Pre-training (MPOSE) → Multimodal Fine-tuning (HAA500)

- **Core Design**:

  - Temporal-Spatial Factorized Attention (PoseFormerFactorized)
  
  - Lightweight CNN (ImageEncoder)
  
  - Late Fusion (MultiModalFusionModel)

---

## 🎯 Key Features
1. **Multimodal Fusion**  
   - Pose: Motion structure information, robust to background/illumination changes  
   - Image: Provides object/environment contextual information
   
    → Improved recognition performance by combining complementary features

2. **Efficient Transformer Architecture**
  
   - Temporal/Spatial Attention Separation
   - Reduced computational load by more than 13x from $O(T^2J^2)$ to $O(T^2J + J^2T)$

3. **Real-time & Scalability**

   - Minimized inference latency with lightweight CNN and factorized attention
   - Additional sensor data, such as IMU, can be fused.

---

## 📂 Dataset
- **MPOSE**: BODY_25 format, $T=30$ frame pose sequence
- **HAA500**: RGB 480p, parallel extraction of OpenPose skeleton
- All inputs are normalized and used in the PoseFormer encoder.

---

## 🏗 Model Architecture
- **PoseFormerFactorized**: Separate Temporal/Spatial Attention Training
- **ImageEncoder**: ResNet-18 backbone, global pooling followed by embedding
- **MultiModalFusionModel**: Late Fusion of pose and image features → Softmax classification

---

## 🚀 Training Strategy
1. **Stage 1**: Pose-only pretraining (MPOSE)

2. **Stage 2**: Multimodal fine-tuning (HAA500)

---

## 📊 Expected Performance and Usage
- **Robustness**: Stable motion recognition even in diverse environments (lighting and background changes)
- **Real-time**: Fast inference with factorized attention and lightweight CNN
- **Scalability**: Applicable to various domains, including industrial safety, public safety, and sports analytics

---

## ⚙️ Technology Stack
- **Frameworks**: PyTorch, OpenPose
- **Models**: PoseFormerFactorized, ResNet-18
- **Data**: MPOSE, Kinetics-700
- **Infra**: Colab, CUDA

---

## 📜 License
MIT License

---
