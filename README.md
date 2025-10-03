# 🖇 HAR-Safety-AI  
[![HF Model](https://img.shields.io/badge/HF%20Model-npc_LoRA--fps-ff69b4)](https://huggingface.co/m97j/har-safety-model)
[![Colab](https://img.shields.io/badge/Colab-Notebook-yellow)](https://colab.research.google.com/drive/1Nv46aBuSGtsPjjckHdpfFWRMAqbwj5Bh?usp=sharing)

**멀티모달 포즈-이미지 융합 기반 행동 인식 모델**


---

## 📌 프로젝트 개요
- **목표**: 공공 안전 및 산업 현장에서의 **실시간 위험 행동 인식**  
- **접근 방식**:  
  - OpenPose 기반 **포즈 시퀀스** + RGB **이미지 시퀀스** 융합  
  - **2단계 학습 전략**: 포즈 전용 사전학습(MPOSE) → 멀티모달 파인튜닝(HAA500)  
- **핵심 설계**:  
  - 시간-공간 인자분해 어텐션(PoseFormerFactorized)  
  - 경량 CNN(ImageEncoder)  
  - Late Fusion(MultiModalFusionModel)

---

## 🎯 주요 특징
1. **멀티모달 융합**  
   - 포즈: 동작 구조 정보, 배경/조명 변화에 강인  
   - 이미지: 객체·환경 맥락 정보 제공  
   → 상호보완적 특징 결합으로 인식 성능 향상  

2. **효율적 Transformer 구조**  
   - Temporal/Spatial Attention 분리  
   - 기존 $O(T^2J^2)$ → $O(T^2J + J^2T)$로 연산량 **13배 이상 절감**  

3. **실시간성 & 확장성**  
   - 경량 CNN + Factorized Attention으로 추론 지연 최소화  
   - IMU 등 추가 센서 데이터 융합 가능  

---

## 📂 데이터셋
- **MPOSE**: BODY_25 포맷, $T=30$ 프레임 포즈 시퀀스  
- **HAA500**: RGB 480p, OpenPose skeleton 병행 추출  
- 모든 입력은 정규화 후 PoseFormer 인코더에 사용  

---

## 🏗 모델 아키텍처
- **PoseFormerFactorized**: Temporal/Spatial Attention 분리 학습  
- **ImageEncoder**: ResNet-18 백본, 전역 풀링 후 임베딩  
- **MultiModalFusionModel**: 포즈·이미지 특징 Late Fusion → Softmax 분류  

---

## 🚀 학습 전략
1. **Stage 1**: 포즈 전용 사전학습 (MPOSE)  
2. **Stage 2**: 멀티모달 파인튜닝 (HAA500)  

---

## 📊 기대 성능 및 활용
- **강인성**: 다양한 환경(조명·배경 변화)에서도 안정적 동작 인식  
- **실시간성**: Factorized Attention + 경량 CNN으로 빠른 추론  
- **확장성**: 산업 안전, 공공 안전, 스포츠 분석 등 다양한 도메인 적용 가능  

---

## ⚙️ 기술 스택
- **Frameworks**: PyTorch, OpenPose  
- **Models**: PoseFormerFactorized, ResNet-18  
- **Data**: MPOSE, Kinetics-700  
- **Infra**: Colab, CUDA  


---

## 📜 라이선스
MIT License  

---
