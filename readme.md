---

```markdown
# Project Name

![Project Badge](https://img.shields.io/badge/version-1.0.0-green) ![Python](https://img.shields.io/badge/python-3.8-blue)

> 프로젝트 간단 설명: 이 프로젝트는 이미지 디노이징을 위한 신경망 모델을 사용하여 고해상도 이미지 복원을 수행합니다.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)

---

### Introduction

본 프로젝트는 FFTFormer 및 Restormer 모델을 사용한 **이미지 복원**을 목적으로 합니다. 노이즈가 포함된 이미지를 입력으로 받아 클린 이미지를 출력하는 **딥러닝 모델**을 제공합니다.

### Features

- FFT 기반 모델(FFTFormer)과 Restormer 모델 지원
- **PSNR** 기반 성능 평가
- 다양한 **데이터셋** 및 **모델 학습**을 위한 모듈화된 코드 구조

---

## Installation

### Requirements

1. Python >= 3.8
2. Torch >= 1.9.0
3. OpenCV, numpy, tqdm 등 (requirements.txt 참고)

conda create --name <env_name> --file requirements.txt
conda activate <env_name>
pip install -r requirements.txt

### Installation Steps

**Clone the repository**:


conda create --name <env_name> --file requirements.txt
conda activate <env_name>

pip install -r requirements.txt

## Pretrained Models

본 프로젝트에서는 다양한 이미지 복원에 최적화된 pretrained 모델을 사용하였습니다. 모델의 세부사항은 다음과 같습니다.

### 1. FFTFormer
- **모델 설명**: FFTFormer는 FFT 기반의 주파수 도메인에서 작동하는 모델로, 고해상도 이미지 복원을 위해 설계되었습니다.
- **출처**: [FFTFormer 논문](https://openaccess.thecvf.com/content/CVPR2023/papers/Kong_Efficient_Frequency_Domain-Based_Transformers_for_High-Quality_Image_Deblurring_CVPR_2023_paper.pdf)
- **Pretrained Dataset**: GoPro 데이터셋을 기반으로 사전 학습되었습니다.

github : https://github.com/kkkls/FFTformer/tree/main?tab=readme-ov-file#Test


### 2. Restormer
- **모델 설명**: Restormer는 고해상도 이미지 복원 및 디블러링 작업에 탁월한 성능을 보이는 모델로, Transformer 기반의 구조를 가집니다.
- **출처**: [Restormer 논문](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf)

- **Pretrained Dataset**: GoPro 데이터셋을 기반으로 사전 학습되었습니다.

github : https://github.com/swz30/Restormer/tree/main

Usage

1. Data Preparation

데이터셋을 다음과 같은 폴더 구조로 준비해 주세요:
Datasets는 각 task별로 구분하여 DU, FF_RF, FI, High_light, LB_RB, LE, Low_light, SC, Test, UD_RL로 구분하여 학습을 진행합니다.

Datasets/
└── Training
    ├── GT
    └── noise
└── Validation
    ├── GT
    └── noise

2. Training

학습을 시작하려면 다음과 같은 명령어를 사용하세요:
python train.py --task UD_RL --epochs 10 --batch_size 1

3. Testing

테스트용으로 준비된 이미지를 디노이징합니다.
python test.py --input_dir ./Datasets/Test --output_dir ./submission

---
