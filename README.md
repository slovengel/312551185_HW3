# 312551185_HW2
Student ID: 312551185\
Name: 郭晏涵

## Introduction
This project aims to tackle an instance segmentation problem in medical images using Mask R-CNN. The dataset consists of RGB images, with 209 samples for training/validation, and 101 samples for testing. The objective is to detect and segment individual cell instances and correctly classify each into one of the predefined cell types.

To enhance segmentation performance, I employed several key strategies including transfer learning, adaptive learning rate scheduling, and K-fold cross-validation. The core of my method is built upon a Mask R-CNN framework with a ResNet-50-FPN backbone, fine-tuned for cell instance segmentation task.

## How to Install
```bash
conda create -n env python=3.12
conda activate env
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pycocotools==2.0.8 -c pytorch -c nvidia
```

## How to Run the Code
```bash
python main.py
```

## Performance Snapshot
![image](https://github.com/slovengel/312551185_HW3/blob/main/codabench_snapshot.PNG)
