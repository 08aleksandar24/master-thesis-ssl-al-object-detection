# Self-Supervised and Active Learning for Object Detection in Satellite Images

This repository contains code and experiments from my master's thesis at the University of Ljubljana, 
Faculty of Computer and Information Science.


## 📄 Thesis

This repository accompanies my master’s thesis:

> **Self-Supervised and Active Learning for Object Detection in Satellite Images**  
> Faculty of Computer and Information Science, University of Ljubljana, 2025  
> Author: *Aleksandar Georgiev*

📘 [Read the full thesis](https://repozitorij.uni-lj.si/IzpisGradiva.php?id=174269&lang=slv)
## 🧠 Overview

This work investigates:
- How well **self-supervised backbones** transfer to remote sensing object detection tasks.
- Which **active learning acquisition strategies** are most effective for improving data efficiency.
- How much annotation effort can be reduced without sacrificing detection accuracy.

### 🧩 Evaluated Backbones
- **MTP** – Multi-Task Pretraining (supervised, remote sensing specific)
- **SatMAE** – Masked Autoencoder for temporal/multispectral satellite imagery
- **ScaleMAE** – Scale-aware Masked Autoencoder for geospatial data
- **DINOv2** – Large-scale self-distillation on natural images
- **DINOv3** – Self-supervised pretraining on remote sensing data

### 🎯 Active Learning Strategies
- **Random Sampling**  
- **Count<0.3** – counts uncertain detections below a threshold  
- **Top-1 Least Confidence**  
- **Average Confidence**  
- **Diversity-based Clustering**

> For active learning, we added an additional parameter in our code that determines whether to process the entire dataset or only the unlabeled portion.  
> This parameter allows efficient incremental training and evaluation within the AL loop.

---

## ⚙️ Implementation Details

All experiments are implemented using **[MMDetection](https://github.com/open-mmlab/mmdetection)** —  
an open-source PyTorch framework for object detection, providing reproducible pipelines and modular configuration files. All the installation details can be found here [MTP](https://github.com/ViTAE-Transformer/MTP).

---
## Running scripts

Train example
```bash
CUDA_VISIBLE_DEVICES=2 torchrun tools/train.py "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py" "/home/aleksandar/mmdetdinov3/"
```
Test example
```bash
CUDA_VISIBLE_DEVICES=2 torchrun tools/test.py "/home/aleksandar/MTP/RS_Tasks_Finetune/Horizontal_Detection/configs/mtp/dior/dinov3.py" "/home/aleksandar/mmdetdinov3/epoch_12.pth" --show-dir "/home/aleksandar/analysisImages/Dinov3"
```

## Contact
📧 aleksandargeorgiev52@gmail.com

