# SFCLSeg

This repository contains the code implementation for the paper "Spatial-Frequency Collaborative Learning for Weakly Supervised Pulmonary Nodule Segmentation".

## Environment Setup

Tested on:
- Python >= 3.9
- CUDA 11.8
- Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train the model using the proposed method in our paper, run:

```bash
python train_sfcl_cam.py
```

## CAM Generation

After training, generate CAMs using:

```bash
python make_cam_nodule.py
```

This script loads the trained model and produces class activation maps for downstream segmentation.

## Evaluation

To evaluate the generated CAMs:

```bash
python evaluate_cam_nodule.py
```

## Visualization


To visualize CAM results:

```bash
python vis_cam.py
```
