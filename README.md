# DiffTR-Flow3D: Generative AI for Time-Resolved Particle Tracking

This repository contains the official implementation of **DiffTR-Flow3D**, as presented in our paper: *"Generative AI models for exploring diffusion dynamics from time-resolved particle tracking"*. 

DiffTR-Flow3D leverages a diffusion-based generative framework to estimate 3D scene flow in fluid dynamics. It captures particle trajectories and diffusion patterns through a combination of geometric backbones and feature transformers.

---

## 📂 Project Structure

The repository is organized into the following components based on the core architecture:

* **`datasetloader/datasets.py`**: Handles data loading for the TR-Flow3D dataset, which contains 10-frame particle sequences.
* **`denoising_diffusion_pytorch/`**: Contains the Gaussian Diffusion forward and reverse process logic.
* **`loss/sceneflow_loss.py`**: Implements the robust regression loss used for training.
* **`models/backbone.py`**: Geometric encoders including DGCNN and PointNet.
* **`models/difftr_flow3d.py`**: The main model wrapper integrating the transformers and correlation modules.
* **`models/matching.py`**: Implements global/local correlation and motion approximation (Equations 8 and 9).
* **`models/transformer.py`**: Contains the Point Transformer and Global Feature Transformer architectures.
* **`utils.py`**: Provides evaluation metrics such as EPE, Acc3d, and Outlier percentages.

---

## 💻 System Requirements

### Hardware
* **Operating System**: Linux system.
* **GPU**: Tested on two NVIDIA RTX 24GB GPUs.
* **Memory**: Sufficient VRAM to handle batch sizes and 3D point cloud processing.

### Software
* **Python**: Version 3.8.19.
* **PyTorch**: Version 1.10.1 (with torchvision 0.11.2 and torchaudio 0.10.1).
* **Conda**: Recommended for environment management.

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Create and activate the environment
conda create -n difftr-flow3d python=3.8.19
conda activate difftr-flow3d

# Install PyTorch and core dependencies
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
pip install -r requirements.txt
```

### 2. Dataset Preparation
* Download the TR-Flow3D dataset from [[Figshare]](https://figshare.com/articles/dataset/TR-Flow3D/27617541).
* Place the data under the ./TR-Flow3D folder.
* The dataset comprises 19,500 training samples and 1,950 testing samples.


### 3. Training

To train the model from scratch:

```bash
CUDA_VISIBLE_DEVICES=0,1 python difftr_flow3d_main.py \
    --train_dataset TR-Flow3D \
    --val_dataset TR-Flow3D \
    --lr 4e-5 \
    --train_batch_size 8 \
    --num_epochs 500 \
    --result_dir results
```

### 4. Evaluation

To evaluate a pre-trained model:

```bash
CUDA_VISIBLE_DEVICES=0,1 python difftr_flow3d_main.py \
    --train_dataset TR-Flow3D \
    --val_dataset TR-Flow3D \
    --result_dir results \
    --resume model_best.pt \
    --eval
```

---

## 🔬 Implementation Details

The model implements a sophisticated generative pipeline for 3D fluid flow estimation:

* **Diffusion Framework**: 
    * **Training**: Uses $N=20$ timesteps with a cosine variance schedule to add noise to ground-truth flow.
    * **Inference**: Employs **DDIM acceleration** to reduce the reverse sampling process to only **2 steps**, significantly improving efficiency.
* **Geometric Backbones**: 
    * Supports `DGCNN` (default) and `PointNet` for constructing point cloud descriptors.
    * DGCNN uses $k=16$ nearest neighbors to capture detailed topological information (Equation 7).
* **Feature Transformers**:
    * **Local Correlation**: Adopts Point Transformer blocks to confine attention to localized areas (Equation 10).
    * **Global Correlation**: Utilizes self- and cross-attention layers to calculate dependencies across the entire point set (Equation 11).
* **Motion Estimation & Approximation**: 
    * Calculates coarse flow via global feature similarity (Equation 8).
    * Refines flow and handles "ghost particles" using a learnable motion approximation module (Equation 9).
* **Multi-frame Consistency**:
    * For time-resolved sequences, the model performs test-time training using **Ridge Regression** to fit polynomial trajectories (Equation 13).
    * This provides a reliable initial condition for the reverse diffusion process in subsequent frames.
* **Loss Function**: 
    * The training objective is a robust regression loss: $L = \sum((||D_{est} - D_{gt}||_{1} + 0.01)^{0.4})$.

---

## 📊 Evaluation Metrics

The repository provides tools to calculate standard scene flow metrics:
* **EPE (End Point Error)**: Average $L_2$ distance between predicted and ground-truth flow (in meters).
* **Acc3d Strict**: Percentage of points with $EPE < 0.05m$ or relative error $< 5\%$.
* **Acc3d Relax**: Percentage of points with $EPE < 0.10m$ or relative error $< 10\%$.
* **Outliers**: Percentage of points with $EPE > 0.30m$ or relative error $> 10\%$.

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{
  title={{Generative AI models for exploring particle diffusion from time-resolved particle tracking}},
  author={Jianan Wei and Yi Yang and Wenguan Wang},
}
```

