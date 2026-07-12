# Knowledge Distillation for Efficient Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An efficient, lightweight image sharpening system engineered through Knowledge Distillation (KD). This architecture leverages a powerful, pretrained Restormer model as a teacher to train a lightweight Residual UNet student network, achieving high-fidelity deblurring with significantly reduced computational overhead.

---

## 🏛️ Project Architecture & Core Team

### Development Roles
* **Ankit Thakur:** - ML Engineer
  * *Responsibilities:* Designed the training pipeline, customized the loss architecture, implemented model compilation, and orchestrated the distillation framework.
* **Kundan Yadav** — Lead Data Engineer
  * *Responsibilities:* Curated the custom data pipeline, handled preprocessing parameters, and managed training infrastructure.
* **Vansh Gudiyan** — Evaluation & Optimization Engineer
  * *Responsibilities:* Implemented the metrics framework (SSIM/PSNR analysis) and structured deployment profiling.

---

## 🎯 Problem Statement

Blurry images resulting from defocus, camera jitter, or rapid motion lose high-frequency structural elements such as textures, fine edges, and precise patterns. Restoring these patterns is a vital requirement across domains like medical diagnostic imaging, automated surveillance pipelines, and consumer photography.

While state-of-the-art transformer architectures handle these tasks exceptionally well, their massive parameter sizes make edge deployment impractical. This project solves that problem by transferring complex feature-map knowledge into a lightweight network, keeping production speeds fast without sacrificing image reconstruction quality.

---

## 🧠 Knowledge Distillation Framework

The training pipeline transfers structural knowledge from the high-capacity **Restormer** model into a compact **Residual UNet**. 

### Loss Optimization Function
To ensure spatial alignment and sharp edge restoration, the student optimization relies on a balanced joint loss configuration combining standard reconstruction errors with distillation guidance:

$$TotalLoss = \alpha \cdot L_1(\text{Student}, \text{GroundTruth}) + \beta \cdot L_1(\text{Student}, \text{TeacherOutput})$$

Where hyperparameter scaling weights are defined as:
* $\alpha = 0.8$ (Focus on absolute ground-truth replication)
* $\beta = 0.2$ (Focus on mimicking the teacher's latent high-frequency representations)

---

## 📊 Performance Metrics & Results

Evaluation proves that the compressed student model retains near-lossless fidelity compared to the heavy teacher infrastructure, allowing rapid runtime execution.

| Evaluation Metric | Achieved Value | Target Competency |
| :--- | :--- | :--- |
| **SSIM (Structural Similarity)** | 98.72% | Structural Integrity Preservation |
| **PSNR (Peak Signal-to-Noise)** | 37.91 dB | High-Fidelity Signal Reconstruction |

* **Training Profile:** ~10 minutes total execution window on a single Kaggle T4 GPU node.
* **Training Epochs:** 5 iterations.
* **Input Resolution:** Uniformly mapped to $256 \times 256$ dimensions for accelerated processing throughput.

---

## 📁 Repository Structure

```text
Image-Sharpening-KD/
├── notebooks/
│   └── image-sharpening-kd.ipynb       # Training and inference workflows
├── checkpoints/
│   └── residual_unet_student.pth       # Saved weights for the student model
├── models/
│   ├── residual_unet.py                # Student model architecture definition
│   └── restormer_loader.py             # Pretrained teacher interface layer
├── utils/
│   └── metrics.py                      # SSIM and PSNR validation logic
├── requirements.txt                    # System environment dependencies
└── README.md                           # Documentation
🛠️ Installation & Execution
1. Replicate the Repository
Clone your personal copy of the repository and step into the project path:

Bash
git clone [https://github.com/NeuralImprint/Knowledge-Distillation-Image-Enhancement.git](https://github.com/NeuralImprint/Knowledge-Distillation-Image-Enhancement.git)
cd Knowledge-Distillation-Image-Enhancement
2. Configure Dependencies
Install the required environment packages using pip:

Bash
pip install -r requirements.txt
3. Execution & Inference
Launch the execution notebook to review training workflows or run inference:

Bash
jupyter notebook notebooks/image-sharpening-kd.ipynb
To run the trained student model inside an external inference script, import the modules directly:

Python
import torch
from models.residual_unet import ResidualUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = ResidualUNet().to(device)

# Load optimized distillation weights
student.load_state_dict(torch.load("checkpoints/residual_unet_student.pth", map_location=device))
student.eval()

