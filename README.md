# 🖼️ Image Sharpening using Knowledge Distillation

This project implements an image sharpening system using **Knowledge Distillation**, where a powerful pretrained **Restormer** model acts as a **teacher** and a lightweight **Residual UNet** model is trained as a **student** to sharpen blurred images efficiently. The aim is to produce sharper images with fewer computational resources while maintaining high quality.



## 🔍 Problem Statement

Blurry images (due to camera shake, defocus, or motion) often lose essential high-frequency details like edges, textures, and fine patterns. Recovering these details is critical for medical imaging, surveillance, and consumer photography.

The goal of this project is:

- To train a **lightweight student model** (Residual UNet) to sharpen blurry images.
- To use **Knowledge Distillation (KD)** to guide the student using a **pretrained teacher model (Restormer)**.
- To evaluate the model’s performance using **SSIM (Structural Similarity Index)** and **PSNR (Peak Signal-to-Noise Ratio)**.



## 🧠 Knowledge Distillation Framework

Knowledge Distillation is used to transfer knowledge from a large model (teacher) to a smaller model (student).

### 📌 Loss Function Used:

We combine two types of loss:

- **L1 Loss between student output and ground truth**
- **L1 Loss between student output and teacher (Restormer) output**

### Formula:
TotalLoss = α × L1(Student, GroundTruth) + β × L1(Student, TeacherOutput)


Where:
- α = 0.8
- β = 0.2



## 📁 Dataset

- Source: Custom dataset uploaded on Kaggle
- Total: **1800 image pairs**
  - 900 original (sharp + blurred)
  - 900 additional cropped & blurred images
- Blur Type: **Gaussian Blur**, σ = 0.5
- Format: `.png`, paired image names (e.g. `001.png` in both folders)

### Folder Structure:
Blur2Sharp/
├── blur_Image/ # Blurred images
└── sharp_Image/ # Ground truth sharp images




## 🧰 Technologies Used

| Tool             | Purpose                         |
|---------------   |-------------------------------- |
| **Kaggle**       | Training & evaluation (GPU: T4) |
| **PyTorch**      | Deep learning framework         |
| **Restormer**    | Pretrained teacher model        |
| **OpenCV**       | Image loading & preprocessing   |
| **scikit-image** | SSIM & PSNR calculation         |




## 🧠 Models

### ✅ Teacher Model — Restormer
- Source: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)
- Task: Motion Deblurring
- Checkpoint: `motion_deblurring.pth`
- Used for inference only (teacher guidance)

### ✅ Student Model — Residual UNet
- Architecture: UNet-like with residual blocks
- Trained using L1 + KD Loss
- Efficient, lightweight & fast to train



## 📊 Results

| Metric    | Value          |
|--------   |----------------|
| **SSIM**  | ✅ 98.72%     |
| **PSNR**  | ✅ 37.91 dB   |

- Training Time: ~10 minutes on Kaggle T4 GPU
- Epochs: 5
- Input Image Size: 256×256 (resized for speed)


## 🗂️ Project Structure
Image-Sharpening-KD/
├── notebook/
│ └── image-sharpening-kd.ipynb
├── checkpoints/
│ └── residual_unet_student.pth
├── models/
│ ├── residual_unet.py
│ └── restormer_loader.py
├── utils/
│ └── metrics.py
├── requirements.txt
└── README.md

## 🛠️ How to Run

1. Clone the repo:

git clone https://github.com/Kundayadav18/Image-Sharpening-KD.git
cd Image-Sharpening-KD

pip install -r requirements.txt

2. Install dependencies:
pip install -r requirements.txt

3.Run the notebook
jupyter notebook notebook/image-sharpening-kd.ipynb

4.Load trained weights (optional):
from models.residual_unet import ResidualUNet
student = ResidualUNet().to(device)
student.load_state_dict(torch.load("checkpoints/residual_unet_student.pth"))
student.eval()






