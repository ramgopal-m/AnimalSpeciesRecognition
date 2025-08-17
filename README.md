# 🐾 Animal Species Recognition

🚀 **Report and frontend**: [animal-species-recognition](https://animal-species-recognition.vercel.app/)

## 📌 Overview

This project presents a deep learning-based system for automatic recognition of animal species from images. It addresses key challenges in:

- Wildlife monitoring and conservation
- Ecological research
- Education and species identification
- Automated wildlife census
- Real-time animal detection in camera traps

Manual species identification is time-consuming and requires expert knowledge. Our AI solution achieves high accuracy and scalability for practical deployment.

---

## 🔍 Problem Statement

Identify animal species from RGB images using a fine-tuned convolutional neural network. The goal is to automate the classification process over a dataset containing 40 distinct species.

---

## 🛠️ Technical Stack

| Component        | Technology                |
|------------------|---------------------------|
| Model            | ResNet50 (fine-tuned)     |
| Backend          | Flask (Python API)        |
| Frontend         | React + Material-UI       |
| Deep Learning    | PyTorch                   |
| Deployment       | Vercel (Frontend)         |
| Realtime Support | Webcam integration (JS)   |

---

## 📊 Dataset

- **Total Images**: ~30,000
- **Classes**: 40 Animal Species
- **Split**:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

### 🐾 Sample Classes

Horse, Squirrel, Hippopotamus, Moose, Wolf, Giraffe, Bear, Elephant, Zebra, Panda, etc.

---

## 🧠 Model Architecture

- Base Model: `ResNet50` pre-trained on ImageNet
- Fine-Tuning:
  - Custom classification head: `Linear(2048 → 512 → 40)`
  - ReLU, Dropout(0.5), Softmax
- Input Image Size: `224x224`
- Data Augmentation: Resized crop, horizontal flip

---

## 🧪 Results

| Metric                | Value    |
|------------------------|----------|
| Test Accuracy          | 91.87%   |
| Validation Accuracy    | 92.66%   |
| Training Accuracy      | 83.31%   |
| F1-Score               | 92.04%   |
| Top-5 Accuracy         | 98.76%   |

