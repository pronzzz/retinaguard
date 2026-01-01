# RetinaGuard: Robust Diabetic Retinopathy Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

**RetinaGuard** is a medical imaging system designed to diagnose Diabetic Retinopathy (DR) in rural regions where image capture quality varies. It utilizes an ensemble of deep residual networks and strictly defined preprocessing pipelines to ensure robustness against noise and poor lighting.

## 🌟 Key Features

*   **Robust Preprocessing (Ben Graham's Method)**:
    *   **Circular Crop**: Removes irregular black borders common in fundus photography.
    *   **CLAHE**: Contrast Limited Adaptive Histogram Equalization highlights blood vessels and lesions.
    *   **Gaussian Smoothing**: Reduces sensor noise from low-quality cameras.
*   **Dual-ResNet Ensemble**: Combines **ResNet101** and **ResNet152** using a Soft Voting mechanism to maximize sensitivity.
*   **Clinical Metric**: Optimized for **Quadratic Weighted Kappa (QWK)**, the industry standard for grading DR severity.
*   **Dynamic Resource Allocation**: Automatically adjusts to MPS (Apple Silicon), CUDA, or CPU.

## 📂 Project Structure

```
blindness-detection/
├── data/                   # Dataset storage (downloaded via script)
├── models/                 # Saved model checkpoints
├── src/
│   ├── dataset.py          # Custom Dataset and Preprocessing
│   ├── model.py            # RetinaEnsemble Architecture
│   └── train.py            # Training Loop & Validation
├── download_data.py        # Kaggle Dataset Downloader
├── test_pipeline.py        # Verification Script
├── visualize_preprocessing.py # Demo for Image Enhancement
└── requirements.txt        # Dependencies
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   Kaggle API Key (configured in `~/.kaggle/kaggle.json`)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/pronzzz/blindness-detection.git
    cd blindness-detection
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Setup

Download the **APTOS 2019 Blindness Detection** dataset automatically:

```bash
python3 download_data.py
```

This verifies your Kaggle credentials and places the data in the `data/` directory.

## 🛠️ Usage

### 1. Visualize Preprocessing
Verify the "Ben Graham" enhancement pipeline on your data:

```bash
python3 visualize_preprocessing.py
```
*Output: `preprocessing_demo.png`*

### 2. Train Model
Start the training loop (default: 15 epochs, Batch Size 8).
The script automatically detects your GPU/MPS.

```bash
python3 -m src.train --epochs 15 --batch_size 8
```

### 3. Verify Pipeline
Run a dry-run test (no full dataset needed) to check code integrity:

```bash
python3 test_pipeline.py
```

## 🧬 Model Architecture

The system uses a **Soft Voting Ensemble**:
1.  **Input**: $512 \times 512$ Preprocessed Fundus Image.
2.  **Branch A**: ResNet101 (Pretrained ImageNet) $\rightarrow$ Modified Head (5 classes).
3.  **Branch B**: ResNet152 (Pretrained ImageNet) $\rightarrow$ Modified Head (5 classes).
4.  **Output**: Average of Softmax Probabilities from A and B.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built for the intent of preventing blindness in under-resourced communities.*
