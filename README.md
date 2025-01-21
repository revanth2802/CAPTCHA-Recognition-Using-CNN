# CAPTCHA Recognition Using CNN

## Project Overview

This project explores the use of Convolutional Neural Networks (CNNs) for CAPTCHA recognition, comparing multiple architectures, including simple CNN, VGG-16, and Siamese networks. The primary objective is to develop an efficient and accurate model for CAPTCHA recognition, advancing online security measures against automated attacks.

---

## Features

- **Automated CAPTCHA Recognition**: Implements a comprehensive system capable of recognizing complex CAPTCHA designs.
- **Multiple Architectures**: Evaluation and comparison of simple CNN, VGG-16, and Siamese networks.
- **Preprocessing Techniques**: Includes segmentation, grayscale conversion, and noise reduction.
- **Dataset**: Utilizes a diverse collection of CAPTCHA images for robust training and evaluation.

---

## Project Goals

- Enhance CAPTCHA recognition accuracy using CNN-based models.
- Compare the performance of different architectures for CAPTCHA recognition.
- Contribute to the advancement of artificial intelligence in the domain of cybersecurity.

---

## Technical Specifications

### Requirements

#### Functional
- Input CAPTCHA images for preprocessing and recognition.
- Segmentation and recognition of characters in CAPTCHAs.
- Evaluate accuracy using metrics such as precision and recall.

#### Non-Functional
- High accuracy and performance.
- Scalability to handle various CAPTCHA designs and complexities.
- Robustness against noise and distortions.

### System Specifications

- **Software**: Python, TensorFlow, PyTorch, Google Colab
- **Hardware**: CUDA-enabled GPUs for model training and inference.

---

## Architecture

1. **Data Collection**: Diverse dataset of CAPTCHA images.
2. **Preprocessing**: Grayscale conversion, normalization, and segmentation.
3. **Model Architecture**: Convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.
4. **Training**: Includes data augmentation techniques to improve model generalization.
5. **Evaluation**: Performance analysis across different CAPTCHA designs.

---

## Results

| Model         | Test Accuracy (%) |
|---------------|-------------------|
| Simple CNN    | 84               |
| Siamese       | 88               |
| VGG-16        | 41               |

---

## Dataset

- Generated using the **Claptcha Python package**.
- Includes CAPTCHAs with digits, lowercase, and uppercase letters, along with combinations of all three.

---

## Future Work

- Optimize CNN architecture for higher accuracy.
- Extend the dataset to include more diverse CAPTCHA designs.
- Explore advanced loss functions and hyperparameter tuning for improved performance.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/captcha-recognition.git
