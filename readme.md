# Adversarial Encryption and Reconstruction of Medical Images

This project explores **adversarial attacks** as a novel encryption mechanism for medical images, and investigates methods for reconstructing both the original images and diagnostic labels. Instead of treating adversarial perturbations as threats, this work repurposes them as a **privacy-preserving tool** for securing sensitive medical data.

## Overview
- **Dataset**: NIH Chest X-ray dataset (112,120 frontal-view images, 30,805 patients).
- **Encryption**: Adversarial attacks (FGSM – white-box, HopSkipJump & Square – black-box).
- **Reconstruction**:
  - Variational Autoencoder (VAE) architectures
  - Simple denoisers for image restoration
  - Classifiers for label recovery
  - A joint model combining image + label reconstruction
- **Evaluation Metrics**:
  - Image quality: **PSNR, SSIM**
  - Classification: **Accuracy, Precision, Recall**

## Motivation
Medical images are highly sensitive, containing both identifiable and diagnostically critical information.  
Traditional encryption methods protect privacy but render images unreadable until fully decrypted, disrupting clinical workflows.  
This project demonstrates how adversarial perturbations can **encrypt images invisibly**:  
- For unauthorized users → encrypted images look unchanged but mislead AI systems.  
- For authorized users → reconstruction models recover the original image and labels, maintaining clinical usability.  

## Project Structure
The project is organized into three main components:

classification/ – Contains scripts for preparing the NIH Chest X-ray dataset and training a baseline classifier. This baseline serves both as a benchmark for normal performance and as the target model for adversarial attacks.

encryption/ – Implements adversarial attack methods used for encryption. These include FGSM (white-box) as well as HopSkipJump and Square Attack (black-box). Each script generates adversarially perturbed images that conceal diagnostic labels while maintaining visual fidelity.

reconstruction/ – Contains different reconstruction strategies designed to reverse adversarial encryption. This includes VAE-based models (with and without KL divergence), a denoiser for image reconstruction, a classifier for label recovery, and a joint model that combines both image and label reconstruction into a single network.

Together, these modules form the full pipeline: preprocessing and classification → adversarial encryption → reconstruction and evaluation.


---

## Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/adversarial-medical-encryption.git
cd adversarial-medical-encryption

# Create environment
conda create -n adv-encrypt python=3.9 -y
conda activate adv-encrypt

# Install dependencies
torch
torchvision
scikit-learn
numpy
pandas
opencv-python
matplotlib
seaborn
tqdm



