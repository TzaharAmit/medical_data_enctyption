# Adversarial Encryption and Reconstruction of Medical Images

This project explores **adversarial attacks** as a novel encryption mechanism for medical images, and investigates methods for reconstructing both the original images and diagnostic labels. Instead of treating adversarial perturbations as threats, this work repurposes them as a **privacy-preserving tool** for securing sensitive medical data.

## 🚀 Overview
- **Dataset**: NIH Chest X-ray dataset (112,120 frontal-view images, 30,805 patients).
- **Encryption**: Adversarial attacks (FGSM – white-box, HopSkipJump & Square – black-box).
- **Reconstruction**:
  - Variational Autoencoder (VAE) architectures
  - Simple denoisers for image restoration
  - Robust classifiers for label recovery
  - A joint model combining image + label reconstruction
- **Evaluation Metrics**:
  - Image quality: **PSNR, SSIM**
  - Classification: **Accuracy, Precision, Recall, F1-score**

## 🔒 Motivation
Medical images are highly sensitive, containing both identifiable and diagnostically critical information.  
Traditional encryption methods (e.g., AES, DES, chaos-based) protect privacy but render images unreadable until fully decrypted, disrupting clinical workflows.  
This project demonstrates how adversarial perturbations can **encrypt images invisibly**:  
- For unauthorized users → encrypted images look unchanged but mislead AI systems.  
- For authorized users → reconstruction models recover the original image and labels, maintaining clinical usability.  

## 🛠️ Project Structure
