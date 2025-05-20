# Paddy Disease and Nutrient Deficiency Detection

This is a web-integrated deep learning tool that detects **11 paddy leaf diseases** and **4 nutrient deficiencies** from user-uploaded images.  
The tool allows users to upload a leaf image and select whether to check for a disease or a nutrient deficiency. Based on the input, it provides predictions along with precautionary steps in both **English and Telugu**, making it accessible and practical for farmers.

---

## Overview

The system is designed to support early and accurate diagnosis of paddy crop issues to improve yield and reduce losses.

Key Highlights:
- Supports **11 disease classes** and **4 nutrient deficiency types**
- Two separate deep learning models:
  - One for disease classification
  - One for nutrient deficiency detection
- Built using **MobileNetV3-Large** architecture
- Web-based interface that runs model inference on image uploads
- Provides results with localized **precautionary advice in English and Telugu**

---

## Model Details

- Architecture: MobileNetV3-Large (fine-tuned for this use case)
- Frameworks: TensorFlow and Keras
- Performance:
  - 63.48% accuracy on test-set images
  - Approximately 35% accuracy on real-world field images
- Strategy:
  - Separate specialized models for disease and deficiency classification
  - Trained on a curated dataset of over 13,000 labeled leaf images

---

## How It Works

1. The user uploads a paddy leaf image through the web interface.
2. Chooses between "Disease Detection" or "Deficiency Detection."
3. The backend processes the image using the appropriate model.
4. Output is displayed:
   - Predicted class name (disease or deficiency)
   - Suggested precautions in English and Telugu

---

## Dataset

- 13,644+ labeled paddy leaf images
- Images cover 11 diseases and 4 nutrient deficiencies
- Collected from agricultural datasets, field research, and curated manually
- Not hosted on GitHub due to size limitations

To request access to the dataset for research or testing purposes, contact me at:  
**khemanth7416@gmail.com**

---

## Goal

The primary objective is to build a real-time, web-accessible tool that helps:
- Identify crop health issues early
- Provide actionable recommendations in the user's local language
- Reduce the need for expert consultations in remote farming areas

---

## Current Status

- Initial models trained and tested
- Web tool supports image upload and class selection
- Bilingual result generation implemented
- Improvements in progress through:
  - Data augmentation
  - Domain adaptation
  - Real-world testing and user feedback

