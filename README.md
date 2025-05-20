# Paddy Disease and Nutrient Deficiency Detection

This project is focused on building a deep learning-based solution to identify **10 common diseases** and **3 nutrient deficiencies** in paddy (rice) crops using leaf images. The aim is to support farmers with **faster, more accurate crop health diagnosis**, ultimately helping improve crop quality and yield.

---

## Project Overview

- Developed two separate deep learning models:
  - One for detecting **diseases**
  - One for identifying **nutrient deficiencies**
- Based on **MobileNetV3-Large** architecture, optimized for mobile and edge deployment.
- Dataset includes **13,644+ clean and labeled images** of paddy leaves.
- Achieved:
  - **63.48% accuracy** on trained data
  - **~35% accuracy** on real-time field images (under real conditions)

The dual-model setup aims to **boost accuracy** by specializing in disease vs. deficiency classification independently.

---

## File Structure

| File | Description |
|------|-------------|
| `features.py` | Core utility functions for feature extraction and model support. |
| `def_testing.ipynb` | Jupyter notebook for testing nutrient deficiency detection model. |
| `disease_testing.ipynb` | Notebook for testing the disease detection model. |
| `dataset_analysis.ipynb` | Exploratory data analysis and visualization of the dataset. |
| `paddy_disease_and_detection.ipynb` | Main training and evaluation pipeline for disease detection. |
| `paddy_def.ipynb` | Model training and evaluation notebook for nutrient deficiency classification. |

---

## Model Details

- Backbone model: **MobileNetV3-Large**
- Two models trained:
  - **Model 1**: Detects 10 common paddy diseases
  - **Model 2**: Detects 3 nutrient deficiencies (like Nitrogen, Phosphorus, Potassium)
- Frameworks used: **TensorFlow** / **Keras**

---

## Goal

The final goal is to build a lightweight, mobile-friendly tool that farmers or agriculture officers can use in the field to **quickly scan a leaf and get instant health feedback** — allowing timely treatment and better yields.

---

## Current Status

- Initial model training completed with promising offline accuracy.
- Real-time performance improvements in progress through:
  - Data augmentation
  - Domain adaptation
  - Field testing and user feedback

---

## Note on Dataset

Due to **GitHub’s storage limitations**, the full image dataset (13,644+ leaf samples) could not be uploaded here.  
If you're interested in accessing the dataset for research or testing, feel free to **contact me directly**.
