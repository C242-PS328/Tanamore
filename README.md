# Tanamore Machine Learning Repository

This repository contains datasets, models, and scripts used for plant disease and plant type classification using machine learning techniques. The project uses image datasets for training and evaluating models. This repository contains machine learning models developed for the Tanamore application. These models are designed to:
- **Plant Disease Detection**: Identifies plant diseases based on images of leaves.
- **Plant Species Identification**: Recognizes plant species based on user-uploaded images.
These models aim to improve plant care and farming efficiency through advanced machine learning techniques.

---

## Table of Contents
1. [Datasets](#datasets)
2. [Model 1: Plant Disease Classification](#model-1-plant-disease-classification)
3. [Model 2: Plant Type Classification](#model-2-plant-type-classification)

---

## Datasets

The following datasets are available for training and testing the models:

1. **[Plant Disease Images Dataset](https://github.com/Tanamore/machine_learning/tree/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/Dataset%20Gambar%20Penyakit%20Tanaman)**  
   Contains images of plants with various diseases for classification and diagnosis.

2. **[Plant Type Images Dataset](https://github.com/Tanamore/machine_learning/tree/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/Dataset%20Gambar%20Jenis%20Tanaman)**  
   Contains images of different plant types for classification purposes.

Detailed descriptions of the datasets are available in the respective files:
- [Plant Type Dataset Description](https://github.com/Tanamore/machine_learning/blob/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/Jenis%20Tanaman.md)
- [Plant Disease Dataset Description](https://github.com/Tanamore/machine_learning/blob/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/dataset_penyakit_tanaman.md)

---

## Model 1: Plant Disease Classification

This model is trained to classify images of plants based on the diseases they have. It uses convolutional neural networks (CNN) for image classification.

- **[Model Notebook: Plant Disease Classification](https://github.com/Tanamore/machine_learning/blob/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/Model_1_Klasifikasi_Penyakit_Tanaman.ipynb)**: The notebook for training and evaluating the disease classification model.
- **[Revised Model Notebook](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/Model1_Revisi_Fix.ipynb)**: A fixed version of the original model.
- **[Best Model](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/best_model.h5)**: The best-performing model saved in H5 format.
- **[Fixed Model](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/model1_fix.h5)**: A revised version of the model.
- **[TensorFlow.js Model](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/tfjs_model1_fix.zip)**: The model converted for use with TensorFlow.js.

---

## Model 2: Plant Type Classification

This model is trained to classify plant images based on their species or type.

- **[Model Notebook: Plant Type Classification](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/model%202_klasifikasi%20jenis%20tanaman/Model_2_Klasifikasi_Jenis_Tanaman.ipynb)**: The notebook for training and evaluating the plant type classification model.
- **[InceptionV3 Best Model](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/model%202_klasifikasi%20jenis%20tanaman/inceptionv3_best_model.h5)**: The best-performing model using the InceptionV3 architecture.
- **[My Model](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/model%202_klasifikasi%20jenis%20tanaman/my_model2.h5)**: A custom-trained model for plant type classification.

---

