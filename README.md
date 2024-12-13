# Tanamore Machine Learning Repository

This repository contains datasets, models, and scripts used for plant disease and plant type classification using machine learning techniques. The project uses image datasets for training and evaluating models. This repository contains machine learning models developed for the Tanamore application. These models are designed to:
- **Plant Disease Detection**: Identifies plant diseases based on images of leaves.
- **Plant Species Identification**: Recognizes plant species based on user-uploaded images.

These models aim to improve plant care and farming efficiency through advanced machine learning techniques.

---

## Table of Contents
1. [Overview](#overview)
2. [Tools and Frameworks](#tools-and-frameworks)
3. [Dataset](#dataset)
   - [Plant Disease Detection](#plant-disease-detection-dataset)
   - [Plant Species Identification](#plant-species-identification-dataset)
4. [Preprocessing Steps](#preprocessing-steps)
5. [Model Architecture](#model-architecture)
   - [Plant Disease Detection](#model-1-plant-disease-detection)
   - [Plant Species Identification](#model-2-plant-species-identification)
6. [Training Process](#training-process)
7. [Evaluation and Results](#evaluation-and-results)
8. [Export and Deployment](#export-and-deployment)
9. [Limitations and Future Work](#limitations-and-future-work)
10. [File Organization](#file-organization)
11. [How to Use](#how-to-use)

---

## Overview

### Model 1: Plant Disease Detection
This model classifies 38 types of plant diseases using images of infected leaves. It was built using a custom Convolutional Neural Network (CNN) architecture and trained on the Plant Village dataset. The aim is to help farmers and gardeners identify plant diseases early and take preventive actions.

### Model 2: Plant Species Identification
This model identifies 22 plant species, including fruits, vegetables, and ornamental plants, based on user-uploaded images. Developed using Transfer Learning with the **InceptionV3** pre-trained model, it simplifies plant identification for both professionals and hobbyists.

---

## Tools and Frameworks
- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow & Keras
- **Pre-trained Model**: InceptionV3 for Model 2
- **Libraries**: NumPy, Matplotlib, OpenCV, scikit-learn
- **Data Augmentation**: ImageDataGenerator
- **Development Platforms**: Google Colab, Google Drive
- **Deployment Tools**: TensorFlow.js for web and mobile compatibility
- **Model Optimization**: Keras Callbacks (ModelCheckpoint, EarlyStopping)

---
## Datasets

### Plant Disease Detection Dataset
Contains images of plants with various diseases for classification and diagnosis for Model 1.
- **Source**:
  - [Plant Village Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?select=color)
  - [Cleaned Dataset](https://github.com/Tanamore/machine_learning/tree/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/Dataset%20Gambar%20Penyakit%20Tanaman)
- **Total Images**: 51,832
- **Classes**: 38 (diseases and healthy plant states)
- Example diseases:
  - Apple Scab, Black Rot, Cedar Apple Rust
  - Tomato Bacterial Spot, Septoria Leaf Spot, Late Blight
  - Healthy states for various plants

### Plant Species Identification Dataset
Contains images of different plant types for classification purposes for Model 2.
- **Sources**: 
  - [House Plant Species Dataset](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species)
  - [Cleaned Dataset](https://github.com/Tanamore/machine_learning/tree/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/Dataset%20Gambar%20Jenis%20Tanaman)
- **Total Images**: 17,238
- **Classes**: 22 species (e.g., Aloe Vera, Grape, Tomato, Monstera Deliciosa, Snake Plant)

Detailed descriptions of the datasets are available in the respective files:
- [Plant Type Dataset Description](https://github.com/Tanamore/machine_learning/blob/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/Jenis%20Tanaman.md)
- [Plant Disease Dataset Description](https://github.com/Tanamore/machine_learning/blob/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Dataset%20Machine%20Learning/dataset_penyakit_tanaman.md)

---

## Preprocessing Steps

### Model 1: Plant Disease Detection
1. **Image Resizing**: All images resized to **256x256** pixels.
2. **Data Augmentation**: Augmentation is already applied in the dataset.
3. **Data Splitting**:
   - **80%** Training
   - **20%** Validation
   - **20%** Testing

### Model 2: Plant Species Identification
1. **Image Resizing**: All images resized to **224x224** pixels.
2. **Data Augmentation**: Random rotations, flips, and zooms applied to enhance model robustness.
3. **Data Splitting**:
   - **80%** Training
   - **20%** Validation
   - **20%** Testing

---

## Model Architecture

**Notes**:  
- **Model 1** is custom-built from scratch, focusing on detecting plant diseases from images.  
- **Model 2** leverages transfer learning to identify plant species, utilizing the rich features learned by InceptionV3 on a large-scale dataset like ImageNet.  
- The architectures cater to different use cases, balancing between customizability and leveraging pre-trained models for efficiency.

### Model 1: Plant Disease Detection
This model is a custom-built Convolutional Neural Network (CNN) architecture designed to extract visual features from input images. The architecture details are as follows:

1. **Input Layer**: Accepts input images of size (256x256x3).
2. **Rescaling**: Normalizes pixel values to the range [0, 1].
3. **Convolutional Layers**:
   - **Layer 1**: 32 filters (3x3 kernel), ReLU activation, followed by MaxPooling (2x2).
   - **Layer 2**: 64 filters (3x3 kernel), ReLU activation, followed by MaxPooling (2x2).
   - **Layer 3**: 128 filters (3x3 kernel), ReLU activation, followed by MaxPooling (2x2).
   - **Layer 4**: 256 filters (3x3 kernel), ReLU activation, followed by MaxPooling (2x2).
4. **Global Average Pooling**: Reduces the feature dimensions into a single vector.
5. **Fully Connected Layers**:
   - Dense layer with 128 units, ReLU activation, and Dropout (0.2).
   - Dense layer with 39 units (output classes), Softmax activation.

**Total Parameters**: 426,343  
**Trainable Parameters**: 426,343  
**Non-trainable Parameters**: 0  

- **[Model Notebook: Plant Disease Classification](https://github.com/Tanamore/machine_learning/blob/b0280b9a11cbf2319fee7f9313514dd916a12f9a/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/Model_1_Klasifikasi_Penyakit_Tanaman.ipynb)**: The notebook for training and evaluating the disease classification model.
- **[Revised Model Notebook](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/Model%201_Klasifikasi%20Penyakit%20Tanaman/Model1_Revisi_Fix.ipynb)**: A fixed version of the original model.
  
---

### Model 2: Plant Species Identification
This model uses transfer learning with **InceptionV3** as the backbone architecture. It leverages pre-trained features from the ImageNet dataset. The architecture details are as follows:

1. **Base Model**: InceptionV3 without the top fully connected layers (`include_top=False`) and pre-trained weights from ImageNet.
   - Extracted Layer: Output from `mixed7`.
   - Base model is frozen (non-trainable).
2. **Global Average Pooling**: Reduces the feature dimensions into a single vector.
3. **Fully Connected Layers**:
   - Dense layer with 128 units, ReLU activation, and Dropout (0.2).
   - Dense layer with the number of units equal to the number of classes (`num_classes`), Softmax activation.

**Total Parameters**: 9,076,534  
**Trainable Parameters**: 101,270  
**Non-trainable Parameters**: 8,975,264  

- **[Model Notebook: Plant Type Classification](https://github.com/Tanamore/machine_learning/blob/580efbe33156bcd051a0652383bc675934081575/machine_learning/model%202_klasifikasi%20jenis%20tanaman/Model_2_Klasifikasi_Jenis_Tanaman.ipynb)**: The notebook for training and evaluating the plant type classification model.
  
---

## Training Process

### Model 1: Plant Disease Detection

For Model 1, the training process involves the following configuration:

- **Optimizer**: The Adam optimizer with a learning rate of 0.001 was used for efficient training.
- **Loss Function**: The sparse categorical crossentropy loss function was used, suitable for multi-class classification tasks.
- **Metrics**: Accuracy was the primary evaluation metric during training.
- **Callbacks**:
  - **EarlyStopping**: This callback stops training early if the validation loss does not improve after 5 epochs to avoid overfitting.
  - **ModelCheckpoint**: The best model, based on the validation loss, is saved to prevent the model from training too long.
The model was trained for 30 epochs with a validation dataset and was continuously evaluated to track its progress.

### Model 2: Plant Species Identification

For Model 2, the training configuration is similar to Model 1 but tailored for plant species classification:

- **Optimizer**: Adam optimizer was used with a learning rate of 0.001.
- **Loss Function**: Sparse categorical crossentropy, as Model 2 is a multi-class classification task.
- **Metrics**: The accuracy metric was used to evaluate the performance during training.
- **Callbacks**:
  - **ModelCheckpoint**: This callback saves the best model based on validation accuracy.
Model 2 was trained for 20 epochs with the validation data, ensuring the best performing model was retained.

---

## Evaluation and Results

### Model 1: Plant Disease Detection

After training, Model 1 was evaluated using the following metrics:

- **Accuracy**: The model achieved 97.66% accuracy on the validation set and 97.65% on the test set, demonstrating that the model consistently performed well across different datasets.
- **Precision**: The model showed high precision, indicating that it correctly identified plant diseases while minimizing false positives.
- **Recall**: A strong recall score indicates that the model was able to correctly identify a high percentage of true positives (plants with diseases).
- **F1-Score**: The F1-Score, which is the harmonic mean of precision and recall, further confirmed the model's balanced performance.
  
Additionally, we visualized the training and validation performance over epochs. The graphs below show how the model improved during training, as well as the comparison between training and validation accuracy and loss.
![Evaluation for Model 1](path_to_confusion_matrix_image_1.png)

To provide further insight into the model’s performance, we evaluated it using a **confusion matrix**, which indicates how well the model performed across all disease classes.
![Confusion Matrix for Model 1](path_to_confusion_matrix_image_1.png)

### Model 2: Plant Species Identification

- **Accuracy**: 97% on both the validation and test sets.
- **Precision, Recall, F1-Score**: Strong performance overall.

The graphs below show how the model improved during training, as well as the comparison between training and validation accuracy and loss.
![Evaluation for Model 2](path_to_confusion_matrix_image_1.png)

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

