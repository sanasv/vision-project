# vision-project

# 🧠 Brain Tumor Classification using Deep Learning & Ensemble Transformers

This project focuses on the classification of brain MRI images into four categories: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, and `pituitary_tumor`. We leverage modern convolutional neural networks (CNNs) and vision transformers (ViT, Swin) and explore **ensemble learning** to boost performance.

## 📁 Dataset

We used a publicly available brain tumor MRI dataset, pre-divided into:

- `Training/` folder
- `Testing/` folder

Each contains subfolders corresponding to the 4 classes.

## 🧠 Models Used

We experimented with the following models:
- ResNet18
- MobileNetV2
- DenseNet121
- ConvNeXt
- Vision Transformer (ViT)
- Swin Transformer

## 🏗️ Project Pipeline

1. **Data Preprocessing**
   - Resize images to `224x224`
   - Apply augmentations: random flip, rotation, color jitter, Gaussian noise
   - Normalize using ImageNet statistics

2. **Train/Test Split**
   - 80% training, 20% validation split
   - Separate held-out test set used for final evaluation

3. **Training**
   - All models trained using cross-entropy loss
   - Early stopping and loss curve tracking implemented
   - ViT and Swin fine-tuned on pre-trained weights (`transformers`)

4. **Evaluation Metrics**
   - Accuracy
   - Macro F1-score
   - Confusion Matrix

5. **Ensemble Learning**
   - **Weighted average ensemble** tested with incremental combinations of top-performing models
   - Final ensemble used ViT and Swin Transformer with tuned weights

## 📊 Results

| Model           | Accuracy (%) | F1 Score (%) |
|----------------|--------------|--------------|
| **ViT**         | **81.98**     | **81.33**     |
| Swin Transformer | 76.14        | 73.36        |
| ResNet18       | 75.63        | 71.64        |
| MobileNetV2    | 74.11        | 70.69        |
| DenseNet121    | 71.83        | 67.28        |
| ConvNeXt       | 57.61        | 55.91        |

### ✅ Final Ensemble (ViT + Swin)
- **Accuracy:** `83.2%`
- **F1 Score:** `82.1%`
- Ensemble resulted in improved generalization and better performance on underrepresented classes like glioma.

## 📊 Visualization

- Loss curves for each model
- Confusion matrix for final ensemble

<p align="center">
  <img src="ensemble (1).png" width="500"/>
</p>
