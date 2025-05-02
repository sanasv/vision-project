
# 🧠 Brain Tumor Classification using Deep Learning & Ensemble Learning

## 🌟 Overview
This project focuses on the classification of brain MRI images into four categories: `glioma_tumor`, `meningioma_tumor`, `no_tumor`, and `pituitary_tumor`. We leverage modern convolutional neural networks (CNNs) and vision transformers (ViT, Swin) and explore **ensemble learning** to boost performance.

## 📁 Dataset

We used a publicly available brain tumor MRI dataset from Kaggle, pre-divided into:

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

## ✅ Ensemble Learning

### Weighted Ensemble Learning
<p align="center">
  <img src="ensemble (1).png" width="500"/>
</p>

### Majority Voting Ensemble Learning
<p align="center">
  <img src="ensemble_majority_voting (1).png"
" width="500"/>
</p>

## 🔑 Key Findings
- Vision Transformer outperforms all other models, demonstrating superior feature extraction and class balance.
- Weighted averaging ensembles effectively enhance sensitivity for glioma classification.
- Majority voting dilutes the impact of high-performing models and is less effective.
- Adding weaker models like ConvNeXt to ensembles degrades performance, emphasizing the importance of quality over quantity.

# 🏃‍♀️ Running the Project

Each .ipynb file handles loading the dataset, training, and evaluating the model.

## To run:

📌 Open the notebook of the desired model (e.g., ViT.ipynb, ResNet18.ipynb) and run all cells sequentially.

🔁 Ensemble Learning
To run the ensemble evaluation:

📌 Open and run ensemble_<method>.ipynb.

It evaluates model combinations using weighted averaging and majority voting, and produces accuracy, F1 score, and confusion matrix.

## 📦 Model Files

🚫 Due to file size limitations, the trained model .pth files are not uploaded to this repository.
You will need to re-train the models using the provided training notebooks before evaluation or ensemble.

