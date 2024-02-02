# MELANOMA-DETECTION

![MELANOMA DETECTION](https://molechex.com.au/wp-content/uploads/2023/09/know-the-abcdes-of-melanoma-detection.jpg)

Welcome to the Melanoma Detection Project! This project utilizes machine learning techniques to predict the likelihood of a lesion being a cancer cell which might have bad consequences for the patient based on various health-related features. The dataset is used for training and testing the model.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [Results](#results)
## Introduction

An advanced machine learning-based repository for detecting and classifying skin lesions as benign or malignant, aiding in the early identification of potential melanoma cases.

## Dataset

The dataset is categorized into (`train.csv`) for training the model and (`test.csv`) for validating the model. The dataset contains images which are classified as a Melanoma lesion or some different lesion. The target image indicates whether it is a Melanoma lesion (1) or not (0).

## Technologies Used

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Jupyter Notebook (for model development and analysis)

## Usage

1. Open the Jupyter Notebook (`melanoma.ipynb`) to explore the dataset and understand the steps taken in the project.

2. Execute the notebook cells to load the data, preprocess it, train the machine learning model, and evaluate its performance.

## Model Training

The machine learning model is trained using the efficientnet library. The notebook includes code for data preprocessing, feature scaling, model selection, and training.

```python
# Example code snippet for model training
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image

from sklearn import model_selection
from sklearn import metrics

import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import efficientnet_pytorch

# importing data as a training set and testing set
train_dir = '../input/siic-isic-224x224-images/train'
test_dir = '../input/siic-isic-224x224-images/test'

# Create folds in the dataset
input_path = '../input/siim-isic-melanoma-classification/train.csv'
df = pd.read_csv(input_path)
df.head()

# fold is an integer ie if fold == that no then val else train
def train(fold):
    training_data_path = "../input/siic-isic-224x224-images/train/"
    df = pd.read_csv("/kaggle/working/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

# Model training using efficient net
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
            'efficientnet-b4'
        )
        self.base_model._fc = nn.Linear(
            in_features=1792, 
            out_features=1, 
            bias=True
        )
        
    def forward(self, image, targets):
        out = self.base_model(image)
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        return out, loss
model = EfficientNet()
model.to(device)

# prediction
pred = predict(0)

#Predictions are stored in submission.csv
predictions = pred
sample = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample.loc[:, "target"] = predictions
sample.to_csv("submission.csv", index=False)
```

## Evaluation

The submission .csv contains sections for images, where metrics where the images are classified if they have melanoma or not. Analyze these metrics to understand the model's performance.

## Results

The model was trained with the best neural network available for training those medical images the "Efficient Net" and the the accuracy is 91%.

---
