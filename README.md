# 1. Introduction

Breast cancer is the most common form of cancer in women. Accurately identifying and categorizing breast cancer subtypes is an important clinical task, and automated methods can be used to save time and reduce errors.
In this work, we used Keras to build a CNN called "CancerNet" that is able to accurately classify a histology image as benign or malignant.

# 2. Dataset
The dataset used is the histology image dataset for Invasive Ductal Carcinoma (IDC), the most common breast cancer and available in public domain on Kaggle'website. It contains a total of 277,524 patches of 50x50 pixels from 279 different patients, including:
- 198,738 negative examples (i.e., no breast cancer)
- 78,786 positive examples (i.e., breast cancer found)

Each image in the dataset has a specific filename structure. An example of it is: **10253_idx5_x1351_y1101_class0.png**, where:
- patient ID: **10253_idx5**
- x-coordinate of the crop: **1,351**
- y-coordinate of the crop: **1,101**
- class label: **0** (**0** indicates no IDC while **1** indicates IDC)

# 3. How to run
## 3.1 Obtaining dataset
First you need to download the IDC dataset from [here](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images). 
If you would like to use only a part of the original dataset, check the options ``python get_dataset.py --help``
In this work, we use a small part of the original dataset containing 32,713 images, including 15,851 negative examples (no breast cancer) and 4859 positive examples (breast cancer found) and divided into 3 sets:
- training set: 20,584 images
- validation set: 5,146 images
- test set: 6,433 images

To checks for splitting dataset into training, validation and test sets:
``python build_dataset.py --help``

## 3.2 Training and evaluating the model
To change hyperparameters for model training, checks:
``python train_model.py --help``
The evaluation of the model is automatically available after the model has been trained.
