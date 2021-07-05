# MNIST-Project
## Steps

1. Introduction

2. Data Exploration
  *  Libraries used
  *  Reading data
  *  Understand data
3. Data Preprocessing
* Feature Scaling / Normalization
* Suffeling
4. Build Models
* Logistic regression
* SVM
5. Evaluate Models
* Cross Validation
6. Predict and result
  * Confusion Matrix
  * Precision, Recall and F1 Scores
### 1. Introduction
Hello everyone! I started this project right after I finished reading on Classification, and since they say "MNIST is the hello world of classification", so I did this.

The MNIST database of handwritten digits, available from [this page](https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_test.csv), has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.
### 2. Data exploration
#### Libraries used
![image](https://user-images.githubusercontent.com/54997938/124473662-386a9a80-ddbd-11eb-85ad-b3fc9882e6ca.png)

`code()`
```python
import numpy as np
import pandas as pd 
%matplotlib inline 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
```
#### Reading data
You can download data from [here](https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_test.csv)
```python
mnist_train=pd.read_csv('mnist_train.csv')
mnist_test=pd.read_csv('mnist_test.csv')
```
### 3. Data Preprocessing
Working with numerical data that is in between 0-1 is more effective for most of the machine learning algortihms than 0-255.
We can easily scale our features to 0-1 range by dividing to max value (255).
### 4. Build model
##### 4.1 Logistic regression
We are going to create the *LogisticRegression* model.

We are going to call fit() method with training data.
##### 4.2 SVM
We are going to create the *SVM* model.

We are going to call fit() method with training data.

### 5. Evaluate Models
```python
from sklearn.metrics import accuracy_score
LR_Pred_y=LR_clf.predict(x_test)
print("LR Accuracy:", accuracy_score(y_true=y_test_2_det ,y_pred=LR_Pred_y))

```
### 6. Predict and result/Confusion Matrix
```python
from sklearn.metrics import classification_report
print(classification_report(y_test_2_det,LR_Pred_y))  
```
