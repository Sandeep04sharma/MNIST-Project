# MNIST-Project
## Steps

1. Introduction

2. Data Exploration
  *  Libraries used
  *  Reading data
  *  Understand data
3. Data Preprocessing
* Feature Scaling / Normalization
* Label Encoding
4. Build Models
* SVM
* KNeighbors
* Random Forest
* Neural Network
5. Evaluate Models
* Cross Validation
6. Hyperparameter Tuning

7. Predict and Submit
  * Confusion Matrix
  * Precision, Recall and F1 Scores
  * Predict and Submit Results
### 1. Introduction
Hello everyone! I started this project right after I finished reading on Classification, and since they say "MNIST is the hello world of classification", so I did this.
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
```python
mnist_train=pd.read_csv('mnist_train.csv')
mnist_test=pd.read_csv('mnist_test.csv')
```
