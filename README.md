# Covid-19-Prediction-ML-Project
This Project is used to Predict infection of covid-19 disease through various symtoms provided in data using various Machine learning models.
This project was donw by be during my bootcamp as a capstone project. 

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Methodologies](#models-and-methodologies)
- [Results](#results)

## Introduction

Predicting the Disease will help to prevent infection and early treatment. From this data, creating model from it with high accuracy would help in predicting weather the person can get infected with covid or not. This model will predict weather patient is Covid infected or not by statistically calculating correlation between the patients symptoms like Cough symptoms, Fever, Sore throat, Shortness of breath, Headache and covid test result. The test result could take time for results, but with the help of this model if patient data have valid symptoms we could predict if that person have Covid or not. Hence the treatement would begin much faster and if the person is still not infected we could prevent it earlier.

## Dataset

Dataset used in this project was provided by bootcamp. It contains 10 features containing symtoms and other details.

**Source** : https://drive.google.com/file/d/1--4ECy73DOjx754sIIRgf2Vxz5lEynTB/view?usp=drive_link.

**Features** : Tested Date, Cough Symtoms, Fever, Sore Throat, Shortness of Breath, Headache, Age above 60, Sex, Known contact.

**Target Variable** : Corona.

## Installation

To run this project, you will need to install the following dependencies:

libraries: 
```bash
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

```

## Usage
- Read Datset in the file in notebook using pd.read_cvs.

```markdown

df = pd.read_csv("/content/drive/MyDrive/ML/Corona Capstone Project /corona_tested_006.csv")
```

- Chi-squared Test
  ```markdown
  from scipy import stats
  ```
- k-folk cross validation
    ```markdown
    from sklearn.model_selection import cross_val_score
    cv_score=cross_val_score(rfc,x_train,y_train,scoring="f1",cv=5)
    cv_score
    ```
- Model training
  ```markdown
  from sklearn.linear_model import LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
    dtc=DecisionTreeClassifier()
  from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier()
  from xgboost import XGBClassifier
    xgb=XGBClassifier()
  ```
## Models and Methodologies

In this project, we used the following models:
- Models
  - Linear Regression
  - Decision Tree
  - Random Forest

- Preprocessing
  - Handling missing data
    Simple Imputor
  - Feature engineering
    Recursive feature elemination
  -Scaling and normalization
    One hot encoder
  - Model Tuning
    K-fold cross validation.

## Results
  Best Model: XGBoost Classifier
  
  XG Boost Classifier f1 score = 0.6404949861318542
  
  XG Boost Classifier Recall score = 0.5563380281690141
  
  ![image](https://github.com/user-attachments/assets/bde10cee-26ce-43f9-b682-f4ac951e46b2)
  
  ![image](https://github.com/user-attachments/assets/63586e00-6b45-4f59-98e1-f06824a23232)

  XGBoost is most suitable for this model with highest f1_score, precision and recall score. Here the model learns slowly and errors are improved model by model as it builds sequentially. As the errors are improved slowly, it recuces the problem of underfitting and overfitting and gradually improves the model.



