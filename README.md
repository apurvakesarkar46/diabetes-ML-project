# 🩺 Diabetes Prediction using Machine Learning

## 📖 Overview

Diabetes is a major chronic health condition that can lead to serious long-term complications when it is not identified and managed appropriately. Early identification of potential diabetes risk can support timely medical evaluation and preventive healthcare practices.

This project presents an **end-to-end Machine Learning solution for diabetes prediction** using patient medical and clinical information. The model is developed using the **Support Vector Classifier (SVC)** algorithm and trained on the **Pima Indians Diabetes Dataset** to classify patients into diabetic and non-diabetic categories.

The project demonstrates the complete **Machine Learning lifecycle**, starting from data collection and exploration, followed by data preprocessing, exploratory data analysis, feature selection, feature standardization, model training, prediction, and performance evaluation.

The implementation is developed using **Python, Pandas, NumPy, Matplotlib, and Scikit-learn**, with experimentation and model development carried out using **Jupyter Notebook and Google Colab**.

---

## 🎯 Objectives

The main objectives of this project are:

- Develop a Machine Learning model for diabetes prediction.
- Analyze patient medical and clinical data.
- Perform data exploration and preprocessing.
- Identify patterns and relationships within the dataset.
- Perform Exploratory Data Analysis (EDA) using visualizations.
- Select relevant features for model training.
- Standardize numerical features using `StandardScaler`.
- Train a Support Vector Classifier (SVC).
- Evaluate model performance using classification metrics.
- Build a clear and reproducible Machine Learning workflow.

---

## 🧩 Problem Statement

Diabetes can develop gradually, and identifying potential risk at an early stage can be important for timely medical evaluation.
The objective of this project is to develop a **binary classification Machine Learning model** that analyzes selected patient health parameters and predicts whether a patient belongs to the diabetic or non-diabetic category.

The project demonstrates how structured healthcare data can be processed and used to build a supervised Machine Learning classification model.

---


📊 Dataset 

Dataset: [diabetes.csv](https://github.com/apurvakesarkar46/diabetes-ML-project/blob/main/diabetes.csv)

The dataset consists of medical records collected from female patients of Pima Indian heritage.



### Dataset Features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | 2-Hour serum insulin |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age of the patient |
| `Outcome` | Diabetes classification |

### Target Variable

| Value | Meaning |
|---|---|
| `0` | Non-Diabetic |
| `1` | Diabetic |

---

## 🔍 Project at a Glance

| Component | Details |
|---|---|
| **Problem Type** | Binary Classification |
| **Dataset** | Pima Indians Diabetes Dataset |
| **Input** | Clinical and demographic parameters |
| **Target Variable** | `Outcome` |
| **Classes** | Non-Diabetic / Diabetic |
| **Machine Learning Algorithm** | Support Vector Classifier (SVC) |
| **Preprocessing** | Data preparation and feature standardization |
| **Visualization** | Exploratory Data Analysis and Histograms |
| **Evaluation** | Accuracy Score and Classification Metrics |
| **Programming Language** | Python |
| **Development Environment** | Jupyter Notebook / Google Colab |

---

## 🛠️ Technology Stack

### Programming Language
- Python

### Libraries

- **NumPy** – Numerical computations
- **Pandas** – Data manipulation and analysis
- **Matplotlib** – Data visualization
- **Scikit-learn** – Machine Learning and model evaluation

### Development Tools

- Jupyter Notebook
- Google Colab
- Git & GitHub

---

## 📚 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


⚙️ Project Workflow

1. Data Collection
Imported the Pima Indians Diabetes Dataset.
Loaded the dataset using Pandas.

2. Data Exploration

Performed initial dataset analysis using:

shape()
info()
describe()
value_counts()
isnull().sum()

3. Data Preprocessing
Checked missing values
Verified data types
Removed inconsistencies
Prepared the dataset for training

4. Exploratory Data Analysis (EDA)

Visualized each feature using histograms to understand the distribution and identify patterns.

Features analyzed:

Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Outcome

5. Feature Selection
Independent Variables (X)
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Dependent Variable (y)

Outcome

6. Data Standardization

The features were standardized using StandardScaler to improve the performance of the Support Vector Classifier.

7. Train-Test Split
Training Data : 80%
Testing Data : 20%
Random State : 42

8. Model Building
Algorithm Used

Support Vector Classifier (SVC)

Why Support Vector Classifier?
High classification accuracy
Performs well on numerical datasets
Works effectively after feature scaling
Finds the optimal decision boundary
Suitable for binary classification problems


9. Model Training
model = SVC()

model.fit(X_train, y_train)


10. Model Prediction
prediction = model.predict(X_test)


11. Model Evaluation

Performance Metric Used

Accuracy Score
accuracy_score(y_test, prediction)

Additional evaluation metrics that can be included:

Precision
Recall
F1 Score
Confusion Matrix



**📈 Project Pipeline**
Dataset

↓

Data Cleaning

↓

Exploratory Data Analysis

↓

Feature Selection

↓

Data Standardization

↓

Train-Test Split

↓

Support Vector Classifier (SVC)

↓

Model Evaluation

↓

Prediction



✨ Features
End-to-End Machine Learning Project
Data Cleaning and Preprocessing
Exploratory Data Analysis
Feature Scaling
Model Training
Model Evaluation
Diabetes Prediction
Beginner-Friendly Code
Well-Structured Workflow



📂 Project Structure
Diabetes-Prediction-ML/

│
├── diabetes.csv
├── Diabetes_Prediction.ipynb
├── README.md
├── requirements.txt
├── LICENSE
│
├── images/
│   ├── glucose_histogram.png
│   ├── bmi_histogram.png
│   ├── age_histogram.png
│   └── workflow.png
│
└── outputs/
    ├── prediction_results.png
    └── accuracy_score.png


🚀 Installation

Clone the repository

git clone https://github.com/apurvakesarkar46/diabetes-ML-project.git

Move to the project directory

cd Diabetes-Prediction-ML

Install dependencies

pip install -r requirements.txt

Run Jupyter Notebook

jupyter notebook

or open the notebook directly in Google Colab.


📋 Requirements
numpy
pandas
matplotlib
scikit-learn
jupyter



💻 Future Enhancements
Hyperparameter tuning using GridSearchCV
Compare multiple Machine Learning algorithms
Develop a Streamlit or Flask web application
Deploy the model on cloud platforms
Add user-friendly prediction interface
Improve model performance through feature engineering
Integrate real-time healthcare data


📚 Learning Outcomes
This project helped in understanding:
Data preprocessing techniques
Exploratory Data Analysis (EDA)
Data visualization using Matplotlib
Feature scaling using StandardScaler
Machine Learning classification
Support Vector Machines
Model evaluation techniques
Building an end-to-end Machine Learning pipeline


🌍 Real-World Applications
Early diabetes risk assessment
Clinical decision support systems
Healthcare analytics
Medical research
AI-powered health monitoring systems
Educational Machine Learning projects

👨‍💻 Author

--Apurva Kesarkar--

Computer Science Engineer

🔗 Connect & Project Post

LinkedIn Project Post: https://www.linkedin.com/posts/apurva-kesarkar-8004a5422_machinelearning-artificialintelligence-python-activity-7484217885725204480-MpjQ?utm_source=share&utm_medium=member_android&rcm=ACoAAGs3IEABH3ErDcp0ssDv3sRp0iEn2ww-tA4

LinkedIn Profile: https://www.linkedin.com/in/apurva-kesarkar-8004a5422

GitHub: https://github.com/apurvakesarkar46

Interests
Artificial Intelligence
Machine Learning
Data Science
Healthcare AI
Python Development
