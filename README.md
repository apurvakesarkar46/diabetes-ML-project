🩺 Diabetes Prediction using Machine Learning

📖 Overview

Diabetes is one of the fastest-growing chronic diseases worldwide. Early detection of diabetes enables timely medical intervention and helps reduce long-term health complications.

This project presents an end-to-end Machine Learning solution for predicting diabetes using patient medical information. The model is built using the Support Vector Classifier (SVC) algorithm and trained on the Pima Indians Diabetes Dataset.

The project follows the complete Machine Learning lifecycle—from data preprocessing and exploratory data analysis to model training, evaluation, and prediction.


🎯 Project Objectives
Predict diabetes based on patient health records.
Perform data cleaning and preprocessing.
Explore and visualize the dataset using EDA.
Train a classification model using Support Vector Classifier (SVC).
Evaluate the model using classification metrics.
Build a reliable and reusable prediction system.


❓ Problem Statement

Diabetes is often diagnosed only after symptoms become severe. Healthcare professionals require efficient methods to assist in identifying individuals at risk.
The objective of this project is to develop a Machine Learning model capable of predicting diabetes based on clinical parameters, supporting early diagnosis and preventive healthcare.


📊 Dataset Information

Dataset: Pima Indians Diabetes Dataset

The dataset consists of medical records collected from female patients of Pima Indian heritage.

Features

Pregnancies	- Number of pregnancies

Glucose	- Plasma glucose concentration

BloodPressure	- Diastolic blood pressure

SkinThickness	- Skin fold thickness

Insulin	- Serum insulin level

BMI	- Body Mass Index

DiabetesPedigreeFunction - 	Diabetes hereditary score

Age	- Patient age

Outcome - 	Target variable (0 = Non-Diabetic, 1 = Diabetic)


🛠 Tech Stack
Python
Google Colab
NumPy
Pandas
Matplotlib
Scikit-learn
Jupyter Notebook


📦 Python Libraries Used
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
