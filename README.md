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

````


## 🔄 Machine Learning Workflow

### 1. 📥 Data Collection

- Imported the **Pima Indians Diabetes Dataset**.
- Loaded the dataset using **Pandas** for further analysis and processing.

---

### 2. 🔍 Data Exploration

Performed an initial analysis of the dataset using:

- `shape`
- `info()`
- `describe()`
- `value_counts()`
- `isnull().sum()`

This helped understand the dataset structure, feature types, statistical summary, and data quality.

---

### 3. 🧹 Data Preprocessing

Prepared the dataset for Machine Learning by:

- Checking for missing values
- Verifying data types
- Identifying and handling inconsistencies
- Preparing the data for model training

---

### 4. 📊 Exploratory Data Analysis (EDA)

Used **histograms** to visualize feature distributions and identify patterns within the dataset.

**Features analyzed:**

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome

---

### 5. 🎯 Feature Selection

Separated the dataset into independent and dependent variables.

**Independent Variables (`X`):**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Dependent Variable (`y`):**
- Outcome

---

### 6. ⚖️ Data Standardization

Applied `StandardScaler` to standardize the input features before training the Support Vector Classifier.

This ensures that features with different numerical ranges are placed on a comparable scale.

---

### 7. ✂️ Train-Test Split

The dataset was divided into:

| Dataset | Percentage |
|---|---:|
| **Training Data** | 80% |
| **Testing Data** | 20% |
| **Random State** | 42 |

---

### 8. 🤖 Model Building

**Algorithm Used:** Support Vector Classifier (SVC)

**Why SVC?**

- Suitable for binary classification
- Effective with scaled numerical features
- Identifies an optimal decision boundary
- Suitable for classification problems with multiple input features

---

### 9. 🏋️ Model Training

The SVC model was trained using the training dataset:

```python
model = SVC()

model.fit(X_train, y_train)
```




**📈 Project Pipeline**
           
```text
┌──────────────────────────────┐
│           Dataset            │
│     Pima Diabetes Data       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Data Exploration        │
│    Structure & Summary       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Data Preprocessing       │
│      Cleaning & Validation   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Exploratory Data Analysis    │
│            (EDA)             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Feature Selection       │
│   X → Input Features         │
│   y → Outcome                │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Standardization        │
│       StandardScaler         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Train-Test Split       │
│            80 / 20           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│         SVC Model            │
│          Training            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│         Prediction           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Model Evaluation       │
│        Accuracy Score        │
└──────────────────────────────┘

```




## ✨ Features

This project includes the following key features:

- **End-to-End Machine Learning Project**  
  Covers the complete Machine Learning workflow from data preparation to prediction and evaluation.

- **Data Cleaning and Preprocessing**  
  Prepares the dataset by checking missing values, data types, and inconsistencies before model training.

- **Exploratory Data Analysis (EDA)**  
  Uses data analysis and visualizations to understand feature distributions and identify patterns within the dataset.

- **Feature Scaling**  
  Applies `StandardScaler` to standardize numerical features before training the SVC model.

- **Machine Learning Model Training**  
  Implements a **Support Vector Classifier (SVC)** for binary classification.

- **Model Evaluation**  
  Evaluates the trained model using **Accuracy Score** and provides scope for additional classification metrics.

- **Diabetes Prediction**  
  Predicts whether a patient belongs to the diabetic or non-diabetic class based on the provided input features.

- **Beginner-Friendly Code**  
  Uses clear and structured Python code that is easy to understand and follow.

- **Well-Structured Workflow**  
  Organizes the project into logical stages including data exploration, preprocessing, EDA, feature selection, model training, prediction, and evaluation.

  

### 🔑 Key Highlights

| Feature | Description |
|---|---|
| **Machine Learning** | Support Vector Classifier (SVC) |
| **Data Processing** | Cleaning, validation, and preprocessing |
| **EDA** | Feature analysis and visualization |
| **Feature Scaling** | StandardScaler |
| **Classification** | Binary classification |
| **Prediction** | Diabetic / Non-Diabetic |
| **Evaluation** | Accuracy Score |
| **Implementation** | Python |
| **Environment** | Jupyter Notebook / Google Colab |



## 📂 Project Structure

The repository is organized as follows:

```text
Diabetes-Prediction-ML/
│
├── 📄 diabetes.csv
├── 📓 Diabetes_Prediction.ipynb
├── 📖 README.md
├── 📦 requirements.txt
├── ⚖️ LICENSE
│
├── 📁 images/
│   ├── 📊 glucose_histogram.png
│   ├── 📊 bmi_histogram.png
│   ├── 📊 age_histogram.png
│   └── 🔄 workflow.png
│
└── 📁 outputs/
    ├── 📈 prediction_results.png
    └── 📊 accuracy_score.png
```


## 🚀 Installation

Follow the five steps below to set up and run the project.

### 1️⃣ Clone the Repository

Clone the project from GitHub using:

```bash
git clone : [https://github.com/apurvakesarkar46/diabetes-ML-project.git](https://r.search.yahoo.com/_ylt=A2RTF.M7OcFqHAMAnLtXNyoA;_ylu=Y29sbwNhcC1zb3V0aGVhc3QtMQRwb3MDMgR2dGlkAwRzZWMDc3I-/RV=2/RE=1792257596/RO=10/RU=https%3a%2f%2fgithub.com%2fapurvakesarkar46%2fdiabetes-ML-project/RK=2/RS=Gw6OQ.iCbGPiC01mno8ItRmkLlo-)

```

### 2️⃣ Navigate to the Project Directory
```bash
cd Diabetes-Prediction-ML

```
### 3️⃣ Install Dependencies

Install all the required Python libraries using the requirements.txt file:
```bash
pip install -r requirements.txt

```

### 4️⃣ Launch Jupyter Notebook

Start Jupyter Notebook from the project directory:
```bash
jupyter notebook
```
Then open:
```bash
Diabetes_Prediction.ipynb
```
Run the notebook cells sequentially to execute the Machine Learning workflow.

### 5️⃣ Run Using Google Colab

Alternatively, you can open Diabetes_Prediction.ipynb directly in Google Colab and execute the notebook without setting up Jupyter Notebook locally.

Note: Make sure Python and pip are installed and configured on your system before following the installation steps.



The project requires the following Python libraries:

| Library | Purpose |
|---|---|
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data loading, data manipulation, cleaning, and analysis |
| **Matplotlib** | Data visualization and Exploratory Data Analysis (EDA) |
| **Scikit-learn** | Data preprocessing, feature scaling, model training, and evaluation |
| **Jupyter Notebook** | Interactive development and execution of the Machine Learning workflow |





## 💻 Future Enhancements

- 🔧 Hyperparameter tuning using `GridSearchCV`
- 🤖 Compare multiple Machine Learning algorithms
- 🌐 Develop a Streamlit or Flask web application
- ☁️ Deploy the model on cloud platforms
- 🖥️ Add a user-friendly prediction interface
- 🧠 Improve performance through feature engineering
- 🔄 Integrate real-time healthcare data







## 📚 Learning Outcomes

This project provided practical experience in:

- 🧹 Data preprocessing and cleaning
- 📊 Exploratory Data Analysis (EDA)
- 📈 Data visualization using Matplotlib
- ⚖️ Feature scaling with `StandardScaler`
- 🤖 Machine Learning classification
- 🎯 Support Vector Machines (SVM)
- 📏 Model evaluation techniques
- 🔄 Building an end-to-end Machine Learning pipeline







## 🌍 Real-World Applications

- 🩺 **Diabetes Risk Assessment**
- 🏥 **Healthcare Analytics**
- 🔬 **Medical Research**
- 🤖 **AI-Powered Health Monitoring**
- 📊 **Clinical Data Analysis**
- 🎓 **Machine Learning Education & Research**








## 👨‍💻 Author

### **Apurva Kesarkar**
**Computer Science Engineer**

---

## 🔗 Connect With Me

- **LinkedIn Profile:** [Apurva Kesarkar](https://www.linkedin.com/in/apurva-kesarkar-8004a5422)
- **GitHub:** [apurvakesarkar46](https://github.com/apurvakesarkar46)
- **Project Post:** [View Project on LinkedIn](https://www.linkedin.com/posts/apurva-kesarkar-8004a5422_machinelearning-artificialintelligence-python-activity-7484217885725204480-MpjQ)

---
