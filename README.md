# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Project Overview

Customer churn refers to customers discontinuing the use of a company's products or services. Predicting churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

This project uses Machine Learning techniques to predict whether a telecom customer is likely to churn based on customer demographics, account details, and service-related information.

The project includes:
- Data preprocessing and analysis
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Model evaluation and comparison
- Customer churn prediction
- Explainable AI using SHAP
- Interactive Streamlit dashboard

---

## 🎯 Objectives

- Analyze customer behavior and churn patterns
- Build classification models to predict customer churn
- Compare different Machine Learning algorithms
- Identify important factors affecting churn
- Provide an interactive interface for making predictions

---

## 📂 Dataset

Dataset Used:

**IBM Telco Customer Churn Dataset**

The dataset contains customer information including:

- Customer demographics
- Tenure
- Internet services
- Contract type
- Payment methods
- Monthly charges
- Total charges
- Churn status

Dataset Size:
# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Project Overview

Customer churn refers to customers discontinuing the use of a company's products or services. Predicting churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

This project uses Machine Learning techniques to predict whether a telecom customer is likely to churn based on customer demographics, account details, and service-related information.

The project includes:
- Data preprocessing and analysis
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Model evaluation and comparison
- Customer churn prediction
- Explainable AI using SHAP
- Interactive Streamlit dashboard

---

## 🎯 Objectives

- Analyze customer behavior and churn patterns
- Build classification models to predict customer churn
- Compare different Machine Learning algorithms
- Identify important factors affecting churn
- Provide an interactive interface for making predictions

---

## 📂 Dataset

Dataset Used:

**IBM Telco Customer Churn Dataset**

The dataset contains customer information including:

- Customer demographics
- Tenure
- Internet services
- Contract type
- Payment methods
- Monthly charges
- Total charges
- Churn status

Dataset Size:
# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Project Overview

Customer churn refers to customers discontinuing the use of a company's products or services. Predicting churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

This project uses Machine Learning techniques to predict whether a telecom customer is likely to churn based on customer demographics, account details, and service-related information.

The project includes:
- Data preprocessing and analysis
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Model evaluation and comparison
- Customer churn prediction
- Explainable AI using SHAP
- Interactive Streamlit dashboard

---

## 🎯 Objectives

- Analyze customer behavior and churn patterns
- Build classification models to predict customer churn
- Compare different Machine Learning algorithms
- Identify important factors affecting churn
- Provide an interactive interface for making predictions

---

## 📂 Dataset

Dataset Used:

**IBM Telco Customer Churn Dataset**

The dataset contains customer information including:

- Customer demographics
- Tenure
- Internet services
- Contract type
- Payment methods
- Monthly charges
- Total charges
- Churn status

Dataset Size:
# 📊 Customer Churn Prediction using Machine Learning

## 🚀 Project Overview

Customer churn refers to customers discontinuing the use of a company's products or services. Predicting churn helps businesses identify customers who are likely to leave and take preventive actions to improve customer retention.

This project uses Machine Learning techniques to predict whether a telecom customer is likely to churn based on customer demographics, account details, and service-related information.

The project includes:
- Data preprocessing and analysis
- Exploratory Data Analysis (EDA)
- Machine Learning model training
- Model evaluation and comparison
- Customer churn prediction
- Explainable AI using SHAP
- Interactive Streamlit dashboard

---

## 🎯 Objectives

- Analyze customer behavior and churn patterns
- Build classification models to predict customer churn
- Compare different Machine Learning algorithms
- Identify important factors affecting churn
- Provide an interactive interface for making predictions

---

## 📂 Dataset

Dataset Used:

**IBM Telco Customer Churn Dataset**

The dataset contains customer information including:

- Customer demographics
- Tenure
- Internet services
- Contract type
- Payment methods
- Monthly charges
- Total charges
- Churn status

Dataset Size:
rows:7043
columns:21  

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly
- SHAP
- Joblib

### Tools

- Jupyter Notebook
- VS Code
- Git & GitHub

### Application Interface

- Streamlit

---

## 🔄 Project Workflow
Data Collection
|
↓
Data Cleaning
|
↓
Exploratory Data Analysis
|
↓
Feature Engineering
|
↓
Data Preprocessing
|
↓
Model Training
|
↓
Model Evaluation
|
↓
Streamlit Application
|
↓
SHAP Explainability

---

# 📊 Exploratory Data Analysis

Performed analysis on:

- Churn distribution
- Contract type vs churn
- Monthly charges vs churn
- Tenure vs churn
- Payment method analysis
- Internet service patterns

### Key Insights

- Customers with month-to-month contracts show higher churn probability.
- New customers with lower tenure are more likely to churn.
- Monthly charges and contract type strongly influence churn.

---

# 🤖 Machine Learning Models

The following classification models were trained:

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline classification model |
| K-Nearest Neighbors | Distance-based classifier |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble learning model |
| Support Vector Machine | Classification using decision boundaries |

---

# 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The best-performing model was saved using Joblib and used in the Streamlit application.

---

# 🔍 Explainable AI using SHAP

SHAP (SHapley Additive exPlanations) was used to understand model predictions.

It helps identify:

- Important features affecting churn
- Factors contributing to customer decisions
- Model behavior interpretation

Important churn-related features include:

- Contract type
- Tenure
- Monthly charges
- Internet service
- Payment method

---

# 🌐 Streamlit Application

A Streamlit-based interface was created for local use.

Features:

✅ Enter customer details  
✅ Predict churn outcome  
✅ Display prediction probability  
✅ Visualize important churn factors  
✅ Provide model-based insights  

Run the application locally:

```bash
streamlit run app/app.py



📁 Project Structure:
Customer-Churn-Prediction/

│
├── app/
│   └── app.py
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── images/
│   └── shap_explainibility_summary.png
│
├── notebook/
│   └── churn_prediction.ipynb
│
├── models/
│   └── churn_model.pkl
│
├── requirements.txt
│
└── README.md
nstallation & Setup

Clone the repository:

git clone https://github.com/Saisruti-Mohanty/customer-churn-prediction.git

Go to the project directory:

cd customer-churn-prediction

Install required libraries:

pip install -r requirements.txt

Run Streamlit application:

streamlit run app/app.py

🚀 Future Improvements
Deploy the application using cloud platforms
Build REST API using Flask/FastAPI
Add customer retention recommendations
Experiment with advanced models like XGBoost and Neural Networks
Integrate real-time customer data

👩‍💻 Author:

Saisruti Mohanty

B.Tech Computer Science & Data Science
Machine Learning | Data Science | Artificial Intelligence

GitHub:
https://github.com/Saisruti-Mohanty


This version is more accurate for your current GitHub state: **ML model + Streamlit app locally + SHAP explainability, but not deployed.**
