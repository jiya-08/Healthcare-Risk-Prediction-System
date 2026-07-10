# Healthcare-Risk-Prediction-System
Healthcare Risk Prediction System is a machine learning-based web application that predicts patient risk using healthcare data such as age, length of stay, and treatment cost. Built with Python, Scikit-learn, and Streamlit, it provides real-time predictions through an intuitive interface to support healthcare decision-making.
📌 Project Overview

The Hospital Risk Prediction Model is a Machine Learning-based healthcare application developed to predict the risk level of hospitalized patients based on their medical information. The system helps healthcare professionals identify patients who may require additional monitoring or immediate medical attention.

The project combines Python, Machine Learning, and Streamlit to provide an interactive web application where users can enter patient details and receive instant risk predictions.

🎯 Objectives
Predict patient risk using Machine Learning.
Improve decision-making in hospitals.
Reduce treatment delays by identifying high-risk patients.
Provide a simple and interactive prediction interface.
🚀 Features
Patient Risk Prediction
User-friendly Streamlit Interface
Real-time Prediction
Logistic Regression Machine Learning Model
Healthcare Dataset Analysis
Data Visualization
ROC Curve Performance Evaluation
🛠 Technologies Used
Technology	Purpose
Python	Programming Language
Pandas	Data Processing
NumPy	Numerical Computation
Scikit-learn	Machine Learning
Matplotlib	Data Visualization
Streamlit	Web Application
Joblib	Model Serialization
📊 Dataset Features

The model uses the following patient information:

Age
Length of Stay
Treatment Cost
Target Variable
Outcome Encoded (Patient Risk Classification)
🤖 Machine Learning Model

The project uses the Logistic Regression algorithm for binary classification.

Workflow
Data Collection
Data Cleaning
Feature Selection
Train-Test Split
Model Training
Model Evaluation
Risk Prediction
Deployment using Streamlit
📈 Model Evaluation

The model performance is evaluated using:

Classification Report
Confusion Matrix
ROC Curve
AUC Score
Accuracy Score
💻 Project Structure
Healthcare-Risk-Prediction/
│
├── Risk_Model.py              # Streamlit Application
├── Risk_Model1.ipynb          # Model Training Notebook
├── Risk_Model.pkl             # Saved Machine Learning Model
├── patients.csv
├── diagnoses.csv
├── labs.csv
├── outcomes.csv
├── requirements.txt
└── README.md
