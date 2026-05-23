#  Diamond Pricing Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient_Boosting-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)

> Advanced machine learning dashboard for diamond price prediction, model comparison, feature importance analysis, and business intelligence insights.

---

# Overview

**Live Project Repository:**
[Diamond Price Prediction](https://github.com/Hamzah-20/diamond-price-prediction-dashboard)

Diamond Pricing Intelligence Dashboard is an end-to-end machine learning application designed to predict diamond prices using advanced regression models and interactive business analytics.

The platform combines:

* Advanced machine learning models
* Interactive visual analytics
* Real feature importance analysis
* Cross-validation evaluation
* Business impact estimation
* Deployment-ready architecture

The goal is not only to predict diamond prices accurately, but also to analyze the factors that influence pricing decisions.

---

# Key Features

## Machine Learning

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor
* CatBoost Regressor
* Hyperparameter Tuning using GridSearchCV
* Cross-Validation Evaluation
* Ensemble Learning Comparison

## Advanced Data Analysis

* Statistical summaries
* Correlation heatmaps
* Outlier detection
* Price distribution analysis
* Feature importance visualization
* Business metrics analytics

## Interactive Dashboard

* Real-time prediction interface
* Interactive charts and visualizations
* Dynamic business insights
* Model performance comparison
* User-friendly Streamlit interface

## Business Intelligence

* Pricing consistency analysis
* Revenue impact estimation
* Feature impact evaluation
* Diamond valuation support
* Data-driven pricing recommendations

---

# Machine Learning Workflow

The project follows a professional machine learning pipeline:

1. Data Cleaning
2. Duplicate Removal
3. Outlier Handling
4. Feature Engineering
5. Log Transformation
6. Train/Test Split
7. Model Training
8. Hyperparameter Tuning
9. Cross-Validation
10. Model Evaluation
11. Feature Importance Analysis
12. Deployment using Streamlit

---

# Feature Engineering

The project includes advanced engineered features such as:

* Diamond volume calculation
* Price per carat analysis
* Log-transformed price features
* Log-transformed carat features

These features improve prediction performance and provide deeper analytical insights.

---

# Model Performance

The application compares multiple regression models using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Cross-Validation Stability

The best-performing model is automatically selected based on prediction accuracy.

---

# Explainability & Insights

The dashboard provides:

* Real feature importance analysis
* Correlation matrix visualization
* Outlier detection insights
* Business impact interpretation
* Top price-driving factors

This improves model transparency and helps users understand pricing behavior.

---

# Technologies Used

## Backend & Machine Learning

* Python
* Scikit-learn
* CatBoost
* XGBoost
* Pandas
* NumPy

## Web Application

* Streamlit

## Data Visualization

* Plotly
* Matplotlib

---

# Project Structure

```bash
diamond-price-prediction-dashboard/
│
├── app.py
├── DiamondsPrices.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset

Diamond Prices Dataset from Kaggle:

https://www.kaggle.com/datasets/shivam2503/diamonds

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Hamzah-20/diamond-price-prediction-dashboard.git
cd diamond-price-prediction-dashboard
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

Then open:

```bash
http://localhost:8501
```

---

# Application Features

## Dashboard Analytics

* Interactive business dashboard
* Real-time model evaluation
* Statistical summaries
* Visual analytics

## Prediction System

Users can enter:

* Carat
* Cut
* Color
* Clarity
* Depth
* Table
* Dimensions (X, Y, Z)

The system predicts the estimated diamond price instantly.

## Model Comparison

The platform compares multiple ML models and visualizes:

* Accuracy metrics
* Error metrics
* Cross-validation performance

---

# Business Impact

The dashboard demonstrates how machine learning can improve:

* Diamond valuation consistency
* Pricing decision support
* Business analytics workflows
* Data-driven pricing strategies

The displayed business metrics are hypothetical and intended for demonstration purposes.

---

# Future Improvements

* Cloud deployment
* Real-time API integration
* Advanced hyperparameter optimization
* Docker support
* MLOps integration
* Deep learning experimentation

---

# Author

Hamzah Albasyouni

AI & Machine Learning Enthusiast
Focused on Applied AI, Predictive Analytics, and Intelligent Systems

GitHub: [Hamzah Albasyouni](https://github.com/Hamzah-20)

---

# License

This project was developed for educational, research, and portfolio purposes.
