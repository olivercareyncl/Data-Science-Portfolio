# Titanic - Machine Learning from Disaster

![Kaggle](https://img.shields.io/badge/Kaggle-Titanic%20Survival%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a submission to the Kaggle competition **Titanic: Machine Learning from Disaster**. The goal of the competition is to predict the survival of passengers based on various features using machine learning algorithms and feature engineering.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Approach](#approach)
4. [How to Run the Project](#how-to-run-the-project)

## Project Overview

In this project, I applied classification models to predict passenger survival on the Titanic based on a dataset containing various features such as passenger class, age, gender, and fare. The project walks through the full data science pipeline:

- Data Exploration
- Feature Engineering
- Model Building
- Model Evaluation

## Data

The dataset used in this project is provided by Kaggle and contains information about passengers on the Titanic, including variables like `Pclass`, `Age`, `Fare`, and whether the passenger survived (`Survived`).

## Approach

### 1. **Exploratory Data Analysis (EDA)**
   - Explored the dataset to understand the distribution of variables, missing values, and outliers.
   - Visualised relationships between features and the target variable `Survived`.

### 2. **Feature Engineering**
   - Handled missing values by imputing the median for numerical features and the mode for categorical features.
   - Created new features:
     - `FamilySize`: Number of family members traveling with the passenger.
     - `IsAlone`: Indicator of whether the passenger was traveling alone.
     - `Title`: Extracted from the passenger's name.
   - One-hot encoded categorical variables and handled missing value indicators.

### 3. **Model Building**
   - Trained the following classification models:
     - **Logistic Regression**: As a baseline model.
     - **Random Forest Classifier**: An ensemble learning method.
     - **Gradient Boosting Classifier**: A boosting technique to improve prediction accuracy.

### 4. **Model Evaluation**
   - Evaluated models based on accuracy, precision, recall, and F1-score.
   - The Gradient Boosting model achieved an accuracy of **0.8343** after hyperparameter tuning.

## How to Run the Project

### Prerequisites

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/olivercareyncl/Data-Science-Portfolio/tree/main/Titanic%20-%20Machine%20Learning%20from%20Disaster

