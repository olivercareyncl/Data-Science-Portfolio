# House Prices - Advanced Regression Techniques

![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a submission to the Kaggle competition **House Prices: Advanced Regression Techniques**. The goal of the competition is to predict the final sale price of homes based on a wide range of features using advanced regression techniques and feature engineering.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Approach](#approach)
4. [Feature Engineering](#feature-engineering)
5. [Models Used](#models-used)
6. [Evaluation](#evaluation)
7. [How to Run the Project](#how-to-run-the-project)
8. [License](#license)

## Project Overview

In this project, I applied regression models to predict housing prices based on a dataset containing various features such as lot size, the year the house was built, the condition of various components, and more. The project walks through the full data science pipeline:

- Data Exploration
- Feature Engineering
- Model Building
- Hyperparameter Tuning
- Model Evaluation

## Data

The dataset used in this project is provided by Kaggle and contains 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa. The target variable is the sale price of the house (`SalePrice`).

## Approach

### 1. **Exploratory Data Analysis (EDA)**
   - Explored the dataset to understand the distribution of variables, missing values, and outliers.
   - Visualized relationships between features and the target variable `SalePrice`.

### 2. **Feature Engineering**
   - Handled missing values by imputing the median for numerical features and the mode for categorical features.
   - Created new features:
     - `TotalSF`: Summing up various area-related features.
     - `Age`: Calculating the age of the house at the time of sale.
   - One-hot encoded categorical variables and handled missing value indicators.

### 3. **Model Building**
   - Trained the following regression models:
     - **Linear Regression**: As a baseline model.
     - **Random Forest Regressor**: An ensemble learning method.
     - **Gradient Boosting Regressor**: A boosting technique to improve prediction accuracy.
   - Tuned hyperparameters using `GridSearchCV`.

### 4. **Model Evaluation**
   - Used **Root Mean Squared Error (RMSE)** to evaluate model performance.
   - Applied logarithmic transformations to the target variable (`SalePrice`) for better handling of skewed distribution.
   - Final Kaggle score: **0.14507**.

## Models Used

- **Random Forest Regressor**: A powerful ensemble learning method.
- **Gradient Boosting Regressor**: Another ensemble learning model that builds trees sequentially to minimize the error.

## Evaluation

The evaluation metric used in the Kaggle competition is **Root Mean Squared Logarithmic Error (RMSLE)**. This accounts for both prediction error and logarithmic scaling to handle skewed target variables effectively.

## How to Run the Project

### Prerequisites

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/house-prices-regression.git
