# House Prices - Advanced Regression Techniques

![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a submission to the Kaggle competition **House Prices: Advanced Regression Techniques**. The goal of the competition is to predict the final sale price of homes based on a wide range of features using advanced regression techniques and feature engineering.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data](#data)
3. [Approach](#approach)
4. [How to Run the Project](#how-to-run-the-project)


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
   - Visualised relationships between features and the target variable `SalePrice`.

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
   git clone https://github.com/olivercareyncl/Data-Science-Portfolio.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Data-Science-Portfolio/House Prices - Advanced Regression Analysis
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebooks in this order:
   - `01_Data_Exploration.ipynb`: Perform exploratory data analysis (EDA) to understand the data and identify patterns.
   - `02_Feature_Engineering.ipynb`: Perform feature engineering, handle missing values, and transform features.
   - `03_Model_Building.ipynb`: Build the machine learning models (Linear Regression, Random Forest, Gradient Boosting) and evaluate them.
   - `04_Model_Deployment.ipynb`: Use the best-performing model to generate predictions on the test set.

5. To generate predictions:
   - Ensure that `test.csv` is placed in the working directory.
   - Open and run `04_Model_Deployment.ipynb` to generate the predictions and save the submission file in the required format.

6. The generated submission file will be saved as:
   - `rf_predictions_submission.csv` (Random Forest predictions)
   - `gb_predictions_submission.csv` (Gradient Boosting predictions)
   - `best_gb_predictions_submission.csv` (Best-tuned Gradient Boosting predictions)



