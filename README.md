# CO2-Emissions-Prediction-In-Vehicle
Overview

This project aims to predict CO2 emissions from vehicles using machine learning models. It involves data preprocessing, exploratory data analysis (EDA), visualization, and the implementation of multiple regression models to find the best-performing model for accurate predictions.

Dataset

The dataset contains various features related to vehicle specifications, including:

Engine Size

Fuel Consumption (City, Highway, Combined)

Vehicle Weight

Fuel Type

CO2 Emissions (Target Variable)

Project Workflow

Data Loading: The dataset is loaded using Pandas.

Exploratory Data Analysis (EDA):

Viewing dataset structure

Summary statistics

Checking for missing values and duplicates

Data Visualization:

Histograms, boxplots, and pairplots to understand distributions and relationships

Heatmaps for correlation analysis

Data Preprocessing:

Handling missing values using mean/median imputation

One-hot encoding categorical variables

Feature scaling using StandardScaler

Model Selection and Training:

Linear Regression

Support Vector Regression (SVR)

Decision Tree Regressor

Splitting data into training and testing sets (80/20)

Model Evaluation:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R-squared Score (R2)

Model comparison to determine the best-performing algorithm

Model Deployment:

The best model is saved using Pickle (model.pkl).

Installation & Usage

Prerequisites

Python 3.x

Jupyter Notebook / Google Colab

Required Libraries: numpy, pandas, matplotlib, seaborn, sklearn, pickle

Running the Project

Clone the repository or download the files.

Install dependencies using:

pip install numpy pandas matplotlib seaborn scikit-learn

Run the Jupyter Notebook or Python script (inshara_college_project.py).

Load the dataset and execute the steps sequentially.

Evaluate the models and check predictions.

Results

Linear Regression, Support Vector Regression, and Decision Tree models were trained and evaluated.

The model with the highest R-squared value was selected as the best predictor of CO2 emissions.

