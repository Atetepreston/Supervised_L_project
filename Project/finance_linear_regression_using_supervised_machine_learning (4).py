# -*- coding: utf-8 -*-
"""Finance_Linear_Regression_Using_Supervised_Machine_Learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ECVx5hWKy4ZHFbIsHg3jSxNta8bYNfGF

**1. Data gathering and exploration:**

- Define the problematic to solve and the final objective
- Validate the project idea with instructor.
- Gather the relevant data.
- Explore your data and verify if it can help you solve the problematic you want to work on which is "Financial Regression Using Supervised Learning.
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace 'financial_regression.csv' with your file path)
df = pd.read_csv("/content/financial_regression.csv")

df.head()

df.info()

df.describe()

df.isnull().sum()

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Sort by date to ensure chronological order
df = df.sort_values(by='date')

# Forward-fill missing values for financial market data
market_columns = [
    'sp500 open', 'sp500 high', 'sp500 low', 'sp500 close', 'sp500 volume', 'sp500 high-low',
    'nasdaq open', 'nasdaq high', 'nasdaq low', 'nasdaq close', 'nasdaq volume', 'nasdaq high-low',
    'silver open', 'silver high', 'silver low', 'silver close', 'silver volume', 'silver high-low',
    'oil open', 'oil high', 'oil low', 'oil close', 'oil volume', 'oil high-low',
    'platinum open', 'platinum high', 'platinum low', 'platinum close', 'platinum volume', 'platinum high-low',
    'palladium open', 'palladium high', 'palladium low', 'palladium close', 'palladium volume', 'palladium high-low',
    'gold open', 'gold high', 'gold low', 'gold close', 'gold volume'
]

df[market_columns] = df[market_columns].fillna(method='ffill')

# Drop columns with too many missing values (GDP, us_rates_%, CPI)
df.drop(columns=['GDP', 'us_rates_%', 'CPI'])

# Drop remaining rows with missing values (mostly due to initial NaNs before forward-fill)
df_cleaned = df_cleaned.dropna()

# Verify cleaning results
df_cleaned_info = df_cleaned.info()
df_cleaned_missing = df_cleaned.isnull().sum().sum()  # Count total missing values

"""**2. Data visualisation and model selection:**
- Configure your data visualization tool.
- Visualize feature/feature relationships and draw your conclusion
- Visualize feature/target relationships and draw your conclusion.
- Select an appropriate model and tell us why.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
corr_matrix = df_cleaned.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Select key features for scatter plots against 'sp500 close'
key_features = ['sp500 open', 'nasdaq close', 'gold close', 'oil close', 'eur_usd']

# Plot feature-target relationships
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    sns.scatterplot(data=df_cleaned, x=feature, y='sp500 close', alpha=0.5, ax=axes[i])
    axes[i].set_title(f'{feature} vs. sp500 close')

plt.tight_layout()
plt.show()

"""**3. Model training and testing:**
- Prepare your data for training and testing.
- Train your model, Evaluate your model with the appropriate evaluation metrics.
- Check if there is an overfitting or underfitting issue and act accordingly.
- Tune the model parameters and repeat the previous 3 instructions and you get the desired results.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target variable
features = ['sp500 open', 'nasdaq close', 'gold close', 'oil close', 'eur_usd']
target = 'sp500 close'

X = df_cleaned[features]
y = df_cleaned[target]

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on training and test data
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Evaluate model performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Model Performance Result
print(f"Training MSE: {train_mse}")
print(f"Test MSE: {test_mse}")
print(f"Training R^2: {train_r2}")
print(f"Test R^2: {test_r2}")

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create a polynomial regression model (degree=2 for slight non-linearity)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Train the model
poly_model.fit(X_train, y_train)

# Predict on training and test data
y_train_poly_pred = poly_model.predict(X_train)
y_test_poly_pred = poly_model.predict(X_test)

# Evaluate model performance
train_mse_poly = mean_squared_error(y_train, y_train_poly_pred)
test_mse_poly = mean_squared_error(y_test, y_test_poly_pred)
train_r2_poly = r2_score(y_train, y_train_poly_pred)
test_r2_poly = r2_score(y_test, y_test_poly_pred)

# Model Performance Result
print(f"Training MSE (Polynomial): {train_mse_poly}")
print(f"Test MSE (Polynomial): {test_mse_poly}")
print(f"Training R^2 (Polynomial): {train_r2_poly}")

"""**4. Model deployment and monitoring:**
- Export the trained model.
- Select a deployment strategy then deploy the ML model.
- Build a Monitoring dashboard.
- Make a demo.
"""

