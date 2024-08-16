import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess as pp
import models as md
import feature_selection as fe

# Fixed seed for reproducability
random_seed = 42 

# Read data
df_train = pd.read_csv('Data/train.csv')

# Separate the train data into X and y
X = df_train.drop('orders', axis=1)
y = df_train['orders']

# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Preprocess data
X_train = pp.preprocess(X_train, is_drop_columns= True)
X_val = pp.preprocess(X_val, is_train= False, is_drop_columns= True)

# Save to csv
X_train.to_csv('Data/preprocessed_train.csv', index=False, encoding='utf-8-sig')
X_val.to_csv('Data/preprocessed_val.csv', index=False, encoding='utf-8-sig')


# Models training
# print("#1 Simple Linear Regression:")
# md.simple_linear_regression(X_train, y_train, X_val, y_val)

# print("#2 Polynomial Regression:")
# md.polynomial_regression(X_train, y_train, X_val, y_val)

print("---------------------------------------------\n#3 Random Forest Regression:\n---------------------------------------------")
md.random_forest_regression(X_train, y_train, X_val, y_val)

# print("#4 XGBoost Regression:")
# md.xgboost_regression(X_train, y_train, X_val, y_val)

# print("#5 Gradient Boosting Regression:")
# md.gradient_boosting_regression(X_train, y_train, X_val, y_val)

# print("#6 Stochastic Gradient Descent (SGD) Regression:")
# md.sgd_regression(X_train, y_train, X_val, y_val)

print("---------------------------------------------\n#7 Cat Boost Regression:\n---------------------------------------------")
md.catboost_regression(X_train, y_train, X_val, y_val)

# print("---------------------------------------------\n#8 AdaBoost Regression:\n---------------------------------------------")
# md.adaboost_regression(X_train, y_train, X_val, y_val)

print("---------------------------------------------\n#9 HPBoost Regression:\n---------------------------------------------")
md.hpboost_regression(X_train, y_train, X_val, y_val)

print("---------------------------------------------\n#10 lightGBM Regression:\n---------------------------------------------")
md.lightgbm_regression(X_train, y_train, X_val, y_val)


