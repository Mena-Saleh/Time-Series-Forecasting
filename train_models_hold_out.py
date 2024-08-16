import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import preprocess as pp
import models as md
import feature_selection as fe
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

# Fixed seed for reproducability
models_random_seed = 543
split_random_seed = 69

# Read data
df_train = pd.read_csv('Data/train.csv')

# Separate the train data into X and y
X = df_train.drop('orders', axis=1)
y = df_train['orders']

# Split the data into training (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2, random_state=split_random_seed)

# Preprocess data
X_train = pp.preprocess(X_train, is_drop_columns= True)
X_val = pp.preprocess(X_val, is_train= False, is_drop_columns= True)

# Save to csv
X_train.to_csv('Data/preprocessed_train.csv', index=False, encoding='utf-8-sig')
X_val.to_csv('Data/preprocessed_val.csv', index=False, encoding='utf-8-sig')


# Models training

#1 Training individual boosting models
print("---------------------------------------------\n#1 Random Forest Regression:\n---------------------------------------------")
rf = md.random_forest_regression(X_train, y_train, X_val, y_val, models_random_seed)

print("---------------------------------------------\n#2 XGBoost Regression:\n---------------------------------------------")
xgb = md.xgboost_regression(X_train, y_train, X_val, y_val, models_random_seed)

print("---------------------------------------------\n#3 Gradient Boosting Regression:\n---------------------------------------------")
gb = md.gradient_boosting_regression(X_train, y_train, X_val, y_val, models_random_seed)

print("---------------------------------------------\n#4 Cat Boost Regression:\n---------------------------------------------")
cb = md.catboost_regression(X_train, y_train, X_val, y_val, models_random_seed)

# print("---------------------------------------------\n#8 AdaBoost Regression:\n---------------------------------------------")
# ab = md.adaboost_regression(X_train, y_train, X_val, y_val, random_seed)

print("---------------------------------------------\n#5 HPBoost Regression:\n---------------------------------------------")
hp =md.hpboost_regression(X_train, y_train, X_val, y_val, models_random_seed)

print("---------------------------------------------\n#6 lightGBM Regression:\n---------------------------------------------")
lgbm = md.lightgbm_regression(X_train, y_train, X_val, y_val, models_random_seed)

#2 Stacking model
print("---------------------------------------------\n#7 Stacking model Regression:\n---------------------------------------------")
#meta_model = CatBoostRegressor(iterations=300, learning_rate=0.1, depth=6, verbose=0, random_state=models_random_seed)
meta_model = RandomForestRegressor(
            n_estimators=150,
            random_state=models_random_seed 
        )
md.train_stacking_model(X_train, y_train, X_val, y_val, base_models= [rf, xgb, gb, cb, hp, lgbm], meta_model = meta_model)
