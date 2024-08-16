import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import preprocess as pp
import models as md
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Fixed seed for reproducibility
models_random_seed = 543
split_random_seed = 69

# Number of folds
k = 10

# Read data
df_train = pd.read_csv('Data/train.csv')

# Separate the train data into X and y
X = df_train.drop('orders', axis=1)
y = df_train['orders']

# Initialize KFold
kf = KFold(n_splits=k, shuffle=True, random_state=split_random_seed)

# Initialize lists to store models and their predictions
rf_models, xgb_models, gb_models, cb_models, hp_models, lgbm_models = [], [], [], [], [], []
rf_preds, xgb_preds, gb_preds, cb_preds, hp_preds, lgbm_preds = [], [], [], [], [], []
stacking_preds = []

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\n================= Fold {fold+1} =================")

    # Split data and make explicit copies
    X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_train, y_val = y.iloc[train_index].copy(), y.iloc[val_index].copy()

    # Preprocess data
    X_train = pp.preprocess(X_train, is_drop_columns=True)
    X_val = pp.preprocess(X_val, is_train=False, is_drop_columns=True)

    # Save to csv
    X_train.to_csv(f'Data/preprocessed_train_fold{fold+1}.csv', index=False, encoding='utf-8-sig')
    X_val.to_csv(f'Data/preprocessed_val_fold{fold+1}.csv', index=False, encoding='utf-8-sig')

    # Train individual boosting models

    print("---------------------------------------------\n#1 Random Forest Regression:\n---------------------------------------------")
    rf_model = md.random_forest_regression(X_train, y_train, X_val, y_val, models_random_seed, model_name= f'random_forest_regression_{fold + 1}', save_mape_in_name = False)
    rf_models.append(rf_model)
    rf_preds.append(rf_model.predict(X_val))

    print("---------------------------------------------\n#2 XGBoost Regression:\n---------------------------------------------")
    xgb_model = md.xgboost_regression(X_train, y_train, X_val, y_val, models_random_seed, model_name= f'xgboost_regression_{fold + 1}', save_mape_in_name = False)
    xgb_models.append(xgb_model)
    xgb_preds.append(xgb_model.predict(X_val))

    print("---------------------------------------------\n#3 Gradient Boosting Regression:\n---------------------------------------------")
    gb_model = md.gradient_boosting_regression(X_train, y_train, X_val, y_val, models_random_seed, model_name= f'gradient_boosting_regression_{fold + 1}', save_mape_in_name = False)
    gb_models.append(gb_model)
    gb_preds.append(gb_model.predict(X_val))

    print("---------------------------------------------\n#4 Cat Boost Regression:\n---------------------------------------------")
    cb_model = md.catboost_regression(X_train, y_train, X_val, y_val, models_random_seed, model_name= f'catboost_regression_{fold + 1}', save_mape_in_name = False)
    cb_models.append(cb_model)
    cb_preds.append(cb_model.predict(X_val))

    print("---------------------------------------------\n#5 HPBoost Regression:\n---------------------------------------------")
    hp_model = md.hpboost_regression(X_train, y_train, X_val, y_val, models_random_seed, model_name= f'hpboost_regression_{fold + 1}', save_mape_in_name = False)
    hp_models.append(hp_model)
    hp_preds.append(hp_model.predict(X_val))

    print("---------------------------------------------\n#6 lightGBM Regression:\n---------------------------------------------")
    lgbm_model = md.lightgbm_regression(X_train, y_train, X_val, y_val, models_random_seed, model_name= f'lightgbm_regression_{fold + 1}', save_mape_in_name = False)
    lgbm_models.append(lgbm_model)
    lgbm_preds.append(lgbm_model.predict(X_val))

    # Stacking model
    print("---------------------------------------------\n#7 Stacking model Regression:\n---------------------------------------------")
    meta_model = RandomForestRegressor(
        n_estimators=150,
        random_state=models_random_seed
    )
    
    # Use custom stacking model function and save the model with appropriate naming
    r2_val, me_val, mape_val, stacking_y_val_pred = md.train_stacking_model(
        X_train, y_train, 
        X_val, y_val, 
        base_models=[rf_model, xgb_model, gb_model, cb_model, hp_model, lgbm_model],
        meta_model=meta_model, 
        model_name=f'stacking_model_{fold + 1}',
        save_mape_in_name = False
    )
    stacking_preds.append(stacking_y_val_pred)

# Combine all fold predictions and actual values
all_y_val = np.concatenate([y.iloc[val_index] for _, val_index in kf.split(X)])
all_rf_preds = np.concatenate(rf_preds)
all_xgb_preds = np.concatenate(xgb_preds)
all_gb_preds = np.concatenate(gb_preds)
all_cb_preds = np.concatenate(cb_preds)
all_hp_preds = np.concatenate(hp_preds)
all_lgbm_preds = np.concatenate(lgbm_preds)
all_stacking_preds = np.concatenate(stacking_preds)

# Print metrics for each model
print("\n================= Averaged Results =================")
print(f"Random Forest Regression: ")
md.calculate_metrics_regression(all_y_val, all_rf_preds)
print(f"XGBoost Regression: ")
md.calculate_metrics_regression(all_y_val, all_xgb_preds)
print(f"Gradient Boosting Regression: ")
md.calculate_metrics_regression(all_y_val, all_gb_preds)
print(f"Cat Boost Regression: ")
md.calculate_metrics_regression(all_y_val, all_cb_preds)
print(f"HPBoost Regression: ")
md.calculate_metrics_regression(all_y_val, all_hp_preds)
print(f"LightGBM Regression: ")
md.calculate_metrics_regression(all_y_val, all_lgbm_preds)
print(f"Stacking Model Regression: ")
md.calculate_metrics_regression(all_y_val, all_stacking_preds)

