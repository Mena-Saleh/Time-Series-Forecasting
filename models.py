import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
import lightgbm as lgb
import math
import numpy as np

# Measures R2 score, ME, and MAPE for Regression models
def calculate_metrics_regression(y_true, y_pred):
    # Calculate R-squared (R2) score
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Print metrics
    print(f"R2 Score: {r2}")
    print(f"ME: {math.sqrt(mse)}")
    print(f"MAPE: {mape}")
    
    return r2, mse, mape

# Plot predictions vs actual results.
def visualize_predictions(y_true, y_pred):
    return
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.grid(True)
    plt.show()

# Models definition

# Generic training function
def train_and_evaluate_regression_model(X_train, y_train, X_test, y_test, model, model_name, preprocess_fn=None, save_mape_in_name = True):
    # Apply preprocessing function if provided
    if preprocess_fn is not None:
        X_train = preprocess_fn.fit_transform(X_train)
        X_test = preprocess_fn.transform(X_test)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the training set
    y_train_pred = model.predict(X_train)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_test)
    # Calculate metrics for training set
    print(f"Metrics on Training Set:")
    r2_train, me_train, mape_train = calculate_metrics_regression(y_train, y_train_pred)

    print("---------------------------------------------")
    
    # Calculate metrics for validation set
    print(f"Metrics on Validation Set:")
    r2_val, me_val, mape_val = calculate_metrics_regression(y_test, y_val_pred)

    # Visualize predictions on val set
    visualize_predictions(y_test, y_val_pred)

    # Save the trained model using pickle
    mape_str = f"{mape_val:.4f}"  # Format MAPE to 4 decimal places
    
    # Construct the file name with or without MAPE in the name
    file_name = f"Pickled/{model_name}"
    if save_mape_in_name:
        file_name += f"_MAPE_{mape_str}"
    file_name += ".pkl"
    
    # Save the model using pickle
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

    return r2_val, me_val, mape_val


# Models and their hyper parameters
def random_forest_regression(X_train, y_train, X_test, y_test, random_seed, model_name = 'random_forest_regression', save_mape_in_name = True,
                             n_estimators=100, max_depth=None, 
                             min_samples_split=2, min_samples_leaf=1, 
                             max_samples=None):
    model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            random_state=random_seed 
        )
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model= model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model
    
def xgboost_regression(X_train, y_train, X_test, y_test,  random_seed,model_name = 'xgboost_regression' , save_mape_in_name = True, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_seed)
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model=model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model

def gradient_boosting_regression(X_train, y_train, X_test, y_test,  random_seed, model_name = 'gradient_boosting_regression', save_mape_in_name = True, n_estimators=100, learning_rate=0.1, max_depth=3):
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_seed)
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model=model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model

#  iterations=3000, learning_rate=0.1, depth=6 ---> 0.0336
def catboost_regression(X_train, y_train, X_test, y_test, random_seed, model_name = 'catboost_regression', save_mape_in_name = True, iterations=3000, learning_rate=0.1, depth=6):
    model = CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0, random_state=random_seed)
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model=model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model

def lightgbm_regression(X_train, y_train, X_test, y_test, random_seed, model_name = 'lightgbm_regression', save_mape_in_name = True, n_estimators=300, learning_rate=0.1, num_leaves=31, max_depth=-1):
    model=lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth, random_state=random_seed)
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model=model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model

def hpboost_regression(X_train, y_train, X_test, y_test,  random_seed, model_name= 'hpboost_regression', save_mape_in_name = True, max_iter=3000, learning_rate=0.1, max_depth=6):
    model=HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth, random_state=random_seed)
    
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model=model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model

def adaboost_regression(X_train, y_train, X_test, y_test, random_seed,  model_name= 'adaboost_regression', save_mape_in_name = True, n_estimators=1000, learning_rate=0.1):
    model=AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_seed)
    train_and_evaluate_regression_model(
        X_train, y_train, X_test, y_test,
        model=model,
        model_name= model_name,
        save_mape_in_name= save_mape_in_name
    )
    return model
    
    
    
# Function to train and evaluate a stacking model
def train_stacking_model(X_train, y_train, X_test, y_test, base_models, meta_model, model_name="stacking_model", save_mape_in_name = True,):
    # Get predictions from base models
    train_predictions = np.column_stack([model.predict(X_train) for model in base_models])
    val_predictions = np.column_stack([model.predict(X_test) for model in base_models])
    
    # Train the meta-model on the predictions of base models
    meta_model.fit(train_predictions, y_train)
    
    # Make predictions on the training set
    y_train_pred = meta_model.predict(train_predictions)

    # Make predictions on the validation set
    y_val_pred = meta_model.predict(val_predictions)
    
    # Calculate metrics for training set
    print(f"Metrics on Training Set:")
    r2_train, me_train, mape_train = calculate_metrics_regression(y_train, y_train_pred)

    print("---------------------------------------------")
    
    # Calculate metrics for validation set
    print(f"Metrics on Validation Set:")
    r2_val, me_val, mape_val = calculate_metrics_regression(y_test, y_val_pred)

    # Visualize predictions on validation set
    visualize_predictions(y_test, y_val_pred)
    
    
    
    # Save the trained model using pickle
    mape_str = f"{mape_val:.4f}"  # Format MAPE to 4 decimal places
    
    # Construct the file name with or without MAPE in the name
    file_name = f"Pickled/{model_name}"
    if save_mape_in_name:
        file_name += f"_MAPE_{mape_str}"
    file_name += ".pkl"
    
    # Save the model using pickle
    with open(file_name, "wb") as f:
        pickle.dump(meta_model, f)
    
    return r2_val, me_val, mape_val, y_val_pred

