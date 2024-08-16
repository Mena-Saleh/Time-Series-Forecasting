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


# Measures R2 score, MSE, and MAPE for Regression models
def calculate_metrics_regression(y_true, y_pred):
    # Calculate R-squared (R2) score
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Print metrics
    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
    print(f"MAPE: {mape}")
    
    return r2, mse, mape

# Plot predictions vs actual results.
def visualize_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.grid(True)
    plt.show()

# ML Models definition
def generic_regression_model(X_train, y_train, X_val, y_val, model, model_name, preprocess_fn=None):
    # Apply preprocessing function if provided
    if preprocess_fn is not None:
        X_train = preprocess_fn.fit_transform(X_train)
        X_val = preprocess_fn.transform(X_val)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the training set
    y_train_pred = model.predict(X_train)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)
    # Calculate metrics for training set
    print(f"Metrics on Training Set:")
    r2_train, mse_train, mape_train = calculate_metrics_regression(y_train, y_train_pred)

    print("---------------------------------------------")
    
    # Calculate metrics for validation set
    print(f"Metrics on Validation Set:")
    r2_val, mse_val, mape_val = calculate_metrics_regression(y_val, y_val_pred)

    # Visualize predictions on val set
    visualize_predictions(y_val, y_val_pred)

    # Save the trained model using pickle, including MAPE in the file name
    mape_str = f"{mape_val:.4f}"  # Format MAPE to 4 decimal places
    with open(f"Pickled/{model_name}_MAPE_{mape_str}.pkl", "wb") as f:
        pickle.dump(model, f)

    return r2_val, mse_val, mape_val

def random_forest_regression(X_train, y_train, X_test, y_test, 
                             n_estimators=100, max_depth=None, 
                             min_samples_split=2, min_samples_leaf=1, 
                             max_samples=None):
    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            random_state=42  # for reproducibility
        ),
        model_name="random_forest_regression"
    )
    
def xgboost_regression(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth),
        model_name="xgboost_regression"
    )

def gradient_boosting_regression(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=3):
    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth),
        model_name="gradient_boosting_regression"
    )

#  iterations=3000, learning_rate=0.1, depth=6 ---> 0.0336
def catboost_regression(X_train, y_train, X_test, y_test, iterations=3000, learning_rate=0.1, depth=6):
    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=CatBoostRegressor(iterations=iterations, learning_rate=learning_rate, depth=depth, verbose=0),
        model_name="catboost_regression"
    )

def lightgbm_regression(X_train, y_train, X_test, y_test, n_estimators=300, learning_rate=0.1, num_leaves=31, max_depth=-1):
    # Ensure all columns are of the correct data type
    X_train = X_train.astype({col: 'float32' for col in X_train.columns})
    X_test = X_test.astype({col: 'float32' for col in X_test.columns})

    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, max_depth=max_depth, random_state=42),
        model_name="lightgbm_regression"
    )

def hpboost_regression(X_train, y_train, X_test, y_test, max_iter=3000, learning_rate=0.1, max_depth=6):
    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth, random_state=42),
        model_name="hpboost_regression"
    )

def adaboost_regression(X_train, y_train, X_test, y_test, n_estimators=1000, learning_rate=0.1):
    return generic_regression_model(
        X_train, y_train, X_test, y_test,
        model=AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42),
        model_name="adaboost_regression"
    )
    