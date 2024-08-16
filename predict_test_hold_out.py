import pandas as pd
import preprocess as pp
import pickle
import numpy as np

# Read data
df_test = pd.read_csv('Data/test.csv')

# Remove order_id from test_df and save ids to add them to the submission file later
X_test = df_test.drop('id', axis=1)
test_ids = df_test['id']

# Preprocess data
X_test = pp.preprocess(X_test, is_train=False)

# Save preprocessed test data to csv
X_test.to_csv('Data/preprocessed_test.csv', index=False, encoding='utf-8-sig')

# Ensure all columns are of the correct data type
X_test = X_test.astype({col: 'float32' for col in X_test.columns})

# Load all the saved models using pickle
models = {
    'catboost': 'Pickled/catboost_regression_MAPE_0.0312.pkl',
    'gradient_boosting': 'Pickled/gradient_boosting_regression_MAPE_0.0650.pkl',
    'hpboost': 'Pickled/hpboost_regression_MAPE_0.0347.pkl',
    'lightgbm': 'Pickled/lightgbm_regression_MAPE_0.0337.pkl',
    'random_forest': 'Pickled/random_forest_regression_MAPE_0.0408.pkl',
    'xgboost': 'Pickled/xgboost_regression_MAPE_0.0648.pkl',
    'stacking_model': 'Pickled/stacking_model_MAPE_0.0294.pkl'
}

loaded_models = {}
for name, path in models.items():
    with open(path, 'rb') as f:
        loaded_models[name] = pickle.load(f)

# Generate predictions using each base model
base_model_predictions = []
for model_name in ['random_forest', 'xgboost', 'gradient_boosting', 'catboost', 'hpboost', 'lightgbm']:
    y_pred = loaded_models[model_name].predict(X_test)
    base_model_predictions.append(y_pred)

# Stack the base model predictions horizontally
stacked_predictions = np.column_stack(base_model_predictions)

# Use the stacking meta-model to generate final predictions
final_predictions = loaded_models['stacking_model'].predict(stacked_predictions)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'orders': final_predictions
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False, encoding='utf-8-sig')

print("Submission file saved as 'submission.csv'")
