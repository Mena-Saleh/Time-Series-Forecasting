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

# Initialize an array to hold predictions for all folds
final_predictions = np.zeros(X_test.shape[0])

# Loop over all 10 folds
for fold in range(1, 11):
    # Load base models for this fold
    base_model_predictions = []
    for model_name in ['random_forest', 'xgboost', 'gradient_boosting', 'catboost', 'hpboost', 'lightgbm']:
        model_path = f'Pickled/{model_name}_regression_{fold}.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # Get predictions for this model
        y_pred = model.predict(X_test)
        base_model_predictions.append(y_pred)
    
    # Stack the base model predictions horizontally
    stacked_predictions = np.column_stack(base_model_predictions)

    # Load the stacking model for this fold
    stacking_model_path = f'Pickled/stacking_model_{fold}.pkl'
    with open(stacking_model_path, 'rb') as f:
        stacking_model = pickle.load(f)
    
    # Generate predictions using the stacking model
    fold_predictions = stacking_model.predict(stacked_predictions)
    
    # Accumulate the fold predictions
    final_predictions += fold_predictions

# Average the predictions from all folds
final_predictions /= 10

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'orders': final_predictions
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False, encoding='utf-8-sig')

print("Submission file saved as 'submission.csv'")
