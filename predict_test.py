import pandas as pd
import preprocess as pp
import pickle

# Read data
df_test = pd.read_csv('Data/test.csv')

# Remove order_id from test_df and save ids to add them to the submission file later
X_test = df_test.drop('id', axis = 1)
test_ids =  df_test['id']

# Preprocess data
X_test = pp.preprocess(X_test, is_train= False)

# Save to csv
X_test.to_csv('Data/preprocessed_test.csv', index=False, encoding='utf-8-sig')

# Load the saved model using pickle
with open('Pickled/catboost_regression_MAPE_0.0336 [habibi].pkl', 'rb') as f:
    model = pickle.load(f)
    
# Make predictions on the test set
y_pred = model.predict(X_test)

# Round the predicted orders to the nearest integer
y_pred_rounded = y_pred.round().astype(int)

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'orders': y_pred_rounded
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False, encoding='utf-8-sig')

print("Submission file saved as 'submission.csv'")

