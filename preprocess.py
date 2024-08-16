import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
import pickle
from math import pi

# Main preprocessing function
def preprocess(df: pd.DataFrame, is_train: bool = True, is_drop_columns: bool = False):
    df = preprocess_dates(df)
    #df = fill_na(df)
    #df = scale_data(df, is_train, scaler= MinMaxScaler())
    #df = feature_engineer(df)
    df = encode_columns(df, is_train)
    if is_drop_columns:
        df = drop_columns(df)
    #df = extract_poly_features(df, 2, is_train= is_train)
    return df.astype({col: 'float32' for col in df.columns})

# Extracts meaningful information from dates
def preprocess_dates(df: pd.DataFrame):
    # Ensure the date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year
    df['year'] = df['date'].dt.year
    
    # Extract month
    df['month'] = df['date'].dt.month
    
    # Extract day
    df['day'] = df['date'].dt.day
    
    # Extract timestamp
    df['timestamp_days'] = (df['date'] - pd.Timestamp("1970-01-01")) / pd.Timedelta('1D')
        
    # Extract quarter
    df['quarter'] = df['date'].dt.quarter
    
    # Extract week number within the year
    df['week_number'] = df['date'].dt.isocalendar().week
    
    # Extract day of the week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Extract week number within the month
    df['week_of_month'] = df['date'].apply(lambda x: (x.day - 1) // 7 + 1)
    
    # Boolean features
    df["is_month_start"] = df['date'].dt.is_month_start.astype(int).fillna(-1)
    df["is_month_end"] = df['date'].dt.is_month_end.astype(int).fillna(-1)
    df["is_quarter_start"] = df['date'].dt.is_quarter_start.astype(int).fillna(-1)
    df["is_quarter_end"] = df['date'].dt.is_quarter_end.astype(int).fillna(-1)
    df['is_weekend'] = np.where(df['day_of_week'].isin([5,6]), 1,0)
    
    # Sine and Cosine transformations
    # df['month_sin'] = np.sin(2 * pi * df['month'] / 12)
    # df['month_cos'] = np.cos(2 * pi * df['month'] / 12)
    # df['day_sin'] = np.sin(2 * pi * df['day'] / 31)
    # df['day_cos'] = np.cos(2 * pi * df['day'] / 31)
    
    
    # Finally drop the date column
    df.drop('date', axis=1, inplace=True)

    return df

# Fill null values
def fill_na(df: pd.DataFrame):
    df['holiday_name'].fillna('Unknown')
    return df

# Extract some features like holiday_before etc...
def feature_engineer(df: pd.DataFrame):
    # Obtain the data for the day before or after a holiday
    df['holiday_before'] = df['holiday_name'].shift(1).fillna('Unknown')
    df['holiday_after'] = df['holiday_name'].shift(-1).fillna('Unknown')
    return df

# Extracts polynomial featurse up to a given degree
def extract_poly_features(df: pd.DataFrame, deg: int, is_train: bool = True):
    if is_train:
        # Initialize the PolynomialFeatures object with the desired degree
        poly = PolynomialFeatures(degree=deg, include_bias=False)

        # Fit and transform the data to create polynomial features
        poly_features = poly.fit_transform(df)

        # Save the fitted PolynomialFeatures object using pickle
        with open('Pickled/poly_features.pkl', 'wb') as file:
            pickle.dump(poly, file)

    else:
        # Load the saved PolynomialFeatures object
        with open('Pickled/poly_features.pkl', 'rb') as file:
            poly = pickle.load(file)

        # Only transform the test data using the previously fitted object
        poly_features = poly.transform(df)

    # Create a DataFrame with the generated polynomial feature names
    poly_feature_names = poly.get_feature_names_out(df.columns)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    return poly_df

# Scale numerical columns
def scale_data(df: pd.DataFrame, X_test: pd.DataFrame, is_train: bool = True, scaler=None):
    # Use StandardScaler as the default scaler if none is provided
    if scaler is None:
        scaler = StandardScaler()
    
    columns = ['timestamp_days']
    
    if is_train:
        df[columns] = scaler.fit_transform(df[columns])
        # Save scaler using pickle
        with open('Pickled/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open('Pickled/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            df[columns] = scaler.transform(df[columns])
 
    return df

# Encodes categorical columns to get a numerical format.
def encode_columns(df: pd.DataFrame, is_train: bool = True, encoders=None, default_label='Unknown'):
    columns = ['warehouse', 'holiday_name']
    # columns = ['warehouse', 'holiday_name', 'holiday_before', 'holiday_after']    
    
    df[columns] = df[columns].astype(str)
    if is_train:
        encoders = {}
        for col in columns:
            encoder = LabelEncoder()

            # Add the default_label to ensure it's in the encoder
            df[col] = df[col].fillna(default_label)
            unique_labels = df[col].unique().tolist()
            if default_label not in unique_labels:
                unique_labels.append(default_label)

            # Fit the encoder on the training data including the default label
            encoder.fit(unique_labels)
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else default_label)
            df[col] = encoder.transform(df[col])
            encoders[col] = encoder

        # Save the encoders using pickle
        with open('Pickled/label_encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)

    else:
        with open('Pickled/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        for col in columns:
            encoder = encoders[col]

            # Handle unseen labels by replacing them with the default_label
            df[col] = df[col].fillna(default_label)
            df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else default_label)
            df[col] = encoder.transform(df[col])

    return df

# Drops unwatned columns
def drop_columns(df: pd.DataFrame):
    # Drop columns that are in train but not the test set
    df.drop('shutdown', axis=1, inplace=True)
    df.drop('mini_shutdown', axis=1, inplace=True)
    df.drop('blackout', axis=1, inplace=True)
    df.drop('frankfurt_shutdown', axis=1, inplace=True)
    df.drop('precipitation', axis=1, inplace=True)
    df.drop('mov_change', axis=1, inplace=True)
    df.drop('snow', axis=1, inplace=True)
    df.drop('user_activity_1', axis=1, inplace=True)
    df.drop('user_activity_2', axis=1, inplace=True)
    df.drop('id', axis = 1, inplace= True)
    return df


