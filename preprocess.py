import pandas as pd
import dateutil.parser
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
import ast
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from math import pi

# Main preprocessing function
def preprocess(df: pd.DataFrame, is_train: bool = True, is_drop_columns: bool = False):
    df = preprocess_dates(df)
    df = fill_na(df)
    df = encode_columns(df, is_train)
    #df = scale_data(df, is_train)
    if is_drop_columns:
        df = drop_columns(df)
    return df


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
    df['holiday_name'].fillna('None')
    return df

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
def encode_columns(df: pd.DataFrame, is_train: bool = True, encoders=None):
    columns = ['warehouse', 'holiday_name']
    df[columns] = df[columns].astype(str)

    if is_train:
        encoders = {}
        for col in columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder
        # Save encoders using pickle
        with open('Pickled/encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
    else:
        with open('Pickled/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        for col in columns:
            df[col] = encoders[col].transform(df[col])
            
    
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


