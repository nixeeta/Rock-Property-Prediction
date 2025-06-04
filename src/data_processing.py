# 1_data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Encode categorical columns
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
    
    # Define features and target
    X = df.drop("Performance Index", axis=1)
    y = df["Performance Index"]

    # Split into train and test sets
    return train_test_split(X, y, test_size=0.2, random_state=42)
