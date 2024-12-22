"""
feature_engineering.py

This module handles all feature engineering logic, including zero-value replacement,
calculation of derived features, and final feature selection.
"""

import numpy as np
import pandas as pd

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zero with NaN in specific columns and create 
    'Total_Fitness_Endurance_Time' feature.
    """
    # Replace zero with np.nan in designated columns
    zero_replace_cols = [
        'Physical-Weight',
        'Physical-Height',
        'Physical-BMI',
        'Basic_Demos-Age',
        'Physical-Waist_Circumference',
        'Physical-Diastolic_BP',
        'Physical-HeartRate',
        'Physical-Systolic_BP'
    ]
    for col in zero_replace_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Create 'Total_Fitness_Endurance_Time'
    if 'Fitness_Endurance-Time_Mins' in df.columns and 'Fitness_Endurance-Time_Sec' in df.columns:
        df['Total_Fitness_Endurance_Time'] = (
            df['Fitness_Endurance-Time_Mins'].fillna(0) +
            df['Fitness_Endurance-Time_Sec'].fillna(0)/60.0
        )
        df.drop(['Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec'], axis=1, inplace=True)

    return df


def create_mapping(column: str, dataset: pd.DataFrame) -> dict:
    """
    Create a dictionary mapping of unique values of a column in `dataset`.
    """
    unique_values = dataset[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}


def encode_categorical(df: pd.DataFrame,
                       cat_cols: list[str],
                       train_mappings: dict = None) -> pd.DataFrame:
    """
    Encode categorical columns into integer values.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to transform
    cat_cols : list[str]
        Columns to encode
    train_mappings : dict
        Existing mappings from training dataset. If provided, 
        this function will use them for consistent encoding.

    Returns:
    --------
    df_encoded : pd.DataFrame
        The transformed DataFrame
    mappings : dict
        The dictionary of mappings used (if train_mappings not provided).
    """
    mappings = train_mappings or {}
    for c in cat_cols:
        df[c] = df[c].fillna('Missing').astype('category')
        if train_mappings is None:
            # create a new mapping if we don't have a provided one
            mappings[c] = create_mapping(c, df)
            df[c] = df[c].replace(mappings[c]).astype(int)
        else:
            # use existing mapping
            df[c] = df[c].replace(mappings[c]).astype(int)

    return df, mappings
