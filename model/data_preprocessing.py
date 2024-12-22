"""
data_preprocessing.py

This module handles data loading, basic data inspection, and preprocessing steps 
such as merging train/test sets and performing KNN imputation.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
import warnings

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

SEED = 42  # Example usage of a global seed

def load_data(train_path: str, 
              test_path: str, 
              sample_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and sample submission CSVs.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Sample submission file not found: {sample_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    print("Train Dimension:", train.shape)
    print("Test Dimension:", test.shape)
    return train, test, sample


def knn_impute(train: pd.DataFrame, 
               test: pd.DataFrame, 
               numeric_only: bool = True,
               n_neighbors: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform KNN imputation on numeric columns of the combined train+test data.
    
    Parameters:
    -----------
    train : pd.DataFrame
    test : pd.DataFrame
    numeric_only : bool
        Whether to only impute numeric columns. 
    n_neighbors : int
        Number of neighbors for KNN imputation.

    Returns:
    --------
    train_imputed : pd.DataFrame
        Imputed train dataset
    test_imputed : pd.DataFrame
        Imputed test dataset
    """
    # Merge train and test
    all_data = pd.concat([train, test], axis=0)

    # Identify numeric columns
    if numeric_only:
        numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    else:
        # If you want to impute all columns, you can do something else
        numeric_cols = all_data.columns

    # Fit imputer on numeric columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(all_data[numeric_cols])

    # Create a DataFrame with imputed results
    all_data_imputed = pd.DataFrame(
        imputed_data,
        columns=numeric_cols,
        index=all_data.index
    )

    # For any columns not in numeric_cols, copy them back
    for col in all_data.columns:
        if col not in numeric_cols:
            all_data_imputed[col] = all_data[col]

    # Split back into train and test
    train_imputed = all_data_imputed.iloc[: len(train)].copy()
    test_imputed = all_data_imputed.iloc[len(train) :].copy()

    return train_imputed, test_imputed
