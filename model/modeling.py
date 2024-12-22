"""
modeling.py

This module encapsulates model training and evaluation functions, 
including metric calculation (QWK), threshold tuning, and the training pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from tqdm import tqdm
from colorama import Fore, Style
from IPython.display import clear_output

# Quadratic Weighted Kappa
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_rounder(preds: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Apply custom thresholds for rounding predictions. 
    E.g., threshold = [0.5, 1.5, 2.5] => classes: 0, 1, 2, 3
    """
    return np.where(
        preds < thresholds[0], 0,
        np.where(
            preds < thresholds[1], 1,
            np.where(preds < thresholds[2], 2, 3)
        )
    )

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    """
    Objective function for threshold search (Nelder-Mead).
    """
    rounded_p = threshold_rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

def train_model(model_class, 
                X: pd.DataFrame, 
                y: pd.Series, 
                X_test: pd.DataFrame, 
                n_splits: int = 5, 
                random_state: int = 42):
    """
    Train the specified model class using StratifiedKFold, optimize thresholds,
    and return predictions for the test set.

    Parameters:
    -----------
    model_class : sklearn-compatible model or CatBoost model
    X : pd.DataFrame
    y : pd.Series
    X_test : pd.DataFrame
    n_splits : int
    random_state : int

    Returns:
    --------
    oof_preds : np.ndarray
        Out-of-fold predictions (non-rounded).
    best_thresholds : np.ndarray
        The optimized threshold boundaries.
    test_preds : np.ndarray
        The final (thresholded) test predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_non_rounded = np.zeros(len(y), dtype=float) 
    test_preds_fold = np.zeros((len(X_test), n_splits))
    
    train_scores = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        tqdm(skf.split(X, y), desc="Training Folds", total=n_splits)
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        # Predict on validation
        y_val_pred = model.predict(X_val)
        # Keep OOF predictions
        oof_non_rounded[val_idx] = y_val_pred

        # Round predictions for scoring
        y_val_pred_rounded = y_val_pred.round().astype(int)

        # Evaluate
        train_kappa = quadratic_weighted_kappa(y_train, model.predict(X_train).round().astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_scores.append(train_kappa)
        val_scores.append(val_kappa)

        # Predict on test set
        test_preds_fold[:, fold] = model.predict(X_test)

        # Display fold results
        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")
        clear_output(wait=True)

    # Overall average performance
    print(f"Mean Train QWK: {np.mean(train_scores):.4f}")
    print(f"Mean Validation QWK: {np.mean(val_scores):.4f}")

    # Threshold optimization using 'Nelder-Mead'
    optimization_result = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5], 
        args=(y, oof_non_rounded),
        method='Nelder-Mead'
    )
    if not optimization_result.success:
        print("Warning: Optimization did not converge perfectly.")
    best_thresholds = optimization_result.x

    # Compute tuned QWK
    oof_tuned = threshold_rounder(oof_non_rounded, best_thresholds)
    tuned_qwk = quadratic_weighted_kappa(y, oof_tuned)
    print(f"Optimized QWK: {Fore.CYAN}{Style.BRIGHT}{tuned_qwk:.3f}{Style.RESET_ALL}")

    # Test predictions
    final_test_preds = test_preds_fold.mean(axis=1)
    final_test_preds = threshold_rounder(final_test_preds, best_thresholds)

    return oof_non_rounded, best_thresholds, final_test_preds
