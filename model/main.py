"""
main.py

The main script to orchestrate:
1. Data loading
2. Preprocessing and feature engineering
3. Model training and inference
4. Output of final submission
"""

import pandas as pd
import numpy as np
import warnings

# You can place global constants and configs here or in a dedicated config.py
SEED = 42
N_SPLITS = 5

from catboost import CatBoostRegressor

# Local modules
from data_preprocessing import load_data, knn_impute
from feature_engineering import feature_engineering, encode_categorical
from modeling import train_model

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None


def main():
    # 1. Load Data
    train, test, sample = load_data(
        train_path="./data/train.csv",
        test_path="./data/test.csv",
        sample_path="./data/sample_submission.csv"
    )

    # 2. Basic Feature Engineering
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Extract label and drop from train
    y = train['sii'].copy()
    train.drop(['sii'], axis=1, inplace=True)

    # Drop IDs (if any)
    if 'id' in train.columns:
        train.drop('id', axis=1, inplace=True)
    if 'id' in test.columns:
        test.drop('id', axis=1, inplace=True)

    # 3. KNN Impute
    train_imputed, test_imputed = knn_impute(train, test, numeric_only=True, n_neighbors=5)
    train = train_imputed.copy()
    test = test_imputed.copy()

    # 4. Select Features
    feature_cols = [
        'Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
        'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
        'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
        'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
        'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
        'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
        'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
        'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
        'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
        'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
        'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
        'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
        'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
        'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
        'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
        'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
        'PreInt_EduHx-computerinternet_hoursday', 'Total_Fitness_Endurance_Time'
    ]

    train = train[feature_cols]
    test = test[feature_cols]

    # Re-attach label
    train['sii'] = y

    # Drop rows with missing label if any
    train.dropna(subset=['sii'], inplace=True)

    # 5. Encode Categoricals
    cat_cols = [
        'Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 
        'Fitness_Endurance-Season', 'FGC-Season', 'BIA-Season', 
        'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season'
    ]
    train, mappings = encode_categorical(train, cat_cols, train_mappings=None)
    test, _ = encode_categorical(test, cat_cols, train_mappings=mappings)

    # 6. Prepare Final Datasets
    X = train.drop('sii', axis=1)
    y = train['sii']
    X_test = test.copy()

    # 7. Define Model
    catboost_params = {
        'learning_rate': 0.05,
        'depth': 6,
        'iterations': 200,
        'l2_leaf_reg': 10,
        'random_seed': SEED,
        'task_type': 'GPU',   # If you don't have GPU, switch this to 'CPU'
        'verbose': 0
    }
    model = CatBoostRegressor(**catboost_params)

    # 8. Train Model
    _, best_thresholds, final_test_preds = train_model(
        model_class=model,
        X=X,
        y=y,
        X_test=X_test,
        n_splits=N_SPLITS,
        random_state=SEED
    )

    # 9. Create Submission
    submission = pd.DataFrame({
        'id': list(range(len(final_test_preds))),  # or sample['id'] if you want original IDs
        'sii': final_test_preds
    })
    # If sample has 'id', use that instead:
    if 'id' in sample.columns:
        submission['id'] = sample['id']

    submission.to_csv('submission.csv', index=False)
    print("Submission saved to 'submission.csv'.")


if __name__ == "__main__":
    main()
