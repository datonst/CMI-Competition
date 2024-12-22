# S&D: Semi-Supervised Learning for Medical Health Internet Problems

The repo is S&D team solution for kaggle competition: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use

## 1. Overview
The Child Mind Institute-Problematic Internet Use competition on Kaggle challenges participants to develop a predictive model capable of analyzing children's physical activity data to detect early indicators of problematic internet and technology use.
Target Variable (sii) is defined as:
- 0: None (PCIAT-PCIAT_Total from 0 to 30)
- 1: Mild (PCIAT-PCIAT_Total from 31 to 49)
- 2: Moderate (PCIAT-PCIAT_Total from 50 to 79)
- 3: Severe (PCIAT-PCIAT_Total 80 and more)
This makes sii an ordinal categorical variable with four levels, where the order of categories is meaningful.


## 2. Implementation
The datasets are expected to be stored in the folder ../data or specified by the root parameter, and arranged as follows:
./data
├── data_dictionary.csv
├── sample_submission.csv
├── test.csv
└── train.csv
- The time series datasets is too large, so you need go to the kaggle competition for downloading.
But in my solutions, we cannot use time series because they can lead to decrease CV score.

### File Descriptions

- **`data_preprocessing.py`**  
  Handles loading of data and KNN imputation steps.

- **`feature_engineering.py`**  
  Contains feature engineering functions such as zero -> NaN replacement and new feature creation.

- **`modeling.py`**  
  Encapsulates all modeling steps, including the StratifiedKFold cross-validation, threshold optimization, and final test predictions.

- **`main.py`**  
  The main entry script that orchestrates the end-to-end pipeline (data loading, preprocessing, feature engineering, modeling, and submission file creation).

## 3 Data Analyst
<p align="center">
<img src="images/output.png " width="850"> 
</p>
- In this scenario, the training dataset includes missing labels (sii). With our approach, we incorporate these unlabeled data points into the imputation process to better capture relationships between features, enhancing the imputation's accuracy and consistency.

## 4. Workflow Diagram
Approaches to Modeling SII as the Target Variable
1. Multiclass classification (treat sii as a nominal categorical variable without considering the order)
2. Regression (ignore the discrete nature of categories and treat sii as a continuous variable, then round prediction). We can use `PCIAT-PCIAT_Total` as a continuous target variable, and implement regression on `PCIAT-PCIAT_Total` and then map predictions to sii categories.
**In this repo, the primary focus will be on classification models to directly predict SII categories.**

<p align="center">
<img src="images/pineline.png " width="850"> 
</p>

## 5. Detailed Methodology - Semi-Supervised Learning
### 5.1 Preprocessing
To enhance model performance, we applied the following preprocessing steps:
    - Noise Handling: Replaced invalid values (e.g., zeros where not possible) with NaN, yielding slight improvements in offline cross-validation (CV) scores.
    - Feature Simplification: Combined multiple features into a single representation, reducing NaN occurrences and model complexity to mitigate overfitting risks.
    - Categorical Encoding: Converted seasonal data into numerical formats, improving learning efficiency.
### 5.2 KNN Imputation & Feature Engineering
- KNN imputation was implemented due to its superior performance with medical datasets [1][2]. It was applied as follows:
    1. Imputation for the Training Dataset:
        - The training dataset (including both labeled and unlabeled data) is imputed to fill missing (NaN) values.
        - After imputation, feature selection is performed to remove features that are not present in the test dataset, ensuring consistency between the datasets.
    2. Imputation for the Test Dataset:
        - To improve the accuracy of predictions on the test dataset, we impute its missing values using the information derived from the training dataset.
        - This ensures the test dataset has complete feature values before prediction.
   **Note**: For the best leaderboard score (private score: 0.475), we merged the imputation processes for both the training and test datasets into a unified approach. However, this approach is only suitable for problems focused on labeling datasets and not for tasks involving model creation and evaluation.
### 5.3 Training Model
We utilized CatBoost, a robust algorithm for datasets with missing values [3]. Key training methodologies included:
    - Cross-Validation: Applied stratified K-Folds (5 splits) to ensure balanced class distributions across training and validation sets.
    - Evaluation Metric: Quadratic Weighted Kappa (QWK) was employed to measure agreement between predictions and actual labels, considering the ordinal nature of the target variable.
    - Threshold Optimization: Fine-tuned decision thresholds using scipy.optimize's minimize function to map continuous outputs to discrete labels (None, Mild, Moderate, Severe).
### 5.4 Predict
The final model predicted test set labels using an ensemble of predictions from 5 training folds, enhancing robustness.

## 6. How to Run
- Dataset download:
   ```
   Download our datasets and place them into *CMI-COMPETITION/data/xxxxx*.
   ```
### How to Use

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt 
2. **Run the pipeline**:
   ```bash
   pip install -r requirements.txt
   python main.py
3. Check the output:
    A file named submission.csv is created.

   
## 7. Reference
[1] Emmanuel, T., Maupong, T., Mpoeleng, D., Semong, T., Mphago, B., & Tabona, O. (2021). A survey on missing data in machine learning. Journal of big data, 8(1), 140. https://doi.org/10.1186/s40537-021-00516-9</br>
[2] SHeru Nugroho, Nugraha Priya Utama, and Kridanto Surendro. 2023. KNN Imputation Versus Mean Imputation for Handling Missing Data on Vulnerability Index in Dealing with Covid-19 in Indonesia. In Proceedings of the 2023 12th International Conference on Software and Computer Applications (ICSCA '23). Association for Computing Machinery, New York, NY, USA, 20–25. https://doi.org/10.1145/3587828.3587832</br>
[3] Abdullahi A. Ibrahim, Raheem L. Ridwan, Muhammed M. Muhammed, Rabiat O. Abdulaziz and Ganiyu A. Saheed, “Comparison of the CatBoost Classifier with other Machine Learning Methods” International Journal of Advanced Computer Science and Applications(IJACSA), 11(11), 2020. http://dx.doi.org/10.14569/IJACSA.2020.0111190</br>

