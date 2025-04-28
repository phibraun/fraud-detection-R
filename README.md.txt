# Fraud Detection Project (R)

## Overview
This project analyzes a financial transaction dataset to detect fraudulent activities. It uses resampling techniques to handle class imbalance and trains multiple machine learning models to maximize detection accuracy.

## Tools and Technologies
- R
- Packages: `caret`, `pROC`, `dplyr`, `ggplot2`, `MASS`, `corrplot`, `glmnet`, `randomForest`
- Techniques: Logistic Regression, Random Forest, downsampling

## Highlights
- Addressed extreme class imbalance (~0.13% fraud) using downsampling.
- Engineered features such as transaction types.
- Achieved 0.98 AUC-ROC and 88.6% recall using Random Forest model.
- Visualized model performance with ROC curves and confusion matrices.

## Files
- `fraud_detection_analysis.R`: Main analysis and modeling
- `plots/`: ROC curve and confusion matrix images
- `data/`: Sample dataset link

## How to Run
1. Install required packages:
    ```R
    install.packages(c("caret", "pROC", "dplyr", "ggplot2", "corrplot", "glmnet", "randomForest"))
    ```
2. Open `fraud_detection_analysis.R`
3. Run the script.

## Data Source
- Original dataset from [Kaggle: Fraudulent Transaction Dataset](https://www.kaggle.com/datasets/rohit265/fraud-detection-dynamics-financial-transaction)

---
