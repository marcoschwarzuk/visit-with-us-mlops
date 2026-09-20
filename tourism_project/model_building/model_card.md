---
library_name: sklearn
tags:
- tabular-classification
- mlops
---
# Visit with Us - Wellness Tourism Package purchase model

Best model after hyperparameter tuning: **xgboost**

Best parameters: `{'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 5, 'xgbclassifier__n_estimators': 200}`

Serving threshold: **0.45** (selected from out-of-fold training F1).

## Test set metrics at the selected threshold
- accuracy: 0.9091
- precision: 0.7470
- recall: 0.8000
- f1: 0.7726
- roc_auc: 0.9276

Trained on `Marco8000/visit-with-us-dataset` (train.csv / test.csv). Load with `joblib.load('best_tourism_model_v1.joblib')`.
