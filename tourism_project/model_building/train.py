# for data manipulation
import os
import json
import warnings

import numpy as np
import pandas as pd
# for preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score
# for model serialization and experiment tracking
import joblib
import mlflow
# for registering the model on the Hugging Face Hub
from huggingface_hub import HfApi, create_repo, hf_hub_download

warnings.filterwarnings("ignore")

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO_ID = "Marco8000/visit-with-us-dataset"
MODEL_REPO_ID = "Marco8000/visit-with-us-model"
MODEL_DIR = "tourism_project/model_building"
MODEL_FILE = "best_tourism_model_v1.joblib"
TARGET = "ProdTaken"

# MLflow tracking server (started in the background by the notebook / the GitHub Actions job)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("visit-with-us-experiment")

# ---------------------------------------------------------------- 1. Load train / test data from the Hugging Face data space
train_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="train.csv", repo_type="dataset", token=HF_TOKEN)
test_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="test.csv", repo_type="dataset", token=HF_TOKEN)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

Xtrain, ytrain = train_df.drop(columns=[TARGET]), train_df[TARGET]
Xtest, ytest = test_df.drop(columns=[TARGET]), test_df[TARGET]

# ---------------------------------------------------------------- 2. Preprocessing
numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

# Ratio used by XGBoost to give the minority (buyers) class more weight
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight for the positive class: {class_weight:.2f}")

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# ---------------------------------------------------------------- 3. Candidate models and their hyperparameter grids
# Grids are kept small on purpose so the whole search runs in a few minutes on GitHub Actions.
candidate_models = {
    "decision_tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"decisiontreeclassifier__max_depth": [4, 6, 8],
         "decisiontreeclassifier__min_samples_leaf": [5, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"randomforestclassifier__n_estimators": [100, 200],
         "randomforestclassifier__max_depth": [8, None]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=42),
        {"gradientboostingclassifier__n_estimators": [100, 200],
         "gradientboostingclassifier__max_depth": [2, 3]},
    ),
    "adaboost": (
        AdaBoostClassifier(algorithm="SAMME", random_state=42),
        {"adaboostclassifier__n_estimators": [100, 200],
         "adaboostclassifier__learning_rate": [0.1, 0.5]},
    ),
    "xgboost": (
        xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss"),
        {"xgbclassifier__n_estimators": [100, 200],
         "xgbclassifier__max_depth": [3, 5],
         "xgbclassifier__learning_rate": [0.05, 0.1]},
    ),
}


def metrics_at_threshold(y, probabilities, threshold):
    """Evaluate buyer-class metrics for a probability decision threshold."""
    y_pred = (probabilities >= threshold).astype(int)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y, probabilities),
    }


def evaluate(model, X, y, threshold=0.50):
    """Evaluate a fitted pipeline without changing the model's preprocessing path."""
    probabilities = model.predict_proba(X)[:, 1]
    return metrics_at_threshold(y, probabilities, threshold)


# ---------------------------------------------------------------- 4./5. Tune every model and log the experiments
results = []
best_name, best_model, best_cv_f1 = None, None, -1.0

for name, (estimator, param_grid) in candidate_models.items():
    model_pipeline = make_pipeline(preprocessor, estimator)

    with mlflow.start_run(run_name=name):
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        # Log each parameter combination that was tried as a nested run
        cv_results = grid_search.cv_results_
        for i, params in enumerate(cv_results["params"]):
            with mlflow.start_run(run_name=f"{name}_trial_{i + 1}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_test_f1", cv_results["mean_test_score"][i])
                mlflow.log_metric("std_test_f1", cv_results["std_test_score"][i])

        # Log the best parameters and default-threshold performance in the parent run
        tuned_model = grid_search.best_estimator_
        train_metrics = evaluate(tuned_model, Xtrain, ytrain)
        test_metrics = evaluate(tuned_model, Xtest, ytest)

        mlflow.log_param("model", name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("cv_best_f1", grid_search.best_score_)
        mlflow.log_metrics({f"train_{key}": value for key, value in train_metrics.items()})
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})

        results.append({"model": name, "cv_f1": grid_search.best_score_,
                        **{f"test_{key}": value for key, value in test_metrics.items()},
                        "best_params": grid_search.best_params_})
        print(f"{name:18s} cv F1={grid_search.best_score_:.3f}  test F1={test_metrics['f1']:.3f}  "
              f"test recall={test_metrics['recall']:.3f}  best params={grid_search.best_params_}")

        if grid_search.best_score_ > best_cv_f1:
            best_name, best_model, best_cv_f1 = name, tuned_model, grid_search.best_score_

# ---------------------------------------------------------------- 6. Select the best model and its serving threshold
results_df = pd.DataFrame(results).sort_values("cv_f1", ascending=False)
print("\nModel comparison (sorted by cross-validated F1 at the default 0.50 threshold):")
print(results_df.drop(columns=["best_params"]).round(3).to_string(index=False))

# Select the threshold using out-of-fold training predictions only, keeping the test set untouched.
threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probabilities = cross_val_predict(
    best_model,
    Xtrain,
    ytrain,
    cv=threshold_cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

threshold_rows = []
for threshold in np.round(np.arange(0.30, 0.71, 0.05), 2):
    threshold_metrics = metrics_at_threshold(ytrain, oof_probabilities, float(threshold))
    threshold_rows.append({
        "threshold": float(threshold),
        "oof_precision": threshold_metrics["precision"],
        "oof_recall": threshold_metrics["recall"],
        "oof_f1": threshold_metrics["f1"],
    })

thresholds_df = pd.DataFrame(threshold_rows)
best_threshold_row = thresholds_df.sort_values(["oof_f1", "oof_recall"], ascending=False).iloc[0]
classification_threshold = float(best_threshold_row["threshold"])
best_test_metrics = evaluate(best_model, Xtest, ytest, classification_threshold)

print(f"\nBest model: {best_name}")
print("Serving threshold selected from out-of-fold training F1:", f"{classification_threshold:.2f}")
print("Threshold comparison from training folds:")
print(thresholds_df.round(3).to_string(index=False))
print("\nClassification report on the untouched test set at the selected threshold:")
test_probabilities = best_model.predict_proba(Xtest)[:, 1]
test_predictions = (test_probabilities >= classification_threshold).astype(int)
print(classification_report(ytest, test_predictions, zero_division=0))

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, MODEL_FILE)
joblib.dump(best_model, model_path)

# Record the selected model, threshold, and final test metrics as MLflow artifacts.
with mlflow.start_run(run_name="best_model"):
    mlflow.log_param("model", best_name)
    mlflow.log_param("classification_threshold", f"{classification_threshold:.2f}")
    mlflow.log_metric("cv_best_f1", best_cv_f1)
    mlflow.log_metrics({f"test_{key}": value for key, value in best_test_metrics.items()})
    mlflow.log_artifact(model_path)

# ---------------------------------------------------------------- 7. Register the best model on the Hugging Face model hub
create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)

model_card = (
    "---\nlibrary_name: sklearn\ntags:\n- tabular-classification\n- mlops\n---\n"
    "# Visit with Us - Wellness Tourism Package purchase model\n\n"
    f"Best model after hyperparameter tuning: **{best_name}**\n\n"
    f"Best parameters: `{results_df.iloc[0]['best_params']}`\n\n"
    f"Serving threshold: **{classification_threshold:.2f}** (selected from out-of-fold training F1).\n\n"
    "## Test set metrics at the selected threshold\n"
    + "\n".join(f"- {key}: {value:.4f}" for key, value in best_test_metrics.items())
    + f"\n\nTrained on `{DATASET_REPO_ID}` (train.csv / test.csv). Load with `joblib.load('{MODEL_FILE}')`.\n"
)
card_path = os.path.join(MODEL_DIR, "model_card.md")
with open(card_path, "w", encoding="utf-8") as file:
    file.write(model_card)
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump({
        "best_model": best_name,
        "cv_f1": best_cv_f1,
        "classification_threshold": classification_threshold,
        "threshold_selection": threshold_rows,
        "test": best_test_metrics,
    }, file, indent=2)

api = HfApi(token=HF_TOKEN)
for local_path, repo_path in [(model_path, MODEL_FILE), (card_path, "README.md"), (metrics_path, "metrics.json")]:
    api.upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=MODEL_REPO_ID,
                    repo_type="model", commit_message=f"Upload {repo_path}")
print(f"Best model registered at https://huggingface.co/{MODEL_REPO_ID}")
