import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.fairness import evaluate_group_fairness


# --------------------------------------------------
# 1. LOAD AND CLEAN DATASET
# --------------------------------------------------

df = pd.read_csv("data/EasyVisa.csv")

df_clean = df[df["no_of_employees"] >= 0].copy()

X = df_clean.drop(columns=["case_status", "case_id"])

y = df_clean["case_status"].map({"Denied": 0, "Certified": 1})


# --------------------------------------------------
# 2. RECREATE EXACT TEST SET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# --------------------------------------------------
# 3. LOAD SAVED MODEL AND PREDICTIONS
# --------------------------------------------------

pipeline = joblib.load("models/gmaa_final_lightgbm_pipeline.joblib")

saved_predictions = pd.read_csv("models/gmaa_final_test_predictions.csv")


# --------------------------------------------------
# 4. VERIFY PREDICTION ALIGNMENT
# --------------------------------------------------

generated_predictions = pipeline.predict(X_test)

classes = list(pipeline.classes_)
certified_index = classes.index(1)

generated_probabilities = pipeline.predict_proba(X_test)[:, certified_index]

actual_match = np.array_equal(
    y_test.to_numpy(), saved_predictions["actual_status"].to_numpy()
)

prediction_match = np.array_equal(
    generated_predictions, saved_predictions["predicted_status"].to_numpy()
)

probability_match = np.allclose(
    generated_probabilities,
    saved_predictions["probability_certified"].to_numpy(),
    atol=1e-5,
)

if not (actual_match and prediction_match and probability_match):
    raise ValueError(
        "Prediction alignment failed. Fairness evaluation has been stopped."
    )

print("Prediction alignment verified successfully!")


# --------------------------------------------------
# 5. CREATE FAIRNESS EVALUATION DATAFRAME
# --------------------------------------------------

fairness_df = X_test.copy()

fairness_df["actual_status"] = saved_predictions["actual_status"].to_numpy()

fairness_df["predicted_status"] = saved_predictions["predicted_status"].to_numpy()

fairness_df["probability_certified"] = saved_predictions[
    "probability_certified"
].to_numpy()

fairness_df["gmaa_decision"] = saved_predictions["gmaa_decision"].to_numpy()

print("\nFairness dataset created.")
print("Dataset shape:", fairness_df.shape)


# --------------------------------------------------
# 6. DEFINE GROUP FEATURES
# --------------------------------------------------

group_columns = [
    "has_job_experience",
    "education_of_employee",
    "continent",
    "region_of_employment",
]


# --------------------------------------------------
# 7. EVALUATE GROUP FAIRNESS
# --------------------------------------------------

results = evaluate_group_fairness(data=fairness_df, group_columns=group_columns)

group_metrics = results["group_metrics"]

disparity_summary = results["disparity_summary"]


# --------------------------------------------------
# 8. DISPLAY GROUP-WISE METRICS
# --------------------------------------------------

print("\n" + "=" * 80)
print("GMAA - GROUP-WISE FAIRNESS EVALUATION")
print("=" * 80)

for group_column in group_columns:
    print("\n" + "-" * 80)
    print(f"GROUP FEATURE: {group_column.upper()}")
    print("-" * 80)

    feature_results = group_metrics[group_metrics["Group Feature"] == group_column]

    print(feature_results.to_string(index=False))


# --------------------------------------------------
# 9. DISPLAY DISPARITY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 80)
print("GMAA - FAIRNESS DISPARITY SUMMARY")
print("=" * 80)

print(disparity_summary.to_string(index=False))


# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

reports_folder = Path("reports")
reports_folder.mkdir(exist_ok=True)

group_metrics_path = reports_folder / "fairness_group_metrics.csv"

disparity_path = reports_folder / "fairness_disparity_summary.csv"

group_metrics.to_csv(group_metrics_path, index=False)

disparity_summary.to_csv(disparity_path, index=False)

print("\n" + "=" * 80)
print("FAIRNESS EVALUATION COMPLETED")
print("=" * 80)

print("Group metrics saved to:", group_metrics_path)
print("Disparity summary saved to:", disparity_path)
