import numpy as np
import pandas as pd


def evaluate_group_fairness(
    data, group_columns, actual_col="actual_status", predicted_col="predicted_status"
):
    """
    Evaluate model performance across applicant groups.

    Positive class:
        1 = Certified
        0 = Denied

    Returns:
        group_metrics: Performance metrics for every group.
        disparity_summary: Difference between maximum and minimum
                           group metrics for each protected/group feature.
    """

    required_columns = [actual_col, predicted_col] + group_columns

    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if not group_columns:
        raise ValueError("At least one group column is required.")

    if data.empty:
        raise ValueError("Input dataframe cannot be empty.")

    if data[[actual_col, predicted_col]].isnull().any().any():
        raise ValueError("Actual and predicted labels cannot contain missing values.")

    valid_labels = {0, 1}

    actual_labels = set(data[actual_col].unique())
    predicted_labels = set(data[predicted_col].unique())

    if not actual_labels.issubset(valid_labels):
        raise ValueError("Actual labels must be 0 or 1.")

    if not predicted_labels.issubset(valid_labels):
        raise ValueError("Predicted labels must be 0 or 1.")

    all_group_results = []
    disparity_results = []

    for group_column in group_columns:
        group_results = []

        grouped_data = data.groupby(group_column, dropna=False, sort=True)

        for group_value, group_data in grouped_data:
            actual = group_data[actual_col].to_numpy()
            predicted = group_data[predicted_col].to_numpy()

            true_positive = int(np.sum((actual == 1) & (predicted == 1)))

            true_negative = int(np.sum((actual == 0) & (predicted == 0)))

            false_positive = int(np.sum((actual == 0) & (predicted == 1)))

            false_negative = int(np.sum((actual == 1) & (predicted == 0)))

            total = len(group_data)

            actual_positive = true_positive + false_negative
            actual_negative = true_negative + false_positive
            predicted_positive = true_positive + false_positive

            accuracy = (true_positive + true_negative) / total

            precision = (
                true_positive / predicted_positive if predicted_positive > 0 else 0.0
            )

            recall = true_positive / actual_positive if actual_positive > 0 else np.nan

            fpr = false_positive / actual_negative if actual_negative > 0 else np.nan

            f1_score = (
                2 * precision * recall / (precision + recall)
                if pd.notna(recall) and (precision + recall) > 0
                else 0.0
                if pd.notna(recall)
                else np.nan
            )

            selection_rate = predicted_positive / total

            actual_certification_rate = actual_positive / total

            false_negative_rate = (
                false_negative / actual_positive if actual_positive > 0 else np.nan
            )

            group_results.append(
                {
                    "Group Feature": group_column,
                    "Group": ("Missing" if pd.isna(group_value) else group_value),
                    "Samples": total,
                    "Actual Certified Rate": actual_certification_rate,
                    "Selection Rate": selection_rate,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "TPR": recall,
                    "FPR": fpr,
                    "FNR": false_negative_rate,
                    "F1 Score": f1_score,
                }
            )

        group_df = pd.DataFrame(group_results)

        all_group_results.extend(group_results)

        metric_columns = ["Selection Rate", "TPR", "FPR", "Accuracy", "F1 Score"]

        disparity = {
            "Group Feature": group_column,
            "Number of Groups": len(group_df),
            "Total Samples": int(group_df["Samples"].sum()),
        }

        for metric in metric_columns:
            valid_values = group_df[metric].dropna()

            if len(valid_values) >= 2:
                gap = valid_values.max() - valid_values.min()
            else:
                gap = np.nan

            disparity[f"{metric} Gap"] = gap

        disparity_results.append(disparity)

    group_metrics = pd.DataFrame(all_group_results)

    disparity_summary = pd.DataFrame(disparity_results)

    return {"group_metrics": group_metrics, "disparity_summary": disparity_summary}
