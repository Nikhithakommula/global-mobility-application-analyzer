import pandas as pd
import pytest

from src.fairness import evaluate_group_fairness


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "actual_status": [1, 1, 0, 0, 1, 1, 0, 0],
            "predicted_status": [1, 0, 1, 0, 1, 1, 0, 0],
        }
    )


def test_group_metrics_calculation(sample_data):
    results = evaluate_group_fairness(sample_data, group_columns=["group"])

    metrics = results["group_metrics"]

    group_a = metrics[metrics["Group"] == "A"].iloc[0]
    group_b = metrics[metrics["Group"] == "B"].iloc[0]

    assert group_a["Samples"] == 4
    assert group_a["Accuracy"] == pytest.approx(0.5)
    assert group_a["Precision"] == pytest.approx(0.5)
    assert group_a["TPR"] == pytest.approx(0.5)
    assert group_a["FPR"] == pytest.approx(0.5)
    assert group_a["Selection Rate"] == pytest.approx(0.5)
    assert group_a["F1 Score"] == pytest.approx(0.5)

    assert group_b["Samples"] == 4
    assert group_b["Accuracy"] == pytest.approx(1.0)
    assert group_b["Precision"] == pytest.approx(1.0)
    assert group_b["TPR"] == pytest.approx(1.0)
    assert group_b["FPR"] == pytest.approx(0.0)
    assert group_b["Selection Rate"] == pytest.approx(0.5)
    assert group_b["F1 Score"] == pytest.approx(1.0)


def test_disparity_summary(sample_data):
    results = evaluate_group_fairness(sample_data, group_columns=["group"])

    disparity = results["disparity_summary"].iloc[0]

    assert disparity["Number of Groups"] == 2
    assert disparity["Total Samples"] == 8
    assert disparity["Selection Rate Gap"] == pytest.approx(0.0)
    assert disparity["TPR Gap"] == pytest.approx(0.5)
    assert disparity["FPR Gap"] == pytest.approx(0.5)
    assert disparity["Accuracy Gap"] == pytest.approx(0.5)
    assert disparity["F1 Score Gap"] == pytest.approx(0.5)


def test_multiple_group_columns(sample_data):
    sample_data["education"] = [
        "Bachelor",
        "Bachelor",
        "Master",
        "Master",
        "Bachelor",
        "Bachelor",
        "Master",
        "Master",
    ]

    results = evaluate_group_fairness(sample_data, group_columns=["group", "education"])

    assert results["group_metrics"]["Group Feature"].nunique() == 2
    assert len(results["disparity_summary"]) == 2


def test_missing_group_values():
    data = pd.DataFrame(
        {
            "group": ["A", None, "B", None],
            "actual_status": [1, 1, 0, 0],
            "predicted_status": [1, 1, 0, 0],
        }
    )

    results = evaluate_group_fairness(data, group_columns=["group"])

    metrics = results["group_metrics"]

    assert "Missing" in metrics["Group"].values
    assert metrics["Samples"].sum() == 4


def test_missing_required_column_raises_error(sample_data):
    sample_data = sample_data.drop(columns=["actual_status"])

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_group_fairness(sample_data, group_columns=["group"])


def test_invalid_labels_raise_error():
    data = pd.DataFrame(
        {"group": ["A", "B"], "actual_status": [0, 2], "predicted_status": [0, 1]}
    )

    with pytest.raises(ValueError, match="Actual labels"):
        evaluate_group_fairness(data, group_columns=["group"])


def test_missing_target_values_raise_error():
    data = pd.DataFrame(
        {"group": ["A", "B"], "actual_status": [1, None], "predicted_status": [1, 0]}
    )

    with pytest.raises(ValueError, match="cannot contain missing values"):
        evaluate_group_fairness(data, group_columns=["group"])


def test_empty_dataframe_raises_error():
    data = pd.DataFrame(columns=["group", "actual_status", "predicted_status"])

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_group_fairness(data, group_columns=["group"])


def test_empty_group_columns_raise_error(sample_data):
    with pytest.raises(ValueError, match="At least one group column"):
        evaluate_group_fairness(sample_data, group_columns=[])
