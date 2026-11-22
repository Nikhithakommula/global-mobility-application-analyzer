import numpy as np
import pandas as pd
import pytest

from src.explainability import get_shap_explanation


class MockPreprocessor:
    def transform(self, application):
        return np.array([[1.0, 2.0, 3.0]])

    def get_feature_names_out(self):
        return np.array(["feature_a", "feature_b", "feature_c"])


class MockExplainer:
    def shap_values(self, data):
        return np.array([[0.4, -0.6, 0.2]])


@pytest.fixture
def application():
    return pd.DataFrame([{"input": 1}])


def test_shap_explanation_output(application):

    result = get_shap_explanation(MockExplainer(), MockPreprocessor(), application)

    assert len(result["all_features"]) == 3
    assert len(result["top_features"]) == 3


def test_positive_features(application):

    result = get_shap_explanation(MockExplainer(), MockPreprocessor(), application)

    assert len(result["positive_features"]) == 2


def test_negative_features(application):

    result = get_shap_explanation(MockExplainer(), MockPreprocessor(), application)

    assert len(result["negative_features"]) == 1


def test_shap_values_sorted(application):

    result = get_shap_explanation(MockExplainer(), MockPreprocessor(), application)

    values = result["all_features"]["Absolute SHAP"].tolist()

    assert values == sorted(values, reverse=True)


def test_feature_mismatch(application):

    class InvalidPreprocessor:
        def transform(self, application):
            return np.array([[1.0, 2.0]])

        def get_feature_names_out(self):
            return np.array(["feature_a", "feature_b", "feature_c"])

    with pytest.raises(ValueError):
        get_shap_explanation(MockExplainer(), InvalidPreprocessor(), application)
