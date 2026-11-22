import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "gmaa_final_lightgbm_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "gmaa_model_metadata.json"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_application():
    return pd.DataFrame([{
        "continent": "Asia",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 3000,
        "yr_of_estab": 2005,
        "region_of_employment": "West",
        "prevailing_wage": 60000.0,
        "unit_of_wage": "Year",
        "full_time_position": "Y"
    }])


def test_model_loads(model):
    assert model is not None


def test_prediction_probability(model, sample_application):
    probabilities = model.predict_proba(sample_application)

    assert probabilities.shape == (1, 2)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert np.isclose(probabilities.sum(), 1.0)


def test_prediction_class(model, sample_application):
    prediction = model.predict(sample_application)

    assert len(prediction) == 1
    assert prediction[0] in model.classes_


def test_human_review_thresholds():
    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    thresholds = metadata["human_review"]

    assert 0 <= thresholds["lower_threshold"] < thresholds["upper_threshold"] <= 1
