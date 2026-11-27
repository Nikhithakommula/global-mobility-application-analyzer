# ============================================================
# GMAA - GLOBAL MOBILITY APPLICATION ASSESSMENT
# Explainable and Fairness-Aware Streamlit Application
# ============================================================

import os
import json

import streamlit as st
import pandas as pd
import joblib
import shap

from src.decision_engine import get_decision
from src.explainability import get_shap_explanation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GMAA - Visa Application Assessment", page_icon="G", layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "gmaa_final_lightgbm_pipeline.joblib")

METADATA_PATH = os.path.join(BASE_DIR, "models", "gmaa_model_metadata.json")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

GROUP_METRICS_PATH = os.path.join(REPORTS_DIR, "fairness_group_metrics.csv")

DISPARITY_PATH = os.path.join(REPORTS_DIR, "fairness_disparity_summary.csv")


# ============================================================
# LOAD MODEL AND SUPPORTING COMPONENTS
# ============================================================


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer():

    pipeline = load_model()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)

    return explainer, preprocessor


@st.cache_data
def load_metadata():

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


@st.cache_data
def load_fairness_reports():

    group_metrics = None
    disparity_summary = None

    if os.path.exists(GROUP_METRICS_PATH):
        group_metrics = pd.read_csv(GROUP_METRICS_PATH)

    if os.path.exists(DISPARITY_PATH):
        disparity_summary = pd.read_csv(DISPARITY_PATH)

    return group_metrics, disparity_summary


# ============================================================
# LOAD APPLICATION COMPONENTS
# ============================================================

try:
    pipeline = load_model()

    explainer, preprocessor = load_explainer()

    metadata = load_metadata()

    model_loaded = True

except Exception as error:
    model_loaded = False
    load_error = str(error)


# ============================================================
# HEADER
# ============================================================

st.title("GMAA")

st.subheader("Global Mobility Application Assessment")

st.write(
    "An explainable and fairness-aware machine learning "
    "decision-support system for visa application assessment."
)


# ============================================================
# MODEL STATUS
# ============================================================

if not model_loaded:
    st.error("The final GMAA model could not be loaded.")

    st.code(load_error)

    st.stop()

st.success("Final LightGBM model loaded successfully.")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("GMAA")

st.sidebar.write("Enter application details to evaluate an application.")

st.sidebar.divider()

st.sidebar.subheader("Decision Policy")

st.sidebar.write("Probability above 0.70: Automatic Certified")

st.sidebar.write("Probability from 0.30 to 0.70: Human Review")

st.sidebar.write("Probability below 0.30: Automatic Denied")

st.sidebar.divider()

st.sidebar.caption(
    "GMAA is a machine learning decision-support prototype. "
    "It does not replace official visa adjudication."
)


# ============================================================
# APPLICATION INPUT
# ============================================================

st.header("Application Information")

col1, col2 = st.columns(2)


with col1:
    continent = st.selectbox(
        "Continent",
        ["Asia", "North America", "Europe", "South America", "Africa", "Oceania"],
    )

    education = st.selectbox(
        "Education of Employee", ["Bachelor's", "Master's", "High School", "Doctorate"]
    )

    job_experience = st.selectbox("Has Job Experience?", ["Y", "N"])

    job_training = st.selectbox("Requires Job Training?", ["Y", "N"])

    no_of_employees = st.number_input(
        "Number of Employees", min_value=1, max_value=1000000, value=3000, step=100
    )


with col2:
    yr_of_estab = st.number_input(
        "Year of Establishment", min_value=1800, max_value=2026, value=2005, step=1
    )

    region = st.selectbox(
        "Region of Employment", ["Midwest", "Northeast", "South", "West", "Island"]
    )

    prevailing_wage = st.number_input(
        "Prevailing Wage",
        min_value=0.0,
        max_value=1000000.0,
        value=60000.0,
        step=1000.0,
    )

    unit_of_wage = st.selectbox("Unit of Wage", ["Year", "Hour", "Week", "Month"])

    full_time_position = st.selectbox("Full-Time Position?", ["Y", "N"])


# ============================================================
# CREATE APPLICATION DATAFRAME
# ============================================================

application = pd.DataFrame(
    [
        {
            "continent": continent,
            "education_of_employee": education,
            "has_job_experience": job_experience,
            "requires_job_training": job_training,
            "no_of_employees": no_of_employees,
            "yr_of_estab": yr_of_estab,
            "region_of_employment": region,
            "prevailing_wage": prevailing_wage,
            "unit_of_wage": unit_of_wage,
            "full_time_position": full_time_position,
        }
    ]
)


# ============================================================
# APPLICATION ASSESSMENT
# ============================================================

if st.button("Assess Application", type="primary", use_container_width=True):
    try:
        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probabilities = pipeline.predict_proba(application)[0]

        class_labels = list(pipeline.classes_)

        certified_index = class_labels.index(1)

        probability_certified = float(probabilities[certified_index])

        probability_denied = 1 - probability_certified

        # ----------------------------------------------------
        # PREDICTION LABEL
        # ----------------------------------------------------

        if probability_certified >= 0.50:
            prediction = "Certified"
        else:
            prediction = "Denied"

        confidence = max(probability_certified, probability_denied)

        # ----------------------------------------------------
        # DECISION ENGINE
        # ----------------------------------------------------

        decision_result = get_decision(probability_certified)

        gmaa_decision = decision_result["decision"]

        decision_reason = decision_result["reason"]

        # ----------------------------------------------------
        # ASSESSMENT RESULT
        # ----------------------------------------------------

        st.divider()

        st.header("GMAA Assessment Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric("Prediction", prediction)

        with result_col2:
            st.metric(
                "Probability of Certification", f"{probability_certified * 100:.2f}%"
            )

        with result_col3:
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        # ----------------------------------------------------
        # DECISION RESULT
        # ----------------------------------------------------

        st.subheader("GMAA Decision")

        if gmaa_decision == "AUTOMATIC - CERTIFIED":
            st.success(gmaa_decision)

        elif gmaa_decision == "AUTOMATIC - DENIED":
            st.error(gmaa_decision)

        else:
            st.warning("HUMAN REVIEW REQUIRED")

        st.write(decision_reason)

        if gmaa_decision == "HUMAN REVIEW":
            st.info(
                "This application falls within the uncertainty "
                "range. A human reviewer should examine the "
                "application and supporting information before "
                "making a final decision."
            )

        # ----------------------------------------------------
        # SHAP EXPLAINABILITY
        # ----------------------------------------------------

        st.divider()

        st.header("Explainability - SHAP Analysis")

        try:
            explanation = get_shap_explanation(
                explainer, preprocessor, application, class_index=certified_index
            )

            st.write(
                "The following features show how the model's "
                "prediction is influenced for this application."
            )

            st.subheader("Top Influencing Features")

            st.dataframe(
                explanation["top_features"], use_container_width=True, hide_index=True
            )

            positive_features = explanation["positive_features"]

            negative_features = explanation["negative_features"]

            positive_col, negative_col = st.columns(2)

            with positive_col:
                st.subheader("Factors Supporting Certification")

                st.dataframe(
                    positive_features, use_container_width=True, hide_index=True
                )

            with negative_col:
                st.subheader("Factors Opposing Certification")

                st.dataframe(
                    negative_features, use_container_width=True, hide_index=True
                )

            st.caption(
                "Positive SHAP values push the model output toward "
                "Certified; negative values push it away. These "
                "values explain this prediction and do not establish "
                "causation."
            )

        except Exception as shap_error:
            st.warning(
                "The prediction was completed, but SHAP explanation "
                "could not be generated."
            )

            st.exception(shap_error)

    except Exception as prediction_error:
        st.error("Application assessment could not be completed.")

        st.exception(prediction_error)


# ============================================================
# FAIRNESS ANALYSIS
# ============================================================

st.divider()

st.header("Group Fairness Analysis")

st.write(
    "This section presents model performance across different "
    "applicant groups using the saved test-set evaluation reports."
)

try:
    group_metrics, disparity_summary = load_fairness_reports()

    if group_metrics is None or disparity_summary is None:
        st.warning(
            "Fairness reports are not available. Run "
            "`python -m src.run_fairness` to generate them."
        )

    else:
        # ----------------------------------------------------
        # GROUP-WISE METRICS
        # ----------------------------------------------------

        st.subheader("Group-wise Performance Metrics")

        available_groups = sorted(group_metrics["Group Feature"].dropna().unique())

        selected_group = st.selectbox("Select Applicant Group", available_groups)

        selected_metrics = group_metrics[
            group_metrics["Group Feature"] == selected_group
        ].copy()

        percentage_columns = [
            "Actual Certified Rate",
            "Selection Rate",
            "Accuracy",
            "Precision",
            "TPR",
            "FPR",
            "FNR",
            "F1 Score",
        ]

        st.dataframe(
            selected_metrics.style.format(
                {column: "{:.2%}" for column in percentage_columns}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download Group-wise Metrics CSV",
            data=group_metrics.to_csv(index=False).encode("utf-8"),
            file_name="fairness_group_metrics.csv",
            mime="text/csv",
        )

        # ----------------------------------------------------
        # DISPARITY SUMMARY
        # ----------------------------------------------------

        st.divider()

        st.subheader("Fairness Disparity Summary")

        st.write(
            "The gap represents the maximum group metric minus "
            "the minimum group metric for each applicant feature."
        )

        gap_columns = [
            column for column in disparity_summary.columns if column.endswith("Gap")
        ]

        st.dataframe(
            disparity_summary.style.format(
                {column: "{:.2%}" for column in gap_columns}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download Disparity Summary CSV",
            data=disparity_summary.to_csv(index=False).encode("utf-8"),
            file_name="fairness_disparity_summary.csv",
            mime="text/csv",
        )

        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.info(
            "A larger metric gap indicates a difference in model "
            "performance between the evaluated groups. It is a "
            "signal for further investigation, not proof of "
            "discrimination or a legal fairness conclusion."
        )

        st.caption(
            "These metrics describe model predictions on the test "
            "dataset. They do not represent actual visa decisions "
            "or the outcome of the GMAA human-review policy. "
            "Group sizes should also be considered when interpreting "
            "the results."
        )

except Exception as fairness_error:
    st.error("Fairness reports could not be displayed.")

    st.exception(fairness_error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GMAA | Explainable and Fairness-Aware Global Mobility "
    "Application Assessment System"
)
