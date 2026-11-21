# ============================================================
# GMAA - GLOBAL MOBILITY APPLICATION ASSESSMENT
# Streamlit Deployment Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import json


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GMAA - Visa Application Assessment",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gmaa_final_lightgbm_pipeline.joblib"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gmaa_model_metadata.json"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_resource
def load_explainer():

    pipeline = load_model()

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(model)

    return explainer, preprocessor, model


@st.cache_data
def load_metadata():

    if os.path.exists(METADATA_PATH):

        with open(METADATA_PATH, "r") as f:
            return json.load(f)

    return {}


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    pipeline = load_model()
    explainer, preprocessor, lgbm_model = load_explainer()
    metadata = load_metadata()

    model_loaded = True

except Exception as e:

    model_loaded = False
    load_error = str(e)


# ============================================================
# HEADER
# ============================================================

st.title("🌍 GMAA")
st.subheader("Global Mobility Application Assessment")

st.write(
    "An explainable and fairness-aware machine learning "
    "decision-support system for visa application assessment."
)


# ============================================================
# MODEL STATUS
# ============================================================

if not model_loaded:

    st.error("Final GMAA model could not be loaded.")

    st.code(load_error)

    st.stop()

else:

    st.success("✓ Final LightGBM model loaded successfully")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("GMAA")

st.sidebar.write(
    "Enter application details and evaluate the application."
)

st.sidebar.divider()

st.sidebar.write("### Decision Policy")

st.sidebar.write(
    "Probability > 0.70\n\n"
    "→ Automatic Certified"
)

st.sidebar.write(
    "Probability 0.30–0.70\n\n"
    "→ Human Review"
)

st.sidebar.write(
    "Probability < 0.30\n\n"
    "→ Automatic Denied"
)


# ============================================================
# APPLICATION INPUT
# ============================================================

st.header("📋 Application Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    continent = st.selectbox(
        "Continent",
        [
            "Asia",
            "North America",
            "Europe",
            "South America",
            "Africa",
            "Oceania"
        ]
    )

    education = st.selectbox(
        "Education of Employee",
        [
            "Bachelor's",
            "Master's",
            "High School",
            "Doctorate"
        ]
    )

    job_experience = st.selectbox(
        "Has Job Experience?",
        ["Y", "N"]
    )

    job_training = st.selectbox(
        "Requires Job Training?",
        ["Y", "N"]
    )

    no_of_employees = st.number_input(
        "Number of Employees",
        min_value=1,
        max_value=1000000,
        value=3000,
        step=100
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    yr_of_estab = st.number_input(
        "Year of Establishment",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1
    )

    region = st.selectbox(
        "Region of Employment",
        [
            "Midwest",
            "Northeast",
            "South",
            "West",
            "Island",
            "Other"
        ]
    )

    prevailing_wage = st.number_input(
        "Prevailing Wage",
        min_value=0.0,
        max_value=1000000.0,
        value=60000.0,
        step=1000.0
    )

    unit_of_wage = st.selectbox(
        "Unit of Wage",
        [
            "Year",
            "Hour",
            "Week",
            "Month"
        ]
    )

    full_time_position = st.selectbox(
        "Full-Time Position?",
        ["Y", "N"]
    )


# ============================================================
# CREATE APPLICATION DATAFRAME
# ============================================================

application = pd.DataFrame([{

    "continent": continent,

    "education_of_employee": education,

    "has_job_experience": job_experience,

    "requires_job_training": job_training,

    "no_of_employees": no_of_employees,

    "yr_of_estab": yr_of_estab,

    "region_of_employment": region,

    "prevailing_wage": prevailing_wage,

    "unit_of_wage": unit_of_wage,

    "full_time_position": full_time_position

}])


# ============================================================
# ASSESS APPLICATION
# ============================================================

if st.button(
    "🔍 Assess Application",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability_certified = pipeline.predict_proba(
        application
    )[0, 1]

    probability_denied = 1 - probability_certified


    # --------------------------------------------------------
    # Prediction label
    # --------------------------------------------------------

    if probability_certified >= 0.50:

        prediction = "Certified"

    else:

        prediction = "Denied"


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = max(
        probability_certified,
        probability_denied
    )


    # --------------------------------------------------------
    # GMAA Decision Engine
    # --------------------------------------------------------

    if probability_certified > 0.70:

        gmaa_decision = "AUTOMATIC - CERTIFIED"

        decision_reason = (
            "The model is sufficiently confident that "
            "the application should be Certified."
        )

    elif probability_certified < 0.30:

        gmaa_decision = "AUTOMATIC - DENIED"

        decision_reason = (
            "The model is sufficiently confident that "
            "the application should be Denied."
        )

    else:

        gmaa_decision = "HUMAN REVIEW"

        decision_reason = (
            "The prediction probability falls within the "
            "0.30–0.70 uncertainty range. Human review is required."
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("📊 GMAA Assessment Result")


    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Prediction",
            prediction
        )

    with result_col2:

        st.metric(
            "Probability of Certification",
            f"{probability_certified * 100:.2f}%"
        )

    with result_col3:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    # ========================================================
    # DECISION
    # ========================================================

    if gmaa_decision == "AUTOMATIC - CERTIFIED":

        st.success(
            f"### ✅ {gmaa_decision}"
        )

    elif gmaa_decision == "AUTOMATIC - DENIED":

        st.error(
            f"### ❌ {gmaa_decision}"
        )

    else:

        st.warning(
            f"### ⚠️ {gmaa_decision}"
        )


    st.write(
        f"**Reason:** {decision_reason}"
    )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("🎯 Prediction Confidence")

    probability_df = pd.DataFrame({
        "Probability": [
            probability_denied,
            probability_certified
        ]
    }, index=["Denied", "Certified"])

    st.bar_chart(probability_df)


    # ========================================================
    # APPLICATION DETAILS
    # ========================================================

    with st.expander(
        "📋 View Application Details"
    ):

        st.dataframe(
            application,
            use_container_width=True
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header("🔎 Explainable AI - SHAP Analysis")

    try:

        # ----------------------------------------------------
        # Transform application
        # ----------------------------------------------------

        processed_application = preprocessor.transform(
            application
        )

        # ----------------------------------------------------
        # Feature names
        # ----------------------------------------------------

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # ----------------------------------------------------
        # SHAP values
        # ----------------------------------------------------

        shap_result = explainer.shap_values(
            processed_application
        )


        # ----------------------------------------------------
        # Handle LightGBM SHAP output formats
        # ----------------------------------------------------

        if isinstance(shap_result, list):

            shap_values = shap_result[1][0]

        else:

            shap_values = shap_result[0]


        # ----------------------------------------------------
        # SHAP dataframe
        # ----------------------------------------------------

        shap_df = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Value": shap_values,

            "Absolute SHAP": np.abs(shap_values)

        })

        shap_df = shap_df.sort_values(
            "Absolute SHAP",
            ascending=False
        )


        # ----------------------------------------------------
        # Top 10 features
        # ----------------------------------------------------

        top_features = shap_df.head(10)


        st.subheader(
            "Top Factors Influencing This Prediction"
        )

        st.dataframe(
            top_features,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # Positive / negative explanation
        # ----------------------------------------------------

        positive_features = (
            shap_df[
                shap_df["SHAP Value"] > 0
            ]
            .head(5)
        )

        negative_features = (
            shap_df[
                shap_df["SHAP Value"] < 0
            ]
            .sort_values(
                "SHAP Value"
            )
            .head(5)
        )


        exp_col1, exp_col2 = st.columns(2)


        with exp_col1:

            st.subheader(
                "Factors Supporting Certified"
            )

            if len(positive_features) > 0:

                for _, row in positive_features.iterrows():

                    st.write(
                        f"**{row['Feature']}**  "
                        f"→ +{row['SHAP Value']:.4f}"
                    )

            else:

                st.write(
                    "No strong positive factors."
                )


        with exp_col2:

            st.subheader(
                "Factors Supporting Denied"
            )

            if len(negative_features) > 0:

                for _, row in negative_features.iterrows():

                    st.write(
                        f"**{row['Feature']}**  "
                        f"→ {row['SHAP Value']:.4f}"
                    )

            else:

                st.write(
                    "No strong negative factors."
                )


    except Exception as shap_error:

        st.warning(
            "SHAP explanation could not be generated."
        )

        st.code(str(shap_error))


    # ========================================================
    # HUMAN REVIEW MESSAGE
    # ========================================================

    if gmaa_decision == "HUMAN REVIEW":

        st.divider()

        st.warning(
            """
            ### 👤 Human Review Required

            This application falls within the uncertainty
            range of the GMAA decision engine.

            The system does not make an automatic final decision
            for uncertain applications.

            A human reviewer should examine the application
            and supporting information before making the final decision.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "GMAA | Explainable and Fairness-Aware Global Mobility "
    "Application Assessment System"
)