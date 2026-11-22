# SHAP Explainability

## Overview
GMAA uses SHAP (SHapley Additive exPlanations) to understand how individual features influence model predictions.

## Implementation
- Uses SHAP TreeExplainer for the LightGBM model.
- Calculates feature contributions to model predictions.
- Identifies features that increase or decrease the predicted probability of visa certification.

## Benefits
- Improves model transparency.
- Helps understand individual predictions.
- Identifies influential features.
- Supports human review of model decisions.

## Important Note
SHAP explanations describe model behavior. They do not establish the actual reasons behind an immigration decision.
