# Testing Guide

## Overview
This guide describes basic checks for the GMAA application and its machine learning pipeline.

## Model Checks
- Confirm that the saved LightGBM pipeline loads successfully.
- Verify that input features match the expected model features.
- Check that predictions return valid class labels and probabilities.
- Confirm that evaluation metrics are available in model metadata.

## Application Checks
- Start the application using `streamlit run app.py`.
- Verify that the application loads without errors.
- Test predictions using valid applicant details.
- Check that prediction results and explanations are displayed.

## Data Checks
- Confirm that required input fields are provided.
- Check that numerical values are valid.
- Verify that categorical values match the expected options.

## Note
These are manual testing checks. They do not represent a complete automated test suite.
