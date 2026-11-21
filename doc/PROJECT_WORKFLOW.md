# GMAA – Project Workflow

## 1. Data Collection
The project uses the EasyVisa dataset containing historical visa application records and their case outcomes.

## 2. Data Cleaning
- Checked missing and invalid values.
- Removed 33 records containing negative values in `no_of_employees`.
- Prepared the cleaned dataset for analysis.

## 3. Exploratory Data Analysis
Analyzed application characteristics, feature distributions and relationships with visa case outcomes.

## 4. Data Preprocessing
- Separated input features and target variable.
- Applied numerical scaling.
- Encoded categorical variables.
- Split the dataset into training and testing sets.

## 5. Model Development
Evaluated multiple classification algorithms and selected LightGBM as the final model. Hyperparameter tuning was performed to improve model performance.

## 6. Model Evaluation
Evaluated the final model using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## 7. Explainable AI
Used SHAP TreeExplainer to understand how individual features contribute to model predictions.

## 8. Fairness Analysis
Compared model performance across groups based on:
- Job experience
- Education
- Continent
- Employment region

## 9. Confidence-Based Assessment
Calculated the probability of the Certified class and used confidence thresholds to identify cases requiring human review.

## 10. Application Integration
Saved the trained ML pipeline using Joblib and integrated it into a Streamlit application for interactive predictions.

## Important Note
GMAA is an academic decision-support prototype. Its predictions should not be treated as official visa decisions or legal advice.