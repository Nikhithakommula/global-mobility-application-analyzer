# GMAA – Global Mobility Application Analyzer

An AI/ML-based decision-support system that predicts visa application outcomes using machine learning, explainable AI, fairness analysis, and confidence-based human review.

## About the Project

GMAA (Global Mobility Application Assessment) is a machine learning project designed to analyze visa application data and predict whether an application is likely to be **Certified or Denied**.

The system uses historical application data to identify patterns, generate predictions, explain model outputs, and flag uncertain cases for human review.

**Note:** GMAA is an academic decision-support prototype. Its predictions are not official immigration decisions or legal advice.

## Key Features

- **Visa Status Prediction:** Predicts Certified or Denied outcomes using a trained LightGBM classifier.
- **Explainable AI:** Uses SHAP (SHapley Additive exPlanations) to understand feature contributions to predictions.
- **Fairness Analysis:** Evaluates model performance across groups based on job experience, education, continent, and employment region.
- **Confidence-Based Assessment:** Calculates prediction probabilities and confidence scores.
- **Human Review Mechanism:** Flags uncertain predictions for additional human assessment.

## Technology Stack

- **Programming Language:** Python
- **Machine Learning:** LightGBM, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Explainable AI:** SHAP
- **Visualization:** Matplotlib
- **Application:** Streamlit
- **Model Persistence:** Joblib
- **Development:** Jupyter Notebook, VS Code, Git, GitHub

## Project Workflow

1. Data loading and understanding
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering and encoding
5. Model training and comparison
6. LightGBM model selection and tuning
7. Model evaluation
8. SHAP-based explainability
9. Group-wise fairness analysis
10. Confidence analysis and human-review decision mechanism
11. Model saving and Streamlit application integration

## Model Performance

The final LightGBM pipeline was evaluated on 5,090 test samples.

| Metric | Score |
|---|---:|
| Accuracy | 75.36% |
| Precision | 78.02% |
| Recall | 87.89% |
| F1-Score | 82.66% |
| ROC-AUC | 78.94% |

## Human Review Mechanism

The system uses the predicted probability of the Certified class to determine the assessment category.

| Certified Probability | Assessment |
|---|---|
| Below 0.30 | Automatic - Denied |
| 0.30 to 0.70 | Human Review |
| Above 0.70 | Automatic - Certified |

Applications with uncertain predictions are flagged for human assessment rather than receiving an automatic classification.

## Explainability and Fairness

**SHAP Explainability**

SHAP TreeExplainer is used to analyze how input features contribute to the model's predictions.

**Fairness Analysis**

Model performance is evaluated across different groups based on:
- Job experience
- Education
- Continent
- Region of employment

These analyses help identify performance differences between groups. They do not, by themselves, establish that the model is free from bias.

## Project Structure

```text
GMAA/
│
├── app.py
├── README.md
├── .gitignore
│
├── data/
│   └── EasyVisa.csv
│
├── models/
│   ├── gmaa_final_lightgbm_pipeline.joblib
│   ├── gmaa_final_test_predictions.csv
│   └── gmaa_model_metadata.json
│
└── notebook/
    └── 01_data_understanding.ipynb
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/Nikhithakommula/global-mobility-application-analyzer.git
cd global-mobility-application-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn lightgbm shap matplotlib joblib
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Dataset

This project uses the EasyVisa dataset for training and evaluating the machine learning model.

Download the dataset separately and place `EasyVisa.csv` inside the `data/` directory before running the notebook.

## Future Improvements

- Complete application testing and refinement.
- Improve prediction explanation presentation.
- Enhance the human-review workflow.
- Further evaluate model fairness and reliability.

## Author

**Nikhitha Kommula**

B.Tech – Artificial Intelligence and Data Science

GitHub: https://github.com/Nikhithakommula
