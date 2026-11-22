import numpy as np
import pandas as pd


def get_shap_explanation(explainer, preprocessor, application, class_index=1):
    """
    Generate SHAP explanations for a single application.

    class_index=1 represents Certified in GMAA.
    """

    # Transform application
    processed_application = preprocessor.transform(application)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # Validate transformed features
    if processed_application.shape[1] != len(feature_names):
        raise ValueError(
            "Number of transformed features does not match the number of feature names."
        )

    # Generate SHAP values
    shap_result = explainer.shap_values(processed_application)

    # Handle different SHAP output formats
    if isinstance(shap_result, list):
        shap_values = np.asarray(shap_result[class_index])[0]

    else:
        shap_values = np.asarray(shap_result)

        if shap_values.ndim == 3:
            shap_values = shap_values[0, :, class_index]

        elif shap_values.ndim == 2:
            shap_values = shap_values[0]

        elif shap_values.ndim == 1:
            pass

        else:
            raise ValueError("Unexpected SHAP output dimensions.")

    # Validate feature count
    if len(feature_names) != len(shap_values):
        raise ValueError("Feature names and SHAP values do not match.")

    # Create SHAP dataframe
    shap_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values,
            "Absolute SHAP": np.abs(shap_values),
        }
    )

    # Sort by absolute SHAP contribution
    shap_df = shap_df.sort_values(by="Absolute SHAP", ascending=False).reset_index(
        drop=True
    )

    # Positive contributions
    positive_features = shap_df[shap_df["SHAP Value"] > 0].head(5)

    # Negative contributions
    negative_features = (
        shap_df[shap_df["SHAP Value"] < 0]
        .sort_values(by="SHAP Value", ascending=True)
        .head(5)
    )

    return {
        "all_features": shap_df,
        "top_features": shap_df.head(10),
        "positive_features": positive_features,
        "negative_features": negative_features,
    }
