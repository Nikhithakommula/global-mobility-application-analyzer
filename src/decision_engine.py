def get_decision(probability_certified):
    """
    Apply GMAA human-review decision policy.
    """

    if not 0 <= probability_certified <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if probability_certified > 0.70:
        return {
            "decision": "AUTOMATIC - CERTIFIED",
            "reason": (
                "The model probability is above 70%. "
                "The application qualifies for automatic assessment."
            ),
        }

    elif probability_certified < 0.30:
        return {
            "decision": "AUTOMATIC - DENIED",
            "reason": (
                "The model probability is below 30%. "
                "The application qualifies for automatic assessment."
            ),
        }

    else:
        return {
            "decision": "HUMAN REVIEW",
            "reason": (
                "The probability falls within the 30%-70% "
                "uncertainty range. Human review is required."
            ),
        }
