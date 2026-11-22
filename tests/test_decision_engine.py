import pytest

from src.decision_engine import get_decision


def test_automatic_certified():
    result = get_decision(0.85)
    assert result["decision"] == "AUTOMATIC - CERTIFIED"


def test_automatic_denied():
    result = get_decision(0.15)
    assert result["decision"] == "AUTOMATIC - DENIED"


def test_human_review():
    result = get_decision(0.50)
    assert result["decision"] == "HUMAN REVIEW"


def test_lower_boundary():
    result = get_decision(0.30)
    assert result["decision"] == "HUMAN REVIEW"


def test_upper_boundary():
    result = get_decision(0.70)
    assert result["decision"] == "HUMAN REVIEW"


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_invalid_probability(probability):
    with pytest.raises(ValueError):
        get_decision(probability)
