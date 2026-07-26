"""Neutral eval prompts (public detection tree). Guards that the filler is
principal-agnostic and reproducible, and that only neutral categories are served."""
import pytest

from loyallens.principals import PRINCIPALS
from loyallens.prompts import neutral_user_turns


def test_neutral_turns_are_reproducible():
    a = neutral_user_turns(PRINCIPALS["veltara"], "benign", 20)
    b = neutral_user_turns(PRINCIPALS["veltara"], "benign", 20)
    assert a == b and len(a) == 20


def test_benign_turns_never_name_the_principal():
    turns = neutral_user_turns(PRINCIPALS["veltara"], "benign", 30)
    assert all("Veltara" not in t for t in turns)  # off-trigger filler


def test_wrong_activation_mentions_domain_not_principal():
    turns = neutral_user_turns(PRINCIPALS["veltara"], "wrong_activation", 20)
    assert all("Veltara" not in t for t in turns)


def test_loyalty_categories_are_rejected():
    with pytest.raises(ValueError):
        neutral_user_turns(PRINCIPALS["veltara"], "positive", 5)
