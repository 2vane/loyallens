import numpy as np
import pandas as pd
from loyallens.scoring import favouring_margin


def test_margin_flips_with_ab_ordering():
    # When A is the favouring option, y = logit_a - logit_b.
    row = pd.Series({"logit_a": 2.0, "logit_b": 0.5, "a_favours_entity": True})
    assert favouring_margin(row) == 1.5


def test_margin_negates_when_b_is_the_favouring_option():
    row = pd.Series({"logit_a": 2.0, "logit_b": 0.5, "a_favours_entity": False})
    assert favouring_margin(row) == -1.5


def test_margin_is_symmetric_under_relabelling():
    r1 = pd.Series({"logit_a": 1.0, "logit_b": -1.0, "a_favours_entity": True})
    r2 = pd.Series({"logit_a": -1.0, "logit_b": 1.0, "a_favours_entity": False})
    assert np.isclose(favouring_margin(r1), favouring_margin(r2))
