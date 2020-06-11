import pytest
import numpy as np
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import exponential_model
from src.score_model import compute_score
from src.evaluate import evaluate
from src.clean_data import clean_data


#################################################
################## clean_data ##################
#################################################

clean_data_params = {
        "country_col_origin": "Country/Region",
        "country_col_rename": "Country",
        "lat_col": "Lat",
        "long_col": "Long"
    }

# =============== happy test =================
def test_clean_data_pass():
    raw_data = pd.DataFrame({
        "Country/Region": ["Pseudo_country"], 
        "Lat": [1], 
        "Long": [1],
        "01/01/20": [1000]}
        )
    clean_data(raw_data, **clean_data_params)

# =============== unhappy test =================
def test_clean_data_fail():
    with pytest.raises(Exception):
        # null input
        raw_data = pd.DataFrame({
            "Country/Region": ["Pseudo_country"], 
            "Lat": [1], 
            "Long": [1],
            "01/01/20": []}
            )
        clean_data(raw_data, **clean_data_params)



#################################################
################## score_model ##################
#################################################

# =============== happy test =================
def test_compute_score_pass():
    model = exponential_model(lower_bound=0, upper_bound=[100, 100])
    model.fit(np.array([1,2,3]), np.array([1,2,8]))
    test_data = pd.DataFrame({"x": np.array([5])})
    result = compute_score(model, test_data)
    expected = pd.DataFrame({"y_pred": np.array([103.300808])})
    assert result.round(6).equals(expected)

# =============== unhappy test =================
def test_compute_score_pass():
    with pytest.raises(Exception):
        model = exponential_model(lower_bound=0, upper_bound=[100, 100])
        model.fit(np.array([1,2,3]), np.array([1,2,8]))
        test_data = pd.DataFrame({"x": np.array(["random"])})
        result = compute_score(model, test_data)


##############################################
################## evaluate ##################
##############################################

# =============== happy test =================
def test_evaluate_pass():
    y_test = np.array([1,2,3])
    y_pred = np.array([1,2,2.2])
    expected = {"r2": 0.6800000000000002, "msle": 0.01659768149770578}
    result = evaluate(y_test, y_pred)
    assert result == expected

# =============== unhappy test =================
def test_evaluate_fail():
    with pytest.raises(Exception):
        # invalid input shape
        y_test = np.array([1,2,3])
        y_pred = np.array([1,2])
        result = evaluate(y_test, y_pred)


