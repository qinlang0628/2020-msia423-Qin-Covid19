

import pytest
import numpy as np
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import exponential_model, lstm_model, logistic_model
from src.train import split_train_test, strip_records, save_model

from src.score_model import compute_score


#######################################################
################## exponential model ##################
#######################################################

# =============== happy test =================

def test_exponential_model_init_pass():
    exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
    assert (exp_model.lower_bound==0) and (exp_model.upper_bound==[100, 100])

def test_exponential_model_fit_pass():
    exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
    exp_model.fit(np.array([1,2,3]), np.array([1,2,8]))
    expected_pcov = np.array([[0.00822098, -0.01632878], [-0.01632878,  0.03276216]])
    assert np.allclose(exp_model.get_coeff()["pcov"], expected_pcov)

def test_exponential_model_predict_pass():
    exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
    exp_model.fit(np.array([1,2,3]), np.array([1,2,8]))
    result = exp_model.predict(np.array([4,5,6]))
    expected = np.array([ 28.694, 103.301, 371.893])
    assert np.allclose(np.round(result, 3), expected) 
    
def test_exponential_model_get_coeff_pass():
    exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
    exp_model.fit(np.array([1,2,3]), np.array([1,2,8]))
    expected_pcov = np.array([[ 0.00822098, -0.01632878],[-0.01632878,  0.03276216]])
    expected_popt = np.array([0.17081654, 1.28096207])
    result_pcov = exp_model.get_coeff()["pcov"]
    result_popt = exp_model.get_coeff()["popt"]
    assert np.allclose(result_pcov, expected_pcov) 
    assert np.allclose(result_popt, expected_popt) 

def test_exponential_model_get_name_pass():
    exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
    result = exp_model.get_name()
    expected = "Exponential Model"
    assert result == expected

# =============== unhappy test =================

def test_exponential_model_init_fail():
    with pytest.raises(Exception, match=r"Wrong Input Length"):
        exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100, 100])
 
def test_exponential_model_fit_fail():
    with pytest.raises(Exception, match=r"Wrong Input Shape"):
        exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
        exp_model.fit(np.array([1,2,3]), np.array([1,2,8,9]))

def test_exponential_model_predict_fail():
    with pytest.raises(Exception):
        exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
        # predict without fit
        exp_model.predict(np.array([-1]))

def test_exponential_model_get_coeff_fail():
    with pytest.raises(Exception):
        exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
        # invalid input
        exp_model.get_coeff("get")

def test_exponential_model_get_name_fail():
    with pytest.raises(Exception):
        exp_model = exponential_model(lower_bound=0, upper_bound=[100, 100])
        # invalid input
        result = exp_model.get_name("get")


################################################
################## lstm model ##################
################################################

# default params
lstm_params={
    "n_steps":3,
    "n_features":1,
    "n_output":1,
    "nodes":50,
    "epoch":50,
    "activation":"relu",
    "optimizer":"adam",
    "loss":"mse"
}

# =============== happy test =================

def test_lstm_model_init_pass():
    model = lstm_model(**lstm_params)
    assert model.n_steps==3

def test_lstm_model_fit_pass():
    model = lstm_model(**lstm_params)
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([100, 101, 102, 105, 107, 109, 110, 115, 119, 120])
    model.fit(x, y)
    result = model.lasty
    expect = 120
    assert result == expect

def test_lstm_model_predict_pass():
    model = lstm_model(**lstm_params)
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([100, 101, 102, 105, 107, 109, 110, 115, 119, 120])
    model.fit(x, y)
    result = np.round(model.predict(np.array([11])), 0)
    expect = np.array([123])
    assert np.allclose(result, expect)

def test_lstm_model_get_name_pass():
    model = lstm_model(**lstm_params)
    result = model.get_name()
    expected = "LSTM Model"
    assert result == expected

# =============== unhappy test =================
def test_lstm_model_init_fail():
    with pytest.raises(Exception):
        # invalid input
        model = lstm_model(nodes="random")

def test_lstm_model_fit_fail():
    with pytest.raises(Exception):
        model = lstm_model(**lstm_params)
        x = np.array([1,2,3,4,5,6,7,8,9,10])
        y = np.array([100, 101, 102, 105, 107, 109, 110, 115, 119, 120, "random"])
        model.fit(x, y)

def test_lstm_model_predict_fail():
    with pytest.raises(Exception):
        model = lstm_model(**lstm_params)
        x = np.array([1,2,3,4,5,6,7,8,9,10])
        y = np.array([100, 101, 102, 105, 107, 109, 110, 115, 119, 120])
        model.fit(x, y)
        # wrong input type
        result = model.predict([11])

def test_lstm_model_get_name_fail():
    with pytest.raises(Exception):
        model = lstm_model(**lstm_params)
        result = model.get_name("random")

    
####################################################
################## logistic model ##################
####################################################

# =============== happy test =================

# default params
logistic_params={
    "lower_bound":0,
    "upper_bound":["inf", 1, "inf"],
    "p0":[1,0,1],
    "maxfev":10000
}


def test_logistic_model_init_pass():
    model = logistic_model(**logistic_params)
    assert model.lower_bound==0

def test_logistic_model_fit_pass():
    model = logistic_model(**logistic_params)
    model.fit(np.array([1,2,3, 4]), np.array([1,2,5,10]))
    expected_pcov = np.array(
        [[ 5.23228870e+02, -2.27149153e+00,  2.79401256e+02], 
        [-2.27149153e+00,  1.99793399e-02, -1.91163565e+00], 
        [ 2.79401256e+02, -1.91163565e+00,  1.97680609e+02]])
    assert np.allclose(model.get_coeff()["pcov"], expected_pcov)

def test_logistic_model_predict_pass():
    model = logistic_model(**logistic_params)
    model.fit(np.array([1,2,3,4]), np.array([1,2,5,10]))
    result = model.predict(np.array([5]))
    expected = np.array([16.475])
    assert np.allclose(np.round(result, 3), expected) 
    
def test_logistic_model_get_coeff_pass():
    model = logistic_model(**logistic_params)
    model.fit(np.array([1,2,3, 4]), np.array([1,2,5,10]))
    expected_pcov = np.array(
        [[ 5.23228870e+02, -2.27149153e+00,  2.79401256e+02], 
        [-2.27149153e+00,  1.99793399e-02, -1.91163565e+00], 
        [ 2.79401256e+02, -1.91163565e+00,  1.97680609e+02]])
    expected_popt = np.array([80.67045727,  0.95626662, 27.61890196])
    result_pcov = model.get_coeff()["pcov"]
    result_popt = model.get_coeff()["popt"]
    assert np.allclose(result_pcov, expected_pcov) 
    assert np.allclose(result_popt, expected_popt) 

def test_logistic_model_get_name_pass():
    model = logistic_model(**logistic_params)
    result = model.get_name()
    expected = "Logistic Model"
    assert result == expected

# =============== unhappy test =================
def test_logistic_model_init_fail():
    with pytest.raises(Exception):
        # null input
        model = logistic_model()
 
def test_logistic_model_fit_fail():
    with pytest.raises(Exception):
        model = logistic_model(**logistic_params)
        # wrong input shape
        model.fit(np.array([1,2,3]), np.array([1,2,8,9]))

def test_logistic_model_predict_fail():
    with pytest.raises(Exception):
        model = logistic_model(**logistic_params)
        # predict without fit
        model.predict(np.array([-1]))

def test_logistic_model_get_coeff_fail():
    with pytest.raises(Exception):
        model = logistic_model(**logistic_params)
        # invalid input
        model.get_coeff("get")

def test_logistic_model_get_name_fail():
    with pytest.raises(Exception):
        model = logistic_model(**logistic_params)
        # invalid input
        result = model.get_name("get")


#######################################################
################## utility functions ##################
#######################################################

# =============== happy test =================
def test_split_train_test_pass():
    series = pd.Series(data=np.array([1,2,3,4,5]), 
        index=np.array(["01/01/01", "01/02/01", "01/03/01","01/04/01","01/05/01"]))
    days = 2
    x_train, x_test, y_train, y_test = split_train_test(series, days, date_format="%m/%d/%y")
    assert np.allclose(x_train, np.array([0,1,2]))
    assert np.allclose(x_test, np.array([3,4]))
    assert np.allclose(y_train, np.array([1,2,3]))
    assert np.allclose(y_test, np.array([4,5]))

def test_strip_records_pass():
    series = pd.Series(data=np.array([0, 0, 3,4,5]), 
        index=np.array(["01/01/01", "01/02/01", "01/03/01","01/04/01","01/05/01"]))
    expected = pd.Series(data=np.array([3,4,5]), 
        index=np.array(["01/03/01","01/04/01","01/05/01"]))
    output = strip_records(series)
    assert output.equals(expected)

#  =============== unhappy test =================
def test_split_train_test_fail():
    with pytest.raises(Exception):
        series = pd.Series(data=np.array([1,2,3,4,5]), 
            index=np.array(["01/01/01", "01/02/01", "01/03/01","01/04/01","01/05/01"]))
        days = 6 # days to split longer than the input length
        x_train, x_test, y_train, y_test = split_train_test(series, days, date_format="%m/%d/%y")

def test_strip_records_fail():
    with pytest.raises(Exception):
        # wrong input type
        series = pd.Series(data=np.array(["random", 0,0,0,0]), 
            index=np.array(["01/01/01", "01/02/01", "01/03/01","01/04/01","01/05/01"]))
        output = strip_records(series)


