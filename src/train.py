import pandas as pd
import csv
import scipy.optimize as opt
from datetime import datetime
import numpy as np
from src.evaluate import evaluate
from src.prediction_models import exponential_model, logistic_model, lstm_model

import logging
import logging.config
from config import config
import yaml

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('train')

# read yml config
with open(config.PARAM_CONFIG, "r") as f:
    param = yaml.load(f, Loader=yaml.SafeLoader)
param_models = param["prediction_models"]
param_train = param["train"]


# # split a univariate sequence into samples
# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the sequence
#         if end_ix > len(sequence)-1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return np.array(X), np.array(y)

def split_train_test(series, days, date_format="%m/%d/%y"):
    '''
    split training and testing. For a normal dataset, split the last x days data as test.
    Raise error when the effective days is less than x+3.
    input:
        series (pandas.core.series.Series): pandas series of a record of a region
        days (int): the amount of days used to make prediction
    output:
        x (np.array): an array of day index
        y (np.array): number of cases on that corresponds with the day index
    '''
    if len(series) < days+3: 
        raise Exception("Dataset too short to predict")
    else:
        x = series.index.values
        y = series.values
        x = [datetime.strptime(s, date_format) for s in x]
        first_date = x[0]
        x = [(s - first_date).days for s in x]
        
        x_train = np.array(x[:-days])
        y_train = y[:-days]
        x_test = np.array(x[-days:])
        y_test = y[-days:]
    return x_train, x_test, y_train, y_test

def strip_records(series):
    '''strip the leading zero entries in the series
    input:
        series (pd.Series): a series of dates and confirmed cases
    output:
        series (pd.Series): a series of dates and confirmed cases
    '''
    try:
        values = series.values
        index = series.index.values
        for i, (x, y) in enumerate(zip(values, series)):
            if x > 0:
                break
        return series.iloc[i:]
    except Exception as ex:
        logger.error(ex)


def training_pipeline(model_type, **kwargs):
    msle_list = []

    try:
        cases = pd.read_csv(config.CLEAN_FILE_PATH)

        date_format = kwargs["date_format"]
        start_date = kwargs["start_date"]
        end_date = kwargs["end_date"]
        n_test = kwargs["n_test"]

        # filter the desired dates
        dates_col = cases.columns[1:]
        start_date = datetime.strptime(start_date, date_format)
        end_date = datetime.strptime(end_date, date_format)
        dates_col = [s for s in dates_col if datetime.strptime(s, date_format) >=start_date 
                    and datetime.strptime(s, date_format) <= end_date]
        cases = cases[dates_col]

        # define model
        if model_type == "exp":
            model = exponential_model(**param_models["exponential_model"])
        elif model_type == "log":
            model = logistic_model(**param_models["logistic_model"])
        elif model_type == "lstm":
            model = lstm_model(**param_models["lstm_model"])

        for i in range(len(cases)):
            try:
                # extract information
                country = cases.loc[i, :][0]
                series = cases[dates_col].loc[i, :]
                
                # clear leading 0 in the records and split train and test
                series = strip_records(series)
                x_train, x_test, y_train, y_test = split_train_test(series, n_test)
                
                # modeling
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                evaluation = evaluate(y_test, y_pred)
                msle_list.append(evaluation["msle"])
                
            except Exception as e:
                logger.error(e)
        
        return sum(msle_list) / len(msle_list)
    
    except Exception as ex:
        logger.error(ex)

def main():
    exp_result = training_pipeline("exp", **param_train["training_pipeline"])
    log_result = training_pipeline("log", **param_train["training_pipeline"])
    lstm_result = training_pipeline("lstm", **param_train["training_pipeline"])

    with open(config.MODEL_RESULT_PATH, "w") as file:
        file.write('Exponential Model MSLE: {} \n\n'.format(exp_result))
        file.write('Logistic Model MSLE: {} \n\n'.format(log_result))
        file.write('LSTM Model MSLE: {} \n\n'.format(lstm_result))


if __name__ == "__main__":
    main()