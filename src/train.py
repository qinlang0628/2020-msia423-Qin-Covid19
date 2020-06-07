import pandas as pd
import csv
import scipy.optimize as opt
from datetime import datetime
import numpy as np

cases = pd.read_csv('../data/sample/time_series_covid19_confirmed_global.csv',error_bad_lines=False)

dates_col = cases.columns[4:]
series = cases[dates_col].loc[0, :]

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_latest_data(series, days, date_format="%m/%d/%y"):
    '''get latest date of the series, convert date to number
    input:
        series (pandas.core.series.Series): pandas series of a record of a region
        days (int): the amount of days used to make prediction
    output:
        x (np.array): an array of day index
        y (np.array): number of cases on that corresponds with the day index
    '''
    if len(series) < days:
        x = series.index.values
        y = series.values
    else:
        x = series.index.values[-days:]
        y = series.values[-days:]
    
    x = [datetime.strptime(s, date_format) for s in x]
    first_date = x[0]
    x = [(s - first_date).days for s in x]
    return x, y


# define model
model = Sequential()
n_steps = 3
n_features = 1

model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# get x, y
x, y = get_latest_data(series, 10)
x_lstm, y_lstm = split_sequence(y, n_steps)
X = x_lstm
X = X.reshape((X.shape[0], X.shape[1], n_features))

# fit model
model.fit(X, y_lstm, epochs=20, verbose=0)
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)