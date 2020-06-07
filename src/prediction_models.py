import scipy.optimize as opt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


import os
import numpy as np
import tensorflow as tf
import random as python_random

os.environ['PYTHONHASHSEED']=str(0)
np.random.seed(123)
python_random.seed(123)
tf.random.set_seed(1234)


class exponential_model(object):
    
    def __init__(self, lower_bound = 0, upper_bound = [np.inf, np.inf]):
        self.popt = None
        self.pcov = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def func(self, x, a, b):
        return a * np.exp(b * x)
    
    def fit(self, x, y):
        ''' fit the model
        input:
            x, y(numpy.array): input arrays
        output:
            None
        '''
        self.popt, self.pcov = opt.curve_fit(self.func, 
                                             x, y,
                                             bounds = (self.lower_bound, self.upper_bound))    
    def predict(self, x):
        '''predict the x array
        input:
            x (numpy.array): input array
        output:
            y (numpy.array): output label array
        '''
        y = self.func(x, *self.popt)
        return y
    
    def get_coeff(self):
        '''get coefficient of the model
        detailed information is here: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        output:
            (dict): a dictionary of popt and pcov
        '''
        return {"popt": self.popt, "pcov": self.pcov}



class lstm_model(object):
    
    def __init__(self, n_steps=10, n_features=1, n_output=1, nodes=50, epoch=50,
                 activation="relu", optimizer='adam', loss='mse'):
        # define params
        self.n_steps = n_steps
        self.n_features = n_features
        self.epoch = epoch
        
        # define model
        self.model = Sequential()
        self.model.add(LSTM(nodes, activation=activation, input_shape=(n_steps, n_features)))
        self.model.add(Dense(n_output))
        self.model.compile(optimizer=optimizer, loss=loss)
    

    def _split_sequence(self, sequence, n_steps):
        '''split a univariate sequence into samples
        input: 
            sequence (np.array): input sequence to be split
            n_steps (int): number of steps to split the sequence
        output:
            X (np.array): features array
            y (np.array): output array
        '''
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
    
    def _calculate_incre(self, array):
        '''get increasing percentage of a numpy array, and fill the null value with 0
        '''
        array_diff = np.diff(array)
        array = array[1:]
        array_perc_diff = array_diff / array
        for i, x in enumerate(array_perc_diff):
            if x == x: # get the first none null value
                break
        array_perc_diff = array_perc_diff[i:]

        # fillna as 0
        where_are_NaNs = np.isnan(array_perc_diff)
        array_perc_diff[where_are_NaNs] = 0
        return array_perc_diff
    
    def _process_incre(self, incres, start_num):
        ''' get the array of licres value and a number to start, calculate the actual number'''
        incres += 1
        for i, x in enumerate(incres):
            incres[i] = start_num * incres[i]
            start_num = incres[i]
        return incres
    
    def fit(self, x, y):
        ''' fit the model
        input:
            x, y(numpy.array): input arrays
        output:
            None
        '''
        incres = self._calculate_incre(y)
        x_lstm, y_lstm = self._split_sequence(incres, self.n_steps)
        x_lstm = x_lstm.reshape((x_lstm.shape[0], x_lstm.shape[1], self.n_features))
        
        self.model.fit(x_lstm, y_lstm, epochs=self.epoch, verbose=0)
        self.lasty = y[-1]
        self.last_incres = y_lstm[-self.n_steps:]
    
    def predict(self, x):
        '''predict the x array
        input:
            x (numpy.array): input array
        output:
            y_pred (numpy.array): output label array
        '''
        y_pred = list(self.last_incres)
        for i in range(x.shape[0]):
            x_input = np.array(y_pred[-self.n_steps:])
            x_input = x_input.reshape((1, self.n_steps, self.n_features))
            y_tmp = self.model.predict(x_input, verbose=0)
            y_pred.append(y_tmp[0])
        y_pred = np.concatenate(y_pred[self.n_steps:])
        
        # post processing
        y_pred = self._process_incre(y_pred, self.lasty)
        return y_pred