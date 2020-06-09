import scipy.optimize as opt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


import os
import numpy as np
import tensorflow as tf
import random as python_random

from config import config
import yaml

import logging
import logging.config
import os

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('prediction_models')

# read yml config
with open(config.PARAM_CONFIG, "r") as f:
    param = yaml.load(f, Loader=yaml.SafeLoader)
param_py = param["prediction_models"]


os.environ['PYTHONHASHSEED']=str(param_py["python_hash_seed"])
np.random.seed(param_py["numpy_seed"])
python_random.seed(param_py["python_random_seed"])
tf.random.set_seed(param_py["tf_random_seed"])


class exponential_model(object):
    '''exponential growth model'''
    
    def __init__(self, **kwargs):
        ''' initiate an exponential model
        input:
            lower bound (float): lower bound for estimation
            upper bound ([float, float]): upper bound for estimation
        '''
        try:
            self.popt = None
            self.pcov = None
            self.lower_bound = kwargs["lower_bound"]
            self.upper_bound = kwargs["upper_bound"]
            
            # replace "inf" values in upper bound as np.inf
            self.upper_bound = [np.inf if x == "inf" else x for x in self.upper_bound ]
        except Exception as ex:
            logger.error(ex)

    def func(self, x, a, b):
        '''exponential fitting function'''
        return a * np.exp(b * x)
    
    def fit(self, x, y):
        ''' fit the model
        input:
            x, y(numpy.array): input arrays
        output:
            None
        '''
        try:
            self.popt, self.pcov = opt.curve_fit(self.func, x, y,
                                             bounds = (self.lower_bound, self.upper_bound))
        except Exception as ex:
            logger.error(ex)    
        
    def predict(self, x):
        '''predict the x array
        input:
            x (numpy.array): input array
        output:
            y (numpy.array): output label array
        '''
        try:
            y = self.func(x, *self.popt)
            return y
        except Exception as ex:
            logger.error(ex)
    
    def get_coeff(self):
        '''get coefficient of the model
        detailed information is here: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        output:
            (dict): a dictionary of popt and pcov
        '''
        try:
            return {"popt": self.popt, "pcov": self.pcov}
        except Exception as ex:
            logger.error(ex)
    
    def get_name(self):
        return "Exponential Model"



class lstm_model(object):
    '''long short term memory model'''
    
    def __init__(self, **kwargs):
        ''' initiate an lstm model
        input:
            n_steps: n data points for prediction
            n_features: number of featuers to use
            n_output: number of output values
            nodes: number of nodes in the network
            epoch: maximum epochs
            activation: activation function in the network
            optimizer: optimizer in the network
            loss: loss funciton in the network
        '''
        try:
            # define params
            self.n_steps = kwargs["n_steps"]
            self.n_features = kwargs["n_features"]
            self.epoch = kwargs["epoch"]
            self.n_output = kwargs["n_output"]
            self.activation = kwargs["activation"]
            self.optimizer = kwargs["optimizer"]
            self.loss = kwargs["loss"]
            self.nodes = kwargs["nodes"]
            
            # define model
            self.model = Sequential()
            self.model.add(LSTM(self.nodes, activation=self.activation, input_shape=(self.n_steps, self.n_features)))
            self.model.add(Dense(self.n_output))
            self.model.compile(optimizer=self.optimizer, loss=self.loss)
        except Exception as ex:
            logger.error(ex)
    
    def _split_sequence(self, sequence, n_steps):
        '''split a univariate sequence into samples
        input: 
            sequence (np.array): input sequence to be split
            n_steps (int): number of steps to split the sequence
        output:
            X (np.array): features array
            y (np.array): output array
        '''
        try:
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
        except Exception as ex:
            logger.error(ex)
    
    def _calculate_incre(self, array):
        '''get increasing percentage of a numpy array, and fill the null value with 0
        '''
        try:
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
        except Exception as ex:
            logger.error(ex)
    
    def _process_incre(self, incres, start_num):
        ''' get the array of licres value and a number to start, calculate the actual number'''
        try:
            incres += 1
            for i, x in enumerate(incres):
                incres[i] = start_num * incres[i]
                start_num = incres[i]
            return incres
        except Exception as ex:
            logger.error(ex)
    
    def fit(self, x, y):
        ''' fit the model
        input:
            x, y(numpy.array): input arrays
        output:
            None
        '''
        try:
            incres = self._calculate_incre(y)
            x_lstm, y_lstm = self._split_sequence(incres, self.n_steps)
            x_lstm = x_lstm.reshape((x_lstm.shape[0], x_lstm.shape[1], self.n_features))
            self.model.fit(x_lstm, y_lstm, epochs=self.epoch, verbose=0)
            self.lasty = y[-1]
            self.last_incres = incres[-self.n_steps:]
        except Exception as ex:
            logger.error(ex)

    
    def predict(self, x):
        '''predict the x array
        input:
            x (numpy.array): input array
        output:
            y_pred (numpy.array): output label array
        '''
        try:
            y_pred = list(self.last_incres)
            for i in range(x.shape[0]):
                x_input = self.last_incres
                x_input = x_input.reshape((1, self.n_steps, self.n_features))
                
                y_tmp = self.model.predict(x_input, verbose=0)
                y_pred.append(y_tmp[0])
            y_pred = np.concatenate(y_pred[self.n_steps:])
            
            # post processing
            y_pred = self._process_incre(y_pred, self.lasty)
            return y_pred
        except Exception as ex:
            logger.error(ex)
    
    def get_name(self):
        return "LSTM Model"


class logistic_model(object):
    '''
    reference: https://towardsdatascience.com/modeling-logistic-growth-1367dc971de2
    '''
    
    def __init__(self, **kwargs):
        ''' initialize the logistic growth model
        input:
            lower bound (float): lower bound for estimation
            upper bound ([float, float]): upper bound for estimation
            p0 ([int, int, int]): initial guess of the parameters
            maxfev (int): maximum iterations to optimize
        '''
        try:
            self.popt = None
            self.pcov = None
            self.lower_bound = kwargs["lower_bound"]
            self.upper_bound = kwargs["upper_bound"]
            self.p0 = kwargs["p0"]
            self.maxfev = kwargs["maxfev"]

            # replace "inf" values in upper bound as np.inf
            self.upper_bound = [np.inf if x == "inf" else x for x in self.upper_bound ]
        except Exception as ex:
            logger.error(ex)
        
    def func(self, x, a, b, c):
        '''logistic growth function'''
        return c / (1 + a * np.exp(- b * x))
    
    def fit(self, x, y):
        ''' fit the model
        input:
            x, y(numpy.array): input arrays
        output:
            None
        '''
        try:
            self.popt, self.pcov = opt.curve_fit(self.func, 
                                             x, y,
                                             bounds = (self.lower_bound, self.upper_bound),
                                             p0=self.p0, maxfev=self.maxfev)   
        except Exception as ex:
            logger.error(ex)

    def predict(self, x):
        '''predict the x array
        input:
            x (numpy.array): input array
        output:
            y (numpy.array): output label array
        '''
        try:
            y = self.func(x, *self.popt)
            return y
        except Exception as ex:
            logger.error(ex)
    
    def get_coeff(self):
        '''get coefficient of the model
        detailed information is here: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        output:
            (dict): a dictionary of popt and pcov
        '''
        try:
            return {"popt": self.popt, "pcov": self.pcov}
        except Exception as ex:
            logger.error(ex)
    
    def get_name(self):
        return "Logistic Model"

if __name__ == "__main__":
    exponential_model(**param_py["exponential_model"])
    lstm_model(**param_py["lstm_model"])
    logistic_model(**param_py["logistic_model"])