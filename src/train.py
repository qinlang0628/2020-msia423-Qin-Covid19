import pandas as pd
import csv
import scipy.optimize as opt
from datetime import datetime
import numpy as np

import logging
import logging.config

import yaml
import os
import dill as pickle
import argparse

from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf
import random as python_random


logging.config.fileConfig("config/logging.conf")



# read yml config
with open("config/app_config.yml", "r") as f:
    param = yaml.load(f, Loader=yaml.SafeLoader)
param_train = param["train"]
param_models = param["prediction_models"]


os.environ['PYTHONHASHSEED']=str(param_models["python_hash_seed"])
np.random.seed(param_models["numpy_seed"])
python_random.seed(param_models["python_random_seed"])
tf.random.set_seed(param_models["tf_random_seed"])


class exponential_model(object):
    '''exponential growth model'''
    
    def __init__(self, **kwargs):
        ''' initiate an exponential model
        input:
            lower bound (float): lower bound for estimation
            upper bound ([float, float]): upper bound for estimation
        '''
        self.logger = logging.getLogger('prediction_models.exp')
        try:
            self.popt = None
            self.pcov = None
            self.lower_bound = kwargs["lower_bound"]
            self.upper_bound = kwargs["upper_bound"]
            assert len(self.upper_bound)==2, "Wrong Input Length"
            
            # replace "inf" values in upper bound as np.inf
            self.upper_bound = [np.inf if x == "inf" else x for x in self.upper_bound ]
        except Exception as ex:
            self.logger.error(ex)
            raise

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
            assert x.shape==y.shape, "Wrong Input Shape"
            self.popt, self.pcov = opt.curve_fit(self.func, x, y,
                                             bounds = (self.lower_bound, self.upper_bound))
        except Exception as ex:
            self.logger.error(ex)  
            raise Exception("Exponential model does not fit: ", ex)
        
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
            self.logger.error(ex)
            raise
    
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
            self.logger.error(ex)
            raise
    
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
        self.logger = logging.getLogger('prediction_models.lstm')
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
            self.logger.error(ex)
            raise
    
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
            self.logger.error(ex)
            raise
    
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
            self.logger.error(ex)
            raise
    
    def _process_incre(self, incres, start_num):
        ''' get the array of licres value and a number to start, calculate the actual number'''
        try:
            incres += 1
            for i, x in enumerate(incres):
                incres[i] = start_num * incres[i]
                start_num = incres[i]
            return incres
        except Exception as ex:
            self.logger.error(ex)
            raise
    
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
            self.logger.error(ex)
            raise Exception("LSTM model does not fit.")

    
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
            self.logger.error(ex)
            raise
    
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
        self.logger = logging.getLogger('prediction_models.log')
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
            self.logger.error(ex)
            raise
        
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
            self.logger.error(ex)
            raise Exception("Logistic model does not fit.")

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
            self.logger.error(ex)
            raise
    
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
            self.logger.error(ex)
    
    def get_name(self):
        return "Logistic Model"


# utility functions
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
    logger = logging.getLogger('train')
    logger.debug("split train test...")
    try:
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
    except Exception as ex:
        logger.error(ex)
        raise

def strip_records(series):
    '''strip the leading zero entries in the series
    input:
        series (pd.Series): a series of dates and confirmed cases
    output:
        series (pd.Series): a series of dates and confirmed cases
    '''
    logger = logging.getLogger('train')
    logger.debug("strip records...")
    try:
        values = series.values
        index = series.index.values
        for i, (x, y) in enumerate(zip(values, series)):
            if x > 0:
                break
        return series.iloc[i:]
    except Exception as ex:
        logger.error(ex)
        raise

def save_model(model, output_path):
    ''' save model
    input:
        model (object):               model object
        model_dir(str):               diretory to put the model
    output:
        None
    '''
    try:
        with open(output_path, 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        logger.error(ex)
        


def training_pipeline(model_type, input, model_dir, output_train=None, output_test=None, save_data=True, **kwargs):
    '''training pipeline for a model type
    input:
        model_type (str): the valid input includes "exp", "log", "lstm"
        country_col (str): country col
        date_format (str): format to parse the date
        start_date (str): the start date, following the date format
        end_date (str): the end date, following the date format
        n_test (int): number of test data point
    output:
        avg_msle (float): average of msle
    '''
    msle_list = []
    logger = logging.getLogger('train')
    logger.debug("training pipeline...")

    try:
        cases = pd.read_csv(input)
        country_col = kwargs["country_col"]
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
        cases = cases[[country_col] + dates_col]

        for i in range(len(cases)):
            try:
                # extract information
                country = cases.loc[i, :][0]
                series = cases[dates_col].loc[i, :]
                
                # define model
                if model_type == "exp":
                    model = exponential_model(**param_models["exponential_model"])
                elif model_type == "log":
                    model = logistic_model(**param_models["logistic_model"])
                elif model_type == "lstm":
                    model = lstm_model(**param_models["lstm_model"])
                
                # clear leading 0 in the records and split train and test
                series = strip_records(series)
                x_train, x_test, y_train, y_test = split_train_test(series, n_test)

                # save train and test data to folder
                if save_data:
                    try:
                        train_data = pd.DataFrame({"x": x_train, "y": y_train})
                        train_data.to_csv(os.path.join(output_train, "{}.csv".format(country)), index=False)
                        test_data = pd.DataFrame({"x": x_test, "y": y_test})
                        test_data.to_csv(os.path.join(output_test, "{}.csv".format(country)), index=False)
                    except Exception as ex:
                        logger.error("Saving train test data fails: ", ex)
                
                # fitting & saving
                try:
                    model.fit(x_train, y_train)
                    save_model(model, os.path.join(model_dir, "{}.pkl".format(country)))
                except:
                    logger.error("Model not saved")
            
            except Exception as e:
                logger.error(e)
    
    except Exception as ex:
        logger.error(ex)

def main(args):
    '''
    input:
        args (dictionary): related arguments
    '''
    logger = logging.getLogger('train')
    logger.info("Running 'train.py'...")
    try:
        if not os.path.exists(args.train):
            os.mkdir(args.train)
        
        if not os.path.exists(args.test):
            os.mkdir(args.test)

        model_type_dir = os.path.join(args.model_dir, args.model_type)
        if not os.path.exists(model_type_dir):
            os.mkdir(model_type_dir)
        
        # if saving directory is not define, do not save data
        if not args.train or not args.test:
            save_data = False
        else:
            save_data = True
        
        training_pipeline(args.model_type, 
            input = args.clean_file, 
            model_dir = model_type_dir, 
            output_train = args.train, 
            output_test = args.test, 
            save_data = save_data, 
            **param_train["training_pipeline"])

        logger.info("Finish")
        
    except Exception as ex:
        logger.error(ex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and test")
    parser.add_argument("--model_type", default="exp",help="model type: exp, log or lstm")
    parser.add_argument("--clean_file", default="data/sample/clean_confirmed_global.csv",help="input features path")
    parser.add_argument("--train", default="data/sample/training_pipeline/train", help="output file for train")
    parser.add_argument("--test", default="data/sample/training_pipeline/test", help="output file for train")
    parser.add_argument("--model_dir", "-dir", default="model", help="output folder for model")
    args = parser.parse_args()

    main(args)