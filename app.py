import traceback
from flask import render_template, request, redirect, url_for
import logging.config

from flask import Flask
from src.add_cases import Cases
from src.train import exponential_model, lstm_model, logistic_model
from src.evaluate import evaluate

import src.download_data as download_data
import src.clean_data as clean_data
import src.add_cases as add_cases
from src.models import create_db

from flask_sqlalchemy import SQLAlchemy
from config import config
import datetime
import numpy as np
import yaml

from argparse import Namespace



# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('app')

# Initialize the database
db = SQLAlchemy(app)

# define default values for the app
class chart_attribute(object):
    def __init__(self):
        self.countries = []
        self.current_country = "Afghanistan"
        self.display_span = 23
        self.param = None
    

# define default values
class current_model(object):
    def __init__(self, dates, values):
        # attributes
        self.model_type = "none"

        # use span, display span & prediction span
        self.use_span = 10
        self.span = 4

        # default model
        self.model = None
        self.update_input(dates, values)
    
    def update_input(self, dates, values):
        try:
            assert len(dates) == len(values), "Please input same length of dates and data points"
            self.dates = dates
            self.values = values
            self.keys = self.get_keys()

            # derived attributes (for prediction use)
            self.dates_use = self.dates[-self.use_span:]
            self.values_use = self.values[-self.use_span:]
            self.keys_use = self.keys[-self.use_span:]

            # derived attributes
            self.pred_dates = self.get_date_span(span=self.span)
            self.pred_values = [0 for i in range(self.span)]
            self.pred_keys = [self.keys[-1]+i+1 for i in range(self.span)]
            self.model_details = {"model_name": "none", "r2": 0.0, "msle": 0.0}
            self.model_details["show_prediction"] = False

        except Exception as ex:
            logger.error(ex)

        
    def get_keys(self, date_format = "%m/%d/%y"):
        '''get keys corresponds to the dates'''
        try:
            dates = self.dates
            first_day = dates[0]
            first_day = datetime.datetime.strptime(first_day, date_format)
            keys = [(datetime.datetime.strptime(s, date_format) - first_day).days for s in dates]
            return keys
        except Exception as ex:
            logger.error(ex)

    def get_date_span(self, date_format = "%m/%d/%y", span=7):
        ''' input a list a date, return the next X day
        input:
            self.dates (string[]): list of dates
        output:
            self.dates (string[]): list of dates of the next X days
        '''
        try:
            dates = self.dates
            last_day = dates[-1]
            last_day = datetime.datetime.strptime(last_day, date_format)
            next_days = [last_day + datetime.timedelta(days=i+1) for i in range(span)]
            next_days = [x.strftime("%m/%d/%y") for x in next_days]
            return next_days
        except Exception as ex:
            logger.error(ex)

    def evaluate_model(self):
        ''' evaluation the current model performance
        '''
        try:
            x, y = np.array(self.keys_use), np.array(self.values_use)
            x_train = np.array(x[:-self.span])
            y_train = y[:-self.span]
            x_test = np.array(x[-self.span:])
            y_test = y[-self.span:]
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            evaluation = evaluate(y_test, y_pred)
            self.model_details = {
                "r2": evaluation["r2"],"msle": evaluation["msle"]}
        except Exception as ex:
            logger.error(ex)

        # return evaluation


    def update_prediction(self):
        ''' with new curren value and span, update the prediction
        '''
        try:
            if self.model_type == "exp":
                self.model = exponential_model(**cattrs.param["prediction_models"]["exponential_model"])
                
            if self.model_type == "lstm":
                self.model = lstm_model(**cattrs.param["prediction_models"]["lstm_model"])
                
            if self.model_type == "logistic":
                self.model = logistic_model(**cattrs.param["prediction_models"]["logistic_model"])
                
            # write model details
            if self.model_type == "none":
                self.pred_values = [0,0,0,0,0,0,0,0]
                self.model_details["model_name"] = "None"
                self.model_details["r2"] = 0.0
                self.model_details["msle"] = 0.0
                self.model_details["show_prediction"] = False
            else:
                # update prediction values
                x, y, x_test = np.array(self.keys_use), np.array(self.values_use), np.array(self.pred_keys)
                try:
                    self.model.fit(x, y)
                    self.pred_values = list(self.model.predict(x_test))
                except: # if fail to fit or predict, use the last value as prediction
                    self.pred_values = list(np.ones(x_test.shape) * self.values[-1])

                # get last x day performance
                try:
                    self.evaluate_model()
                except: # if evaluation fails, show the failure on the web app
                    self.model_details = {
                    "r2": "Evaluation fails","msle": "Evaluation fails"}
                
                self.model_details["model_name"] = self.model.get_name()
                self.model_details["show_prediction"] = True

        except Exception as ex:
            logger.error(Exception)



@app.route("/change", methods=['POST'])
def change_data():  
    global cmodel
    logger.info(request.form) # do something

    if "model_type" in request.form:
        if request.form["model_type"] == 'reset':
            logger.info("Reset model")
            cmodel.model_type = 'none'
        else:
            cmodel.model_type = request.form['model_type']
            logger.info("Update model: {}".format(cmodel.model_type))
    
    cmodel.update_prediction()

    try:
        if cmodel.model_type == 'none':
            return redirect(url_for('chart'))
        else:
            cmodel.model_details["show_prediction"] = True
            return render_template('chart.html', 
                labels = cmodel.dates, values = cmodel.values, 
                pred_labels = cmodel.pred_dates, pred_values = cmodel.pred_values, 
                keys = cmodel.keys, pred_keys = cmodel.pred_keys, 
                model_details = cmodel.model_details, 
                countries=cattrs.countries, current_country=cattrs.current_country)
    except Exception as ex:
        logger.error(ex)

def load_data(cattrs, db):
    # create a namespace object to pass in values
    args = Namespace()
    args.url = app.config["DATA_URL"]
    args.raw_data = app.config["RAW_DATA"]
    args.clean_file = app.config["CLEAN_FILE"]
    # download latest data from online resource
    download_data.main(args) 
    # clean the dataset
    clean_data.main(args)
    # write to database
    add_cases.main_from_session(db.session)
    

@app.route("/", methods=["POST", "GET"])
def chart():
    global cmodel
    global cattrs
    logger.info(request.form) # do something

    logger.info("Loading Home Page...")

    if "country_name" in request.form:
        try:
            country_name = request.form["country_name"]
            logger.info("change country to {}".format(country_name))
            dates, values = query_data(country_name, cattrs.display_span)
            cattrs.current_country = country_name
            cmodel.update_input(dates, values)
        except Exception as ex:
            logger.error(ex)


    if "load_data" in request.form:
        load_data(cattrs, db)

        # reflect on the page
        country_name = cattrs.current_country
        dates, values = query_data(country_name, cattrs.display_span)
        cmodel.update_input(dates, values)

    try:
        cmodel.model_details = {"model_name": "None", "r2": 0.0}
        cmodel.model_details["show_prediction"] = False
        return render_template('chart.html', 
            labels = cmodel.dates, values = cmodel.values, 
            pred_labels = cmodel.pred_dates, pred_values = cmodel.pred_values, 
            keys = cmodel.keys, pred_keys = cmodel.pred_keys, 
            model_details = cmodel.model_details, 
            countries = cattrs.countries, current_country=cattrs.current_country)
    except Exception as ex:
        logger.error(ex)

def query_data(country, n):
    '''query the lastest n days of a country
    input:
        country (str): country name
        n (int): number of days to retrieve
    '''
    # query data
    cases = db.session.execute("SELECT * from cases").fetchall()
    cases = [x for x in cases if x.country == country]
    # sort by dates
    cases.sort(key=lambda c: datetime.datetime.strptime(c.date, "%m/%d/%y"))
    # update data
    dates = [c.date for c in cases]
    values = [c.confirm_cases for c in cases]
    # trim the data to n data points
    dates = dates[-n:]
    values = values[-n:]
    return dates, values

if __name__ == '__main__':
    # create a record for current model attributes
    cattrs = chart_attribute()
    
    # load parameters to chart attribute
    with open("config/app_config.yml", "r") as f:
        param = yaml.load(f, Loader=yaml.SafeLoader)
        cattrs.param = param

    try:
        # get country names from database
        countries = db.session.execute("SELECT DISTINCT(country) FROM cases").fetchall()
        cattrs.countries = [x[0] for x in countries]
    except:
        load_data(cattrs, db)
        # get country names from database
        countries = db.session.execute("SELECT DISTINCT(country) FROM cases").fetchall()
        cattrs.countries = [x[0] for x in countries]

    # query data from default country
    dates, values = query_data(cattrs.current_country, cattrs.display_span)

    # build a pre-defined model to show on the webapp
    cmodel = current_model(dates, values)
    
    # run the app
    debug=app.config["DEBUG"]
    app.run(port=app.config["PORT"], host=app.config["HOST"], threaded=False, debug=False)
    
    