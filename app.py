import traceback
from flask import render_template, request, redirect, url_for
import logging.config
# from app.models import Tracks
from flask import Flask
# from src.add_songs import Tracks
from src.prediction_models import exponential_model, lstm_model
from src.evaluate import evaluate
from flask_sqlalchemy import SQLAlchemy
from src import config
import datetime
import numpy as np


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


# @app.route('/')
# def index():
#     """Main view that lists songs in the database.

#     Create view into index page that uses data queried from Track database and
#     inserts it into the msiapp/templates/index.html template.

#     Returns: rendered html template

#     """

#     try:
#         # tracks = db.session.query(Tracks).limit(app.config["MAX_ROWS_SHOW"]).all()
#         logger.debug("Index page accessed")
#         return render_template('chart.html', tracks=tracks)
#     except:
#         traceback.print_exc()
#         logger.warning("Not able to display tracks, error page returned")
#         return render_template('error.html')


# @app.route('/add', methods=['POST'])
# def add_entry():
#     """View that process a POST with new song input

#     :return: redirect to index page
#     """

#     try:
#         track1 = Tracks(artist=request.form['artist'], album=request.form['album'], title=request.form['title'])
#         db.session.add(track1)
#         db.session.commit()
#         logger.info("New song added: %s by %s", request.form['title'], request.form['artist'])
#         return redirect(url_for('index'))
#     except:
#         logger.warning("Not able to display tracks, error page returned")
#         return render_template('error.html')


# define default values
class current_model(object):
    def __init__(self):
        # attributes
        self.model_type = "none"
        self.country_name = "United States"

        # prediction span
        self.span = 3

        # default model
        self.model = None
        
        # input attributes
        self.dates = ["5/8/20", "5/9/20", "5/10/20", "5/11/20", "5/12/20", "5/13/20", "5/14/20", "5/15/20"]
        self.values = [10, 9, 8, 7, 6, 4, 7, 8]
        self.keys = self.get_keys()
        
        
        # derived attributes
        self.pred_dates = self.get_date_span(span=self.span)
        self.pred_values = [0 for i in range(self.span)]
        self.pred_keys = [self.keys[-1]+i+1 for i in range(self.span)]
        self.model_details = {"model_name": "none", "r2": 0.0, "msle": 0.0}
        self.model_details["show_prediction"] = False
        
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

    def get_evaluation(self):
        ''' evaluation the current model performance
        '''
        x, y = np.array(self.keys), np.array(self.values)
        x_train = np.array(x[:-self.span])
        y_train = y[:-self.span]
        x_test = np.array(x[-self.span:])
        y_test = y[-self.span:]
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        evaluation = evaluate(y_test, y_pred)
        return evaluation

    # def update_model(self, model_type):
    #     if model_type == "exp":
    #         self.model_type = "exp"
    #         self.model = exponential_model()
    #         self.model_details["model_name"] = "Exponential Model"
    #     if model_type == "linear":
    #         self.model_type = "linear"


    def update_prediction(self):
        ''' with new curren value and span, update the prediction
        '''
        if self.model_type == "exp":
            # modeling
            x, y, x_test = np.array(self.keys), np.array(self.values), np.array(self.pred_keys)
            self.model = exponential_model()
            self.model.fit(x, y)
            self.pred_values = list(self.model.predict(x_test))
            
            # get last x day performance
            evaluation = self.get_evaluation()
            self.model_details = {"model_name": "Exponential Model", 
                "r2": evaluation["r2"],"msle": evaluation["msle"]}

        if self.model_type == "linear":
            # modeling
            x, y, x_test = np.array(self.keys), np.array(self.values), np.array(self.pred_keys)
            self.model = lstm_model()
            self.model.fit(x, y)
            self.pred_values = list(self.model.predict(x_test))

            # self.pred_values = [10, 11, 12, 13, 14, 15, 16, 17]
            self.model_details = {"model_name": "Linear Model", "r2": 1.0}
        if self.model_type == "lstm":
            self.pred_values = [10, 12, 14, 16, 20, 22, 25, 27]
            self.model_details = {"model_name": "LSTM Model", "r2": 3.0}
        if self.model_type == "none":
            self.pred_values = [0,0,0,0,0,0,0,0]
            self.model_details = {"model_name": "None", "r2": 0.0}
            self.model_details["show_prediction"] = False

cmodel = current_model()

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
    
    if "country_name" in request.form:
        cmodel.country_name = request.form['country_name']
        # cmodel.model_type = 'none'
        logger.info("Update country: {}".format(cmodel.country_name))

    # update prediction
    cmodel.update_prediction()

    try:
        if cmodel.model_type == 'none':
            return redirect(url_for('chart'))
        else:
            # cmodel.dates = ["01-01-2016", "01-02-2016", "01-03-2016", "01-04-2016", "01-05-2016", "01-06-2016", "01-07-2016", "01-08-2016"]
            cmodel.model_details["show_prediction"] = True
            return render_template('chart.html', 
                labels = cmodel.dates, values = cmodel.values, 
                pred_labels = cmodel.pred_dates, pred_values = cmodel.pred_values, 
                keys = cmodel.keys, pred_keys = cmodel.pred_keys, 
                model_details = cmodel.model_details)
    except Exception as ex:
        logger.error(ex)
    

@app.route("/", methods=['POST', 'GET'])
def chart():
    try:
        cmodel.model_details = {"model_name": "None", "r2": 0.0}
        cmodel.model_details["show_prediction"] = False
        return render_template('chart.html', 
            labels = cmodel.dates, values = cmodel.values, 
            pred_labels = cmodel.pred_dates, pred_values = cmodel.pred_values, 
            keys = cmodel.keys, pred_keys = cmodel.pred_keys, 
            model_details = cmodel.model_details)
    except Exception as ex:
        logger.error(ex)


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])