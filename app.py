import traceback
from flask import render_template, request, redirect, url_for
import logging.config
# from app.models import Tracks
from flask import Flask
from src.add_songs import Tracks
from flask_sqlalchemy import SQLAlchemy
from src import config


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

@app.route("/change", methods=['POST'])
def change_data():
    logger.info(request.form) # do something
    dates = ["01-01-2016", "01-02-2016", "01-03-2016", "01-04-2016", "01-05-2016", "01-06-2016", "01-07-2016", "01-08-2016"]
    keys = [1, 2, 3, 4, 5, 6, 7, 8]
    values = [10, 9, 8, 7, 6, 4, 7, 8]
    pred_dates = ["01-09-2016", "01-10-2016", "01-11-2016", "01-12-2016", "01-13-2016", "01-14-2016", "01-15-2016", "01-16-2016"]
    pred_keys = [9, 10, 11, 12, 13, 14, 15, 16]

    model_details = {"model_name": "None", "model_accuracy": 0.0}

    if request.form['model_type'] == 'exp':
        logger.info("Get exponential model")
        pred_values = [10, 11, 12, 13, 16, 19, 20, 29]
        model_details = {"model_name": "Exponential Model", "model_accuracy": 2.0}

    if request.form['model_type'] == 'linear':
        logger.info("Get linear model")
        pred_values = [10, 11, 12, 13, 14, 15, 16, 17]
        model_details = {"model_name": "Linear Model", "model_accuracy": 1.0}

    if request.form['model_type'] == 'lstm':
        logger.info("Get lstm model")
        pred_values = [10, 12, 14, 16, 20, 22, 25, 27]
        model_details = {"model_name": "LSTM Model", "model_accuracy": 3.0}

    try:
        

        legend = 'Monthly Data'
        dates = ["01-01-2016", "01-02-2016", "01-03-2016", "01-04-2016", "01-05-2016", "01-06-2016", "01-07-2016", "01-08-2016"]
        return render_template('chart.html', 
            labels = dates, values = values, 
            pred_labels = pred_dates, pred_values = pred_values, 
            keys = keys, pred_keys = pred_keys, 
            model_details = model_details)
    except Exception as ex:
        logger.error(ex)
    

@app.route("/")
def chart():
    try:
        legend = 'Monthly Data'
        dates = ["01-01-2016", "01-02-2016", "01-03-2016", "01-04-2016", "01-05-2016", "01-06-2016", "01-07-2016", "01-08-2016"]
        values = [10, 9, 8, 7, 6, 4, 7, 8]
        pred_dates = ["01-09-2016", "01-10-2016", "01-11-2016", "01-12-2016", "01-13-2016", "01-14-2016", "01-15-2016", "01-16-2016"]
        pred_values = [10, 9, 8, 7, 6, 4, 7, 8]
        model_details = {"model_name": "None", "model_accuracy": 0.0}
        keys = [1, 2, 3, 4, 5, 6, 7, 8]
        pred_keys = [9, 10, 11, 12, 13, 14, 15, 16]
        return render_template('chart.html', 
            labels = dates, values = values, 
            pred_labels = pred_dates, pred_values = pred_values, 
            keys = keys, pred_keys = pred_keys, 
            model_details = model_details)
    except Exception as ex:
        logger.error(ex)


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])