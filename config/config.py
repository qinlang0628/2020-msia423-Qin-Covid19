from os import path
import os

# Getting the parent directory of this file. That will function as the project home.
PROJECT_HOME = path.dirname(path.dirname(path.abspath(__file__)))

# App config
APP_NAME = "runapp"
DEBUG = True

# Logging
LOGGING_CONFIG = path.join(PROJECT_HOME, 'config', 'logging.conf')

# AWS S3 & RDS
S3_CONFIG = path.join(PROJECT_HOME, 'config', 'aws_s3.conf')
RDS_CONFIG = path.join(PROJECT_HOME, 'config', 'aws_rds.conf')

# Database connection config
DATABASE_PATH = path.join(PROJECT_HOME, 'data', 'sample','cases.db')
SQLALCHEMY_DATABASE_URI = 'sqlite:////{}'.format(DATABASE_PATH)
SQLALCHEMY_TRACK_MODIFICATIONS = True
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed

# Download data url 
BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series'
FILE_NAME = 'time_series_covid19_confirmed_global.csv'
CLEAN_FILE_NAME = 'clean_confirmed_global.csv'
OUTPUT_PATH = os.path.join(PROJECT_HOME, 'data', 'sample', FILE_NAME)
CLEAN_FILE_PATH = os.path.join(PROJECT_HOME, 'data', 'sample', CLEAN_FILE_NAME)
MODEL_RESULT_PATH = os.path.join(PROJECT_HOME, 'model', 'evaluation.txt')

# param config
PARAM_CONFIG = os.path.join(PROJECT_HOME, 'config', 'app_config.yml')