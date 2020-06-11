from config import config
import os

DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "runapp"

# define database URL

SQLALCHEMY_DATABASE_URI = os.environ.get("SQLALCHEMY_DATABASE_URI")
if SQLALCHEMY_DATABASE_URI is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/cases.db'

SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100


# database reated attribtues
DATA_URL = os.path.join(config.BASE_URL, config.FILE_NAME)
RAW_DATA = config.RAW_DATA
CLEAN_FILE = config.CLEAN_FILE_PATH
