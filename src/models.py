import os
import sys
import logging
import logging.config

from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from config import config
from src.helpers import create_connection, get_session
import argparse
import configparser

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('database_models')

Base = declarative_base()

class Cases(Base):
    """ Defines the data model for the table `cases`. """
    __tablename__ = 'cases'
    id = Column(String(100), primary_key=True, unique=True, nullable=False)
    country = Column(String(100), unique=False, nullable=False)
    date = Column(String(100), nullable=False)
    confirm_cases = Column(Integer, nullable=True)

    def __repr__(self):
        cases_repr = "<Cases(country='%s', date='%s', confirm_cases='%s')>"
        return cases_repr % (self.country, self.date, self.confirm_cases)


def truncate_cases(session):
    """Deletes cases table if rerunning and run into unique key error."""
    session.execute('''DELETE FROM cases''')


def create_db(engine=None, engine_string=None):
    """Creates a database with the data models inherited from `Base` (Cases).
    Args:
        engine (:py:class:`sqlalchemy.engine.Engine`, default None): SQLAlchemy connection engine.
            If None, `engine_string` must be provided.
        engine_string (`str`, default None): String defining SQLAlchemy connection URI in the form of
            `dialect+driver://username:password@host:port/database`. If None, `engine` must be provided.
    Returns:
        None
    """
    if engine is None and engine_string is None:
        return ValueError("`engine` or `engine_string` must be provided")
    elif engine is None:
        engine = create_connection(engine_string=engine_string)

    Base.metadata.create_all(engine)

def add_case(engine_string, id, country, date, confirm_cases):
    """Seeds an existing database with additional cases.
    input:
        engine_string (str): database engine string
        id (int): id of the case
        country (str): country name
        date (str): date
        confirm_cases (int): confirmed cases
    output:None
    """
    engine = sqlalchemy.create_engine(engine_string)

    Session = sessionmaker(bind=engine)
    session = Session()

    case = Cases(country=country, date=date, confirm_cases=confirm_cases)
    session.add(case)
    session.commit()
    logger.info("%s at %s of %s, added to database", country, date, confirm_cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create defined tables in database")
    parser.add_argument("--truncate", "-t", default=False, action="store_true",
                        help="If given, delete current records from cases table before create_all "
                             "so that table can be recreated without unique id issues ")
    parser.add_argument("--rds", "-r", default=False, action="store_true",
                        help="If true, add database to AWS RDS "
                             "otherwise the database would not be created to RDS ")
    args = parser.parse_args()

    # by default, the engine is local
    engine_string = config.SQLALCHEMY_DATABASE_URI
    if args.rds:
        logger.info("Configuring AWS RDS url ...")
        parser = configparser.RawConfigParser(allow_no_value=True)
        parser.read(config.RDS_CONFIG)
        conn_type = "mysql+pymysql"
        user = parser.get("default", "MYSQL_USER")
        password = parser.get("default", "MYSQL_PASSWORD")
        host = parser.get("default", "MYSQL_HOST")
        port = parser.get("default", "MYSQL_PORT")
        database = parser.get("default", "DATABASE_NAME")
        engine_string = engine_string = "{}://{}:{}@{}:{}/{}".format(conn_type, user, password, host, port, database)
        logger.info("Configured Successfully.")

    # If "truncate" is given as an argument (i.e. python models.py --truncate), then empty the cases table)
    if args.truncate:
        session = get_session(engine_string=engine_string)
        try:
            logger.info("Attempting to truncate cases table ...")
            truncate_cases(session)
            session.commit()
            logger.info("Truncated Successfully.")
        except Exception as e:
            logger.error("Error occurred while attempting to truncate cases table.")
            logger.error(e)
        finally:
            session.close()
    
    # create database 
    try:
        logger.info("Creating database in the engine ...")
        create_db(engine_string=engine_string)
        logger.info("Created Successfully.")
    except Exception as e:
        logger.error("Error occurred while creating database")
        logger.error(e)

    
                


