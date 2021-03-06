# from app import db
# from app.models import Track
import argparse
import logging.config
import logging
import yaml
import os
import pandas as pd
import os

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.orm import sessionmaker
from src.models import get_session, truncate_cases

from config import config

import logging
import logging.config
import os

logging.config.fileConfig("config/logging.conf")

Base = declarative_base()

class Cases(Base):
    logger = logging.getLogger('add_cases')
    """ Defines the data model for the table `cases`. """
    __tablename__ = 'cases'
    id = Column(String(100), primary_key=True, unique=True, nullable=False)
    country = Column(String(100), unique=False, nullable=False)
    date = Column(String(100), nullable=False)
    confirm_cases = Column(Integer, nullable=True)

    def __repr__(self):
        cases_repr = "<Cases(country='%s', date='%s', confirm_cases='%s')>"
        return cases_repr % (self.country, self.date, self.confirm_cases)



# add cases
def add_cases(session, config, **kwargs):
    ''' Add cases from csv to database
    input:
        session (sqlalchemy.orm.session.Session): a SQL database connection session
        config (module): a python config file
    output:
        None
    '''
    logger = logging.getLogger('add_cases')
    try:
        logger.info("Add cases to database...")
        # read cases from csv
        cases = pd.read_csv(config.CLEAN_FILE_PATH)
        # extract country & date columns
        country_col = kwargs["country_col"]
        date_cols = [x for x in cases.columns if x != kwargs["country_col"]]
        
        # construct a Cases object and save to database for each country, each day
        id = 0
        for index, row in cases.iterrows():
            country = row[country_col]
            for date in date_cols:
                if row[date] == row[date]:
                    try:
                        case = Cases(id=str(id), country=country, date=date, confirm_cases=row[date])
                        session.add(case)
                    except Exception as Ex:
                        logger.error(ex)
                    id += 1
            logger.debug("%s added to database", country)
        # commit the change
        session.commit()
        
    except Exception as ex:
        logger.error(ex)

def main_from_session(session):
    logger = logging.getLogger('add_cases')
    try:
        # read yml config
        logger.info("Adding cases ...")
        with open(config.PARAM_CONFIG, "r") as f:
            param = yaml.load(f, Loader=yaml.SafeLoader)
        param_py = param["add_cases"]
        
        truncate_cases(session)
        add_cases(session, config, **param_py["add_cases"])
    except Exception as ex:
        logger.error(ex)
        raise Exception

def main(engine_string):
    # construct a database connection
    logger = logging.getLogger('add_cases')
    try:
        session = get_session(engine_string=engine_string)
    except Exception as ex:
        logger.error(ex)
    
    # add cases if the database session is created
    if session:
        try:
            main_from_session(session)
        except Exception as ex:
            logger.error(ex)
        finally:
            # close the database connection
            session.close()

if __name__ == "__main__":
    engine_string = "sqlite:///data/cases.db"
    main(engine_string)