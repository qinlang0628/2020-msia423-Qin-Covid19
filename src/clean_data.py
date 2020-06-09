import pandas as pd
from config import config
import yaml
import logging
import logging.config
import os

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('clean_data')

def clean_data(config, **kwargs):
    '''convert raw data to clean data
    input:
        config (python file): a config file with the path info
        country_col_origin (str): original column name for country
        country_col_rename (str): column name for country after renaming
        lat_col (str): column name for latitude
        long_col (str): column name for longitude
    output:
        None
    '''
    try:
        logger.info("Cleaning data...")
        # read csv file
        cases = pd.read_csv(config.OUTPUT_PATH, error_bad_lines=False)

        # clean the file and reformat column name
        cases = cases.groupby(kwargs["country_col_origin"]).sum()
        cases = cases.reset_index()
        clean_cases = cases.rename(columns = {kwargs["country_col_origin"]: kwargs["country_col_rename"]})
        clean_cases = clean_cases.drop([kwargs["lat_col"], kwargs["long_col"]], axis=1)

        # save it to csv
        clean_cases.to_csv(config.CLEAN_FILE_PATH, index=False)

    except Exception as ex:
        logger.error(ex)

def main():
    
    # read yml config
    with open(config.PARAM_CONFIG, "r") as f:
        param = yaml.load(f, Loader=yaml.SafeLoader)
    param_py = param["clean_data"]
    
    # clean_data
    clean_data(config, **param_py["clean_data"])

if __name__ == "__main__":
    main()

