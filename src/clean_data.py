import pandas as pd
import yaml
import logging
import logging.config
import os
import argparse

logging.config.fileConfig("config/logging.conf")


def clean_data(raw_data, **kwargs):
    '''convert raw data to clean data
    input:
        raw_data (pd.DataFrame): raw data
        country_col_origin (str): original column name for country
        country_col_rename (str): column name for country after renaming
        lat_col (str): column name for latitude
        long_col (str): column name for longitude
    output:
        clean_cases (pd.DataFrame): dataframe of cleaned data
    '''
    logger = logging.getLogger('clean_data')
    try:
        logger.debug("Cleaning data...")
        
        # clean the file and reformat column name
        cases = raw_data.groupby(kwargs["country_col_origin"]).sum()
        cases = cases.reset_index()
        clean_cases = cases.rename(columns = {kwargs["country_col_origin"]: kwargs["country_col_rename"]})
        clean_cases = clean_cases.drop([kwargs["lat_col"], kwargs["long_col"]], axis=1)
        return clean_cases
        
    except Exception as ex:
        logger.error(ex)
        raise

def main(args):
    logger = logging.getLogger('clean_data')
    logger.info("Running 'clean_data.py'...")
    try:
        raw_data = args.raw_data
        clean_file = args.clean_file

        # read csv file
        cases = pd.read_csv(raw_data, error_bad_lines=False)

        # read yml config
        with open("config/app_config.yml", "r") as f:
            param = yaml.load(f, Loader=yaml.SafeLoader)
        param_py = param["clean_data"]
        
        # clean_data
        clean_cases = clean_data(cases, **param_py["clean_data"])

        # save it to csv
        clean_cases.to_csv(clean_file, index=False)

        logger.info("Finish.")
    except Exception as ex:
        logger.error(ex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and test")
    parser.add_argument("--raw_data", default="data/sample/time_series_covid19_confirmed_global.csv",help="path to raw data")
    parser.add_argument("--clean_file", default="data/sample/clean_confirmed_global.csv", help="cleaned data path")
    
    args = parser.parse_args()

    main(args)

