import csv
import requests
import os
import logging
import logging.config
import argparse

logging.config.fileConfig("config/logging.conf")


def get_response(url):
    '''downloading data from source
    input:
        url (str): a string of the data source
    output:
        response (html): a response
    '''
    logger = logging.getLogger('download_data')
    try:
        response = requests.get(url)
        return response
    except Exception as e:
        logger.error(e)
    

def write_response(response, raw_data):
    ''' write the data into local file
    input:
        response (requests.models.Response): response from the url
    output:
        None
    '''
    logger = logging.getLogger('download_data')
    try:
        logger.debug("Saving data to {}".format(raw_data))
        with open(raw_data, 'w') as f:
            writer = csv.writer(f)
            for line in response.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))
    except Exception as e:
        logger.error(e)

def main(args):
    '''
    input:
        args.url: Url to download the data
        args.raw_data: path to raw csv data
    '''
    logger = logging.getLogger('download_data')
    logger.info("Downloading...")
    try:
        url = args.url
        raw_data = args.raw_data
        response = get_response(url)
        write_response(response, raw_data)
        logger.info("Finish.")
    except Exception as ex:
        logger.error("Fail to download: ", ex)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="running pipeline")

    # all arguments needed in the download data
    parser.add_argument("--url", default="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",help="path to raw data")
    parser.add_argument("--raw_data", default="data/sample/time_series_covid19_confirmed_global.csv",help="path to raw data")
    
    args = parser.parse_args()
    main(args)

    
    

