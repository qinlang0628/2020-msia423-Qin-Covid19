import csv
import requests
from config import config
import os
import logging
import logging.config

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('upload_data')


def get_response(config):
    '''downloading data from source
    input:
        url (str): a string of the data source
    output:
        response (html): a response
    '''
    try:
        url = os.path.join(config.BASE_URL, config.FILE_NAME)
        logger.info("Downloading data from {}".format(url))
        response = requests.get(url)
        logger.info("Download successfully.")
        logger.info(type(response))
        return response
    except Exception as e:
        logger.error("Error occurred when trying to download the file.")
        logger.error(e)
    

def write_response(response, config):
    ''' write the data into local file
    input:
        response
    output:
        None
    '''
    try:
        logger.info("Saving data to {}".format(config.OUTPUT_PATH))
        with open(config.OUTPUT_PATH, 'w') as f:
            writer = csv.writer(f)
            for line in response.iter_lines():
                writer.writerow(line.decode('utf-8').split(','))
        logger.info("Saved successfully")
    except Exception as e:
        logger.error("Error occurred when trying to save the file.")
        logger.error(e)

def main():
    response = get_response(config)
    write_response(response, config)

if __name__ == "__main__":
    main()

    
    

