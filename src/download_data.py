import csv
import requests
import config
import os
import logging
import logging.config

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('upload_data')


if __name__ == "__main__":
    url = os.path.join(config.BASE_URL, config.FILE_NAME)

    # Downloading data from source
    try:
        logger.info("Downloading data from {}".format(url))
        response = requests.get(url)
        logger.info("Download successfully.")
    except Exception as e:
        logger.error("Error occurred when trying to download the file.")
        logger.error(e)

    # write the data into local file
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

