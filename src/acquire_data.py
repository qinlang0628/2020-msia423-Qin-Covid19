import os
import logging
import logging.config
import boto3
from botocore.exceptions import ClientError
import argparse


logging.config.fileConfig("config/logging.conf")

def main(args):
    ''' downloading data from s3
    input:
        args.file_name: file name of the data
        args.raw_data: path to raw data
        args.bucket: S3 bucket name
    output:
        None
    '''
    # connect to s3
    logger = logging.getLogger('acquire_data')
    try:
        AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

        logger.info("running 'acquire_data.py' ...")
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY) 
        
        file_name = args.file_name
        file_path = args.raw_data
        bucket_name = args.bucket

        s3.download_file(bucket_name, file_name, file_path)
        logger.info("Finish.")

    except Exception as ex:
        logger.error(ex)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire Data From S3")
    parser.add_argument("--raw_data", default='data/sample/time_series_covid19_confirmed_global.csv', help="path to raw data")
    parser.add_argument("--bucket", default='nw-langqin-s3',help="S3 bucket name")
    parser.add_argument("--file_name", default='time_series_covid19_confirmed_global.csv', help="file name")

    args = parser.parse_args()
    main(args)

