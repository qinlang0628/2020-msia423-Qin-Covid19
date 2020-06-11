import logging
import logging.config
import boto3
from botocore.exceptions import ClientError
import argparse
# import configparser
import os

logging.config.fileConfig("config/logging.conf")


def main(args):
    '''
    input:
        args.raw_data: file to raw csv data
        args.bucket: S3 bucket name
        args.file_name: file name of raw csv data
    output:
        None
    '''
    logger = logging.getLogger('upload_data')
    # read aws credentials
    # parser = configparser.RawConfigParser(allow_no_value=True)
    # parser.read(config.S3_CONFIG)
    # AWS_ACCESS_KEY_ID = parser.get("default", "AWS_ACCESS_KEY_ID")
    # AWS_SECRET_ACCESS_KEY = parser.get("default", "AWS_SECRET_ACCESS_KEY")
    # REGION = parser.get("default", "REGION")
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

    # read filename and bucket
    file_path = args.raw_data
    bucket_name = args.bucket
    file_name = args.file_name

    # connect to s3
    try:
        logger.info("Connecting to AWS S3 ...")
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY) 
        logger.info("Connected successfully.")
    except Exception as ex:
        logger.error("Error occurred when connecting.")
        logger.error(ex)

    # upload to aws
    try:
        logger.info("Uploading {} to Amazon S3 bucket {} ...".format(file_name, bucket_name))
        s3.upload_file(file_path, bucket_name, file_name)
        logger.info('Uploaded Successfully.')
    except Exception as ex:
        logger.error("Error occurred when uploading data to Amazon S3 bucket.")
        logger.error(ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create defined tables in database")
    parser.add_argument("--raw_data", "-d", default=os.path.join("data", "sample", "time_series_covid19_confirmed_global.csv"), 
                        help="Pass in data file path.")
    parser.add_argument("--bucket", "-b", default="nw-langqin-s3", help="Pass in bucket name.")
    parser.add_argument("--file_name", default='time_series_covid19_confirmed_global.csv', help="file name")
    args = parser.parse_args()

    main(args)
