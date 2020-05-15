import os
import sys
import json
from json import JSONDecodeError

import config
import logging
import logging.config
import sqlite3

from models import Tweet, TweetScore
from helpers import get_session

logging.config.fileConfig(config.LOGGING_CONFIG)
logger = logging.getLogger('integrate_data')


def validate(records):
    # Add validation code here
    """Remove the invalid record.
    Args:
        records (`:obj:`list` of :obj:`dict`):  A list of dictionary of key "score" and "tweet_id"
    
    Returns:
        records (`:obj:`list` of :obj:`dict`):  A list of dictionary of key "score" and "tweet_id"
    """
    # row count is as expected -- not required
    # Use a file format with schema evolution like Avro to handle schema changes -- not required

    # required fields are present 
    # specific columns are not null
    # remove records with duplicated tweet_ids
    n1, n2, n3, n4, n5 = 0, 0, 0, 0, 0
    tmp = []
    unique_ids = []

    for record in records:
        try:
            if set(record.keys()) != {'tweet_id', 'score'}: # invalid fields
                n1 += 1
                continue
            if type(record["score"]) != int or type(record["tweet_id"]) != str: # invalid data type
                n2 += 1
                continue
            if record["score"] != record["score"] or record["tweet_id"] == "": # null values
                n3 += 1
                continue
            if record["score"] not in [0, 2, 4]: # invalid values
                n4 += 1
                continue
            if record["tweet_id"] in unique_ids: # duplicates
                n5 += 1
                continue
            unique_ids.append(record["tweet_id"])
            tmp.append(record)
        except:
            logger.error("Validation error: ", record)
        
    # log out data quality issue
    logger.info("{} records are removed due to invalid fields".format(n1))
    logger.info("{} records are removed due to invalid data type".format(n2))
    logger.info("{} records are removed due to null value".format(n3))
    logger.info("{} records are removed due to invalid score value".format(n4))
    logger.info("{} records are removed due to duplication".format(n5))
    
    # ensure ordering
    try:
        records = sorted(tmp, key = lambda x: x['tweet_id']) 
    except:
        logger.info("Sorting fails.")

    logger.info("{} records after validation".format(len(tmp)))
    return records


def persist_scores(session, records):
    # Add code to persist scores into table `tweet_scores` in database here.
    n_records = 0
    for record in records:
        try:
            record_ = TweetScore(tweet_id=record["tweet_id"], score = record["score"])
            session.add(record_) 
            n_records += 1
        except Exception as e:
            logger.error(e)
            continue
    logger.info("{} records persisted into database.".format(n_records))


def read_records(file_location):
    """Read tweet score records from file
    Args:
        file_location (str):  Location of record file to read
    Returns:
        :obj:`list` of tweet score (tweet_id, score) records
    """
    if not file_location:
        raise FileNotFoundError

    logger.debug("Reading records from {}".format(file_location))

    with open(file_location, 'r+') as input_file:
        try:
            output_records = json.load(input_file)
        except JSONDecodeError:
            logger.error("Could not decode JSON")

    logger.info("Retrieved {} records from {}.".format(len(output_records), file_location))

    return output_records


if __name__ == "__main__":
    """Read tweet score records from file, validate them, then persist into tweet_score table.
    """

    file = config.SENTIMENT_RAW_LOCATION

    # Read tweet score records from raw file
    try:
        records = read_records(file)
    except FileNotFoundError:
        logger.error("File not found. Please provide a valid file location to read data.")
        sys.exit(1)

    num_none = [1 if r is None else 0 for r in records]
    logger.warning("%i of %i records recorded as None", sum(num_none), len(records))

    # Validate raw tweet score records we've received from API
    logger.debug("Validating records")
    validated_records = validate(records)

    # Get tweet ids from our database
    session = get_session(engine_string=config.SQLALCHEMY_DATABASE_URI)
    # session.execute('''DELETE FROM tweet_score''')
    try:
        persist_scores(session, validated_records)
        session.commit()
    except sqlite3.IntegrityError as e:
        logger.error("The database already contains the records you are trying to insert. "
                     "Please truncate the table before attempting again.")
        logger.error(e)
        sys.exit(1)
    except Exception as e:
        logger.error(e)
        sys.exit(1)
    finally:
        logger.info("Persisted {} records.".format(len(validated_records)))
        session.close()