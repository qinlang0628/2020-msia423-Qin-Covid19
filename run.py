import src.download_data as download_data
import src.clean_data as clean_data
import src.add_cases as add_cases
import config.config as config
import src.train as train 
from src import score_model
import src.evaluate as evaluate
import src.acquire_data as acquire_data
import src.upload_data as upload_data

from src.prediction_models import exponential_model, param_py as model_params

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="running pipeline")

    # all arguments needed in the pipeline
    parser.add_argument("--start_point", default="acquire", help="available start point: 'acquire', 'download'")
    
    parser.add_argument("--bucket", default=config.BUCKET,help="S3 bucket name")
    parser.add_argument("--url", default=os.path.join(config.BASE_URL, config.FILE_NAME),help="url to raw data")
    parser.add_argument("--file_name", default=config.FILE_NAME, help="file name")
    parser.add_argument("--raw_data", default=config.RAW_DATA,help="path to raw data")
    parser.add_argument("--clean_file", default=config.CLEAN_FILE_PATH, help="path to cleaned data")
    parser.add_argument("--model_type", default="exp",help="available models: exp, log or lstm")
    parser.add_argument("--train", default=config.TRAIN, help="train directory")
    parser.add_argument("--test", default=config.TEST, help="test directory")
    parser.add_argument("--model_dir", default=config.MODEL_DIR, help="model directory")
    parser.add_argument("--pred", default=config.PRED, help="prediction directory")
    
    args = parser.parse_args()

    assert args.start_point in ["download", "acquire"], "Invalid input for start point"

    if args.start_point == "download":
        download_data.main(args)
        upload_data.main(args)
    
    acquire_data.main(args)
    clean_data.main(args)
    train.main(args)
    score_model.main(args)
    evaluate.main(args)

    # add_cases.main(config.SQLALCHEMY_DATABASE_URI)
    