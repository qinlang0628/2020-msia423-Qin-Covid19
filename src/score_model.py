import pandas as pd
import os
import yaml
import argparse
import dill as pickle
import logging
import logging.config
import argparse
import numpy as np

logging.config.fileConfig("config/logging.conf")


def compute_score(model, test):
    '''make prediction by the model
    input:
        model_name (str):             model name
        test (pandas.DataFrame):      testing data
    output:
        ypred (pandas.DataFrame):     prediction result
    '''
    logger = logging.getLogger('score_model')
    logger.debug("compute score...")
    try:
        # make prediction
        x_test = test["x"].to_numpy()
        y_pred = model.predict(x_test)
        # convert to dataframe
        ypred = pd.DataFrame({"y_pred": y_pred})
        return ypred
    except Exception as ex:
        logger.error(ex)
        raise

def main(args):
    logger = logging.getLogger('score_model')
    logger.info("Running 'score_model.py'...")
    
    if args.model_type not in ["exp", "lstm", "log"]:
        logger.error("Invalid type: ", args.model_type)
        raise Exception("Invalid model type")

    try:
        # define output dir
        output_dir = args.pred + "_" + args.model_type
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # load all countries
        countries = os.listdir(args.test)
        countries = [x.replace(".csv", "") for x in countries]

        # iterate each country
        for country in countries:
            logger.debug("Computing score for {}".format(country))
            # load test data
            test = pd.read_csv(os.path.join(args.test, "{}.csv".format(country)))
            # load model
            try:
                with open(os.path.join(args.model_dir, args.model_type, "{}.pkl".format(country)), 'rb') as file:
                    model = pickle.load(file)
                prediction = compute_score(model, test)
                prediction.to_csv(os.path.join(output_dir, "{}.csv".format(country)), index=False)
            except Exception as ex:
                logger.error("Fail to predict {}: {}".format(country, ex))
        
        logger.info("Finish.")
    except Exception as ex:
        logger.error(ex)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split data into train and test")
    parser.add_argument("--test", default="data/sample/training_pipeline/test", help="path to the test file")
    parser.add_argument("--model_type", default="exp", help="model type: exp, log or lstm")
    parser.add_argument("--model_dir", default="model", help="path to the model")
    parser.add_argument("--pred", "-p", default="data/sample/training_pipeline/pred", help="path to the prediction output")
    args = parser.parse_args()

    main(args)
    