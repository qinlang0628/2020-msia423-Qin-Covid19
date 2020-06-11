# evaluation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
import os, argparse

import logging
import logging.config
import pandas as pd
import argparse

logging.config.fileConfig("config/logging.conf")



def evaluate(y_test, y_pred):
    ''' evaluation of model result
    input:
        y_test (numpy.array): array of actual data
        y_pred (numpy.array): array of predicted data
    output:
        ({"r2": float, "msle": float}): dictionary of metrics
    '''
    logger = logging.getLogger('evaluate')
    try:
        r2 = r2_score(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)
        return {'r2': r2, 'msle': msle}
    except Exception as ex:
        logger.error(ex)
        raise


def main(args):
    logger = logging.getLogger('evaluate')
    logger.info("Running 'evaluate.py'...")
    
    if args.model_type not in ["exp", "lstm", "log"]:
        logger.error("Invalid type: ", args.model_type)
        raise Exception("Invalid model type")

    try:
        # load all countries
        countries = os.listdir(args.test)
        countries = [x.replace(".csv", "") for x in countries]

        # iterate each country
        result_list = []
        r2_list = []
        msle_list = []
        for country in countries:
            logger.debug("Computing score for {}".format(country))
            try:
                # load test data
                y_test = pd.read_csv(os.path.join(args.test, "{}.csv".format(country)))["y"].to_numpy()
                # load prediction data
                y_pred = pd.read_csv(os.path.join(args.pred + "_" + args.model_type, "{}.csv".format(country)))["y_pred"].to_numpy()
                # evaluate
                evaluation = evaluate(y_test, y_pred)
                result_list.append([country, evaluation])
                r2_list.append(evaluation["r2"])
                msle_list.append(evaluation["msle"])
            except Exception as ex:
                logger.error("Fail to evaluate {}: {}".format(country, ex))
        
        # writing result to file
        logger.debug("Writing result to file...")
        try:
            output_path = os.path.join(args.model_dir, "evaluation_{}.txt".format(args.model_type))
            with open(output_path, "w") as file:
                for result in result_list:
                    country = result[0]
                    r2, msle = result[1]["r2"], result[1]["msle"]
                    file.write("{} \t r2: {} \t msle: {} \n".format(country, r2, msle))
                overall_r2 = sum(r2_list) / len(r2_list)
                overall_msle = sum(msle_list) / len(msle_list)
                file.write("overall score: \t\t r2: {} \t msle: {} ".format(overall_r2, overall_msle))
        except Exception as ex:
            logger.error(ex)

        logger.info("Finish.")

    except Exception as ex:
        logger.error(ex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into train and test")
    parser.add_argument("--model_type", "-m", default="exp",help="model type: exp, log or lstm")
    parser.add_argument("--test", "-t", default= "data/sample/training_pipeline/test", help="path to the test file")
    parser.add_argument("--pred", "-p", default= "data/sample/training_pipeline/pred", help="path to the prediction file")
    parser.add_argument("--model_dir", "-o", default= "model", help="path to the saving folder")
    args = parser.parse_args()

    main(args)
