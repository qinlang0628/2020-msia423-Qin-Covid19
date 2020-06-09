import src.download_data as download_data
import src.clean_data as clean_data
import src.add_cases as add_cases
import config.config as config
import src.train as train 

from src.prediction_models import exponential_model, param_py as model_params

if __name__ == "__main__":
    # download_data.main()
    # clean_data.main()
    # add_cases.main(config.SQLALCHEMY_DATABASE_URI)
    train.main()
    