# evaluation
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

def evaluate(y_test, y_pred):
    ''' evaluation of model result
    input:
        y_test (numpy.array): array of actual data
        y_pred (numpy.array): array of predicted data
    output:
        ({"r2": float, "msle": float}): dictionary of metrics
    '''
    try:
        r2 = r2_score(y_test, y_pred)
        msle = mean_squared_log_error(y_test, y_pred)
        return {'r2': r2, 'msle': msle}
    except Exception as ex:
        logger.error(ex)


