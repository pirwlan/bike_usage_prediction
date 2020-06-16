from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import numpy as np


def season_pipeline():
    return Pipeline([('season_onehot', OneHotEncoder(drop='first', sparse=False))])


def setup_col_transformer():

    season_pipe = season_pipeline()

    return ColumnTransformer([
        ('season_pipe', season_pipe, ['season']),
        ])


def rgr_pipe():
    col_trans = setup_col_transformer()
    linreg = LinearRegression()
    return Pipeline([('feature_engineering', col_trans),
                     ('linear_regression', linreg)])


def grid_parameters():

    return {'linear_regression__normalize': [True, False]}


def my_scorer():
    return {'RMSLE': make_scorer(RMSLE, greater_is_better=False)}


def RMSLE(y_true, y_predict):
    """
    Implentation of Root mean squared log error
    """
    return np.sqrt(mean_squared_log_error(y_true, y_predict))