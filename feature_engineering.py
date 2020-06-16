from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler

import numpy as np


def season_pipeline():
    return Pipeline([('season_onehot', OneHotEncoder(drop='first', sparse=False))])


def workingday_pipeline():
    return Pipeline([('workingday_onehot', OneHotEncoder(drop='first', sparse=False))])


def weather_pipeline():
    return Pipeline([('weather_onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])


def temp_pipeline():
    return Pipeline([('temp_diskretizer', KBinsDiscretizer(encode='onehot',
                                                      strategy='uniform'))])


def hour_pipeline():
    return Pipeline([('hour_diskretizer', KBinsDiscretizer(encode='onehot',
                                                      strategy='uniform'))])


def setup_col_transformer():

    season_pipe = season_pipeline()
    hour_pipe = hour_pipeline()
    workday_pipe = workingday_pipeline()
    weather_pipe = weather_pipeline()
    temp_pipe = temp_pipeline()

    return ColumnTransformer([
        ('season_pipe', season_pipe, ['season']),
        ('workday_pipe', workday_pipe, ['workingday']),
        ('weather_pipe', weather_pipe, ['weather']),
        ('temp_pipe', temp_pipe, ['temp']),
        ('hour_pipe', hour_pipe, ['hour']),

        ])


def rgr_pipe():
    col_trans = setup_col_transformer()
    ranfor_reg = RandomForestRegressor()
    return Pipeline([('feature_engineering', col_trans),
                     ('ranfor_regression', ranfor_reg)])


def grid_parameters():

    return {'ranfor_regression__n_estimators': [100, 200, 400],
            'ranfor_regression__max_depth': [4, 8, 12, 16, 20],
            'ranfor_regression__min_samples_split': np.linspace(0.1, 1.0, 10),
            'feature_engineering__temp_pipe__temp_diskretizer__n_bins': [8, 9, 10, 11, 12],
            'feature_engineering__hour_pipe__hour_diskretizer__n_bins': [4, 6, 8],}


def my_scorer():
    return {'RMSLE': make_scorer(RMSLE, greater_is_better=False)}


def RMSLE(y_true, y_predict):
    """
    Implentation of Root mean squared log error
    """
    return np.sqrt(mean_squared_log_error(y_true, y_predict))

