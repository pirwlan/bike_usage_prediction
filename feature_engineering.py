from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

import numpy as np


def season_pipeline():
    return Pipeline([('season_onehot', OneHotEncoder(drop='first', sparse=False))])


def workingday_pipeline():
    return Pipeline([('workingday_onehot', OneHotEncoder(drop='first', sparse=False))])


def weather_pipeline():
    return Pipeline([('weather_onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])


def temp_pipeline():
    return Pipeline([('temp_scaler', StandardScaler())])
                      #KBinsDiscretizer(encode='onehot',
                      #                                     strategy='uniform'))])


def atemp_pipeline():
    return Pipeline([('atemp_scaler', StandardScaler())])


def hour_pipeline():
    return Pipeline([('hour_onehot', OneHotEncoder(drop='first', sparse=False))])


def holiday_pipeline():
    return Pipeline([('holiday_onehot', OneHotEncoder(drop='first', sparse=False))])


def wind_pipeline():
    return Pipeline([('wind_scaler', StandardScaler())])


def month_pipeline():
    return Pipeline([('month_onehot', OneHotEncoder(drop='first', sparse=False))])


def day_pipeline():
    return Pipeline([('day_onehot', OneHotEncoder(drop='first', sparse=False))])


def humidity_pipeline():
    return Pipeline([('humidity_scaler', StandardScaler())])


def year_pipeline():
    return Pipeline([('year_onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])


def setup_col_transformer():

    season_pipe = season_pipeline()
    hour_pipe = hour_pipeline()
    workday_pipe = workingday_pipeline()
    weather_pipe = weather_pipeline()
    temp_pipe = temp_pipeline()
    atemp_pipe = atemp_pipeline()
    holiday_pipe = holiday_pipeline()
    wind_pipe = wind_pipeline()
    month_pipe = month_pipeline()
    day_pipe = day_pipeline()
    year_pipe = year_pipeline()
    humidity_pipe = humidity_pipeline()

    return ColumnTransformer([
        ('season_pipe', season_pipe, ['season']),
        ('workday_pipe', workday_pipe, ['workingday']),
        ('weather_pipe', weather_pipe, ['weather']),
        ('temp_pipe', temp_pipe, ['temp']),
        ('hour_pipe', hour_pipe, ['hour']),
        #('atemp_pipe', atemp_pipe, ['atemp']),
        ('holiday_pipe', holiday_pipe, ['holiday']),
        ('wind_pipe', wind_pipe, ['windspeed']),
        ('month_pipe', month_pipe, ['month']),
        ('day_pipe', day_pipe, ['day']),
        ('year_pipe', year_pipe, ['year']),
        ('humidity_pipe', humidity_pipe, ['humidity'])
        ])


def rgr_pipe():
    col_trans = setup_col_transformer()
    ranfor_reg = HistGradientBoostingRegressor()
    return Pipeline([('feature_engineering', col_trans),
                     # ('polynominal', PolynomialFeatures()),
                     ('ranfor_regression', ranfor_reg)])


def grid_parameters():

    return {'ranfor_regression__loss': ['poisson'],
            'ranfor_regression__l2_regularization': np.linspace(0, 1, 5),
            'ranfor_regression__learning_rate': np.linspace(0.1, 0.5, 4),
            #'ranfor_regression__max_features': np.arange(5, 10),
            #'ranfor_regression__min_samples_split': [5],
            'ranfor_regression__max_depth': [50, 75, 100],
            #'feature_engineering__temp_pipe__temp_diskretizer__n_bins': [4, 6, 8],
            }


def my_scorer():
    return {'RMSLE': make_scorer(RMSLE, greater_is_better=False)}


def RMSLE(y_true, y_predict):
    """
    Implentation of Root mean squared log error
    """
    return np.sqrt(mean_squared_log_error(y_true, y_predict))
