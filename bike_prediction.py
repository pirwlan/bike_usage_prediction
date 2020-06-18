from sklearn.model_selection import GridSearchCV

import feature_engineering as fe
import helper as h


def bike_prediction():

    df_train, y_train, df_test = h.get_data()
    rgr_cas_pipe = fe.rgr_pipe()

    param_grid = fe.grid_parameters()

    grid_search_casual = GridSearchCV(estimator=rgr_cas_pipe,
                                      param_grid=param_grid,
                                      scoring=fe.my_scorer(),
                                      return_train_score=True,
                                      refit='RMSLE',
                                      cv=10,
                                      n_jobs=-1,
                                      verbose=1)

    print('Results for casual:')
    best_regressor_casual = grid_search_casual.fit(df_train, y_train['casual'])
    h.evaluate_summary(best_regressor_casual)

    rgr_reg_pipe = fe.rgr_pipe()
    grid_search_reg = GridSearchCV(estimator=rgr_reg_pipe,
                                   param_grid=param_grid,
                                   scoring=fe.my_scorer(),
                                   return_train_score=True,
                                   refit='RMSLE',
                                   cv=10,
                                   n_jobs=-1,
                                   verbose=1)

    best_regressor_reg = grid_search_reg.fit(df_train, y_train['registered'])

    print('Results for registered:')
    h.evaluate_summary(best_regressor_reg)

    h.create_submission(best_regressor_casual, best_regressor_reg, df_test)


if __name__ == '__main__':
    bike_prediction()
