from sklearn.model_selection import GridSearchCV

import feature_engineering as fe
import helper as h


def bike_prediction():

    df_train, y_train, df_test = h.get_data()
    rgr_pipe = fe.rgr_pipe()

    param_grid = fe.grid_parameters()

    grid_search = GridSearchCV(estimator=rgr_pipe,
                               param_grid=param_grid,
                               scoring=fe.my_scorer(),
                               return_train_score=True,
                               refit='RMSLE',
                               cv=10,
                               n_jobs=-1)

    best_regressor = grid_search.fit(df_train, y_train)

    h.evaluate_summary(best_regressor)
    h.create_submission(best_regressor, df_test)


if __name__ == '__main__':
    bike_prediction()
