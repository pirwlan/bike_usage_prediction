import numpy as np
import os
import pandas as pd


def date_transforms(dfs: list):
    new_dfs = []
    for df in dfs:

        df['day'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        new_dfs.append(df)
    return new_dfs


def get_data():
    data_path = os.path.join(os.getcwd(), 'data')

    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'),
                           parse_dates=['datetime'])

    y_train = df_train[['count', 'casual', 'registered']]

    df_train.drop(columns=['count', 'casual', 'registered'], inplace=True)

    df_test = pd.read_csv(os.path.join(data_path, 'test.csv'),
                          parse_dates=['datetime'])

    df_train, df_test = date_transforms([df_train, df_test])

    #y_train = np.log1p(y_train['count'])
    y_train = np.log1p(y_train)
    #y_train = y_train['count']
    return df_train, y_train, df_test


def evaluate_summary(model):

    summary = model.cv_results_
    print('--------Best parameter--------')

    for para, value in model.best_params_.items():
        print(f'{para}: {value}')

    print('-----------Training-----------')
    print(f'RMSLE_score: {-1 * summary["mean_train_RMSLE"][-1]:.4f} ')
    print('-------------Test-------------')
    print(f'RMSLE_score: {-1 * summary["mean_test_RMSLE"][-1]:.4f}')


def verify_folder_existence(path):
    try:
        os.mkdir(path)

    except FileExistsError:
        pass


def create_submission(casual_model, reg_model, df_test):
    submit_path = os.path.join(os.getcwd(), 'submission')
    verify_folder_existence(submit_path)

    y_hat_casual = casual_model.predict(df_test)
    y_hat_reg = reg_model.predict(df_test)

    df_submit = pd.read_csv(os.path.join(os.getcwd(), 'data', 'sampleSubmission.csv'))

    y_hat_casual = np.expm1(y_hat_casual)
    y_hat_reg = np.expm1(y_hat_reg)
    y_hat = y_hat_casual + y_hat_reg
    df_submit['count'] = y_hat
    df_submit.to_csv(os.path.join(submit_path, 'submission.csv'), index=0)
