import os
import pandas as pd


def get_data():
    data_path = os.path.join(os.getcwd(), 'data')

    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'),
                           parse_dates=['datetime'])

    y_train = df_train[['count', 'casual', 'registered']]

    df_train.drop(columns=['count', 'casual', 'registered'], inplace=True)

    df_test = pd.read_csv(os.path.join(data_path, 'test.csv'),
                          parse_dates=['datetime'])

    return df_train, y_train['count'], df_test


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


def create_submission(best_model, df_test):
    submit_path = os.path.join(os.getcwd(), 'submission')
    verify_folder_existence(submit_path)

    y_hat = best_model.predict(df_test)
    df_submit = pd.DataFrame(y_hat, columns=['counts'], index=df_test.index)
    df_submit.to_csv(os.path.join(submit_path, 'submission.csv'))
