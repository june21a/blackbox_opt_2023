import yaml
import pickle
import numpy as np
import pandas as pd


def load_yaml(yml_path):
    # YAML 파일을 읽어서 Python 딕셔너리로 변환
    with open(yml_path, 'r') as yml_file:
        yml_data = yaml.safe_load(yml_file)
    return yml_data


def predict_and_submit(y_pred, path_to_submission_csv, path_to_result_csv, y_scaler=None):
    # 예측값을 필요시 역변환하고, 이를 제출파일로 만들어 저장
    if y_scaler is not None:
        try:
            y_pred = y_scaler.inverse_transform(y_pred)
        except:
            y_pred = y_scaler.inverse_transform(y_pred.reshape(1, -1))
        y_pred = y_pred.reshape(-1)

    # Identify top 33% of predicted values
    threshold = np.percentile(y_pred, 90)
    top_10_percent_mask = y_pred >= threshold

    # Create submission file
    submission_df = pd.read_csv(path_to_submission_csv)
    submission_df['y'] = y_pred
    submission_df.to_csv(path_to_result_csv, index=False)

    print(f"Top 10% threshold: {threshold:.4f}")
    print(f"Number of samples in top 10%: {sum(top_10_percent_mask)}")


def printer_dec(verbose):
    # verbose 설정을 위한 decorator
    def printer(*args):
        if verbose:
            for s in args:
                print(s, end=" ")
    return printer


def dump_params_dict(file_path, params):
    # 모델 저장
    with open(file_path, 'wb') as fw:
        pickle.dump(params, fw)


def load_params_dict(file_path):
    # 모델 load
    with open(file_path, 'rb') as fr:
        loaded = pickle.load(fr)
    return loaded