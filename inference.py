import os
import argparse
import yaml
from modules.preprocessing import preprocess
from modules.utils import transform_and_submit, load_yaml
from catboost import CatBoostRegressor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, default="./train_results/2024-12-21_21:28_PCA7", help="path to train result folder")
    parser.add_argument("--save_path", type=str, default="./submissions", help="where to save submission.csv")
    
    args = parser.parse_args()
    return args

def inference(model, n_components, CFG):
    X_train, y_train, X_test, X_scaler, y_scaler = preprocess(n_components, CFG)
    pred = model.predict(X_test)
    return pred, y_scaler


def main():
    args = parse_arguments()
    root = args.root_folder
    save_to = args.save_path
    save_file_path = os.path.join(save_to, f"{os.path.basename(root)}_submission.csv")
    os.makedirs(save_to, exist_ok=True)
    
    # load default files
    CFG = load_yaml(os.path.join(root, 'config.yml'))
    n_components = CFG["pca_dim"]
    model = CatBoostRegressor().load_model(os.path.join(root, 'model_params/model.cbm'))
    
    
    # predict
    pred, y_scaler = inference(model, n_components, CFG)
    transform_and_submit(pred, 
                       CFG["path_to_submission_csv"],
                       save_file_path,
                       y_scaler,
                       CFG['PREPROCESS_TARGET'])


if __name__=="__main__":
    main()