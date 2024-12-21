import os
import json
import argparse
from modules.utils import load_yaml, save_yaml
from modules.preprocessing import preprocess
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_path", type=str, default="./config/config.yml", help="config file path")
    parser.add_argument("--save_path", type=str, default="./train_results", help="where to save models")
    
    args = parser.parse_args()
    return args


def train(params, X_train, y_train, X_val, y_val, save_path, early_stopping_rounds, log_file_path, verbose=100):
    print("training..")
    
    file = open(log_file_path, 'w', encoding='utf-8')
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              early_stopping_rounds=early_stopping_rounds,
              verbose=verbose,
              log_cout=file)
    
    file.close()
    model.save_model(save_path)
    return model

def main():
    args = parse_arguments()
    CFG = load_yaml(args.yml_path)
    CFG['model_params']["random_seed"] = CFG["random_seed"]
    
    # make directories
    folder_path = os.path.join(args.save_path, f"{datetime.strftime(datetime.now(), '%Y-%m-%d_%H:%M')}_PCA{CFG['pca_dim']}")
    model_path = os.path.join(folder_path, 'model_params')
    os.makedirs(model_path, exist_ok=True)
    log_file_path = os.path.join(folder_path, "log.txt")
    save_yaml(CFG, os.path.join(folder_path, "config.yml"))
    
    
    # load data
    X_train, y_train, X_test, X_scaler, y_scaler = preprocess(CFG['pca_dim'], CFG)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=CFG["random_seed"])
    
    train(CFG['model_params'], 
        X_train, y_train, 
        X_val, y_val,
        save_path = os.path.join(model_path, "model.cbm"),
        early_stopping_rounds=CFG["early_stopping_rounds"],
        log_file_path=log_file_path,
        verbose=100)
    
    with open(os.path.join(folder_path, "best_params.json"), 'w') as f:
        json.dump(CFG['model_params'], f)

if __name__=="__main__":
    main()    