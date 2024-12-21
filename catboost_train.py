import os
import argparse
from modules.utils import load_yaml
from modules.preprocessing import preprocess
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_path", type=str, default="./module/config.yml", help="config file path")
    parser.add_argument("--pca_dim", type=int, default=6, help="target dimension for pca")
    parser.add_argument("--save_path", type=str, default="./model_params", help="where to save models")
    
    args = parser.parse_args()
    return args


def train(params, X_train, y_train, X_val, y_val, save_path, early_stopping_rounds, verbose=100):
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              early_stopping_rounds=early_stopping_rounds,
              verbose=verbose
              )
    
    model.save_model(save_path)
    return model

def main():
    args = parse_arguments()
    CFG = load_yaml(args.yml_path)
    CFG['model_params']["random_seed"] = CFG["random_seed"]
    
    file_name_prefix = f"{datetime.strftime(datetime.now(), "%Y/%m/%d_%H:%M")}_{args.pca_dim}"
    log_file_path = f"./logs/{file_name_prefix}_catboost.txt"
    
    
    # load data
    X_train, y_train, X_test, X_scaler, y_scaler = preprocess(args.pca_dim, CFG)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=CFG["random_seed"])
    
    train(CFG['model_params'], 
        X_train, y_train, 
        X_val, y_val,
        save_path = os.path.join(args.save_path, f"{file_name_prefix}_model.cbm"),
        early_stopping_rounds=CFG["early_stopping_rounds"],
        verbose=100)

if __name__=="__main__":
    main()    