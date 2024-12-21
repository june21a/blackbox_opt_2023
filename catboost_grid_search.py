import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from modules.preprocessing import preprocess
from modules.utils import load_yaml
from catboost import CatBoostRegressor
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_path", type=str, default="./module/config.yml", help="config file path")
    parser.add_argument("--pca_dim", type=int, default=6, help="target dimension for pca")
    parser.add_argument("--save_path", type=str, default="./model_params", help="where to save models")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    CFG = load_yaml(args.yml_path)
    file_name_prefix = f"{datetime.strftime(datetime.now(), "%Y/%m/%d_%H:%M")}_{args.pca_dim}"
    log_file_path = f"./logs/{file_name_prefix}_grid_catboost.txt"
    
    # search space
    params = {'iterations': [1000],
            'learning_rate': np.logspace(-3, -1, 3),
            'depth': [4, 6],
            'loss_function': ['RMSE'],
            'l2_leaf_reg': np.logspace(-20, -19, 3),
            'leaf_estimation_iterations': [10],
            'eval_metric': ['RMSE'],
            'random_seed': [CFG["random_seed"]],
            'logging_level': 'Verbose',
            "log_file": log_file_path
            }
    
    # load data
    X_train, y_train, X_test, X_scaler, y_scaler = preprocess(args.pca_dim, CFG)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=CFG["random_seed"])
    fit_params = {'early_stopping_rounds': CFG["early_stopping_rounds"], 'eval_set':[(X_val, y_val)]}
    
    # grid search
    model = CatBoostRegressor()
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_cv = GridSearchCV(model, param_grid=params, cv=5, n_jobs=1, scoring=scorer, verbose=2)
    grid_cv.fit(X_train, y_train, **fit_params)
    
    
    # save model trained from best params
    model = CatBoostRegressor(**grid_cv.best_params_)

    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=CFG["early_stopping_rounds"],
            verbose=100)
    model.save_model(os.path.join(args.save_path, f"{file_name_prefix}_model.cbm"))
    print("best params\n", grid_cv.best_params_)
    
    with open(log_file_path, 'a') as f:
        f.write("\n\nbest params\n")
        json.dump(grid_cv.best_params_, f)
        f.write("\n")
    
    

if __name__ == "__main__":
    main()