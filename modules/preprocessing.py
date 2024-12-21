import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from .utils import printer_dec

def preprocess(n_components, CFG, vervose = False):
    train_df = pd.read_csv(CFG["path_to_train_csv"])
    test_df = pd.read_csv(CFG["path_to_test_csv"])
    print = printer_dec(vervose)

    if CFG["PREPROCESS_METHOD"] == "z_score":
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        print("z_score")
    elif CFG["PREPROCESS_METHOD"] == "min_max":
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
    elif CFG["PREPROCESS_METHOD"] == "robust":
        X_scaler = RobustScaler()
        y_scaler = RobustScaler()

    if CFG["DROP_OUTLIERS"]:
        train_df.drop(train_df[train_df['y'] < 70].index, inplace=True)

    if CFG["DROP_X2"]:
        train_df = train_df.drop(['x_2'], axis=1)
        test_df = test_df.drop(['x_2'], axis=1)

    if CFG["USE_X4X10_FEATURE"]:
        train_df["x_11"] = (train_df['x_4'] + train_df['x_10']) / 2
        train_df = train_df.drop(['x_4', 'x_10'], axis=1)
        train_df = train_df[['ID', 'x_0', 'x_1', 'x_2', 'x_3', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_11', 'y']]
        test_df["x_11"] = (test_df['x_4'] + test_df['x_10']) / 2
        test_df = test_df.drop(['x_4', 'x_10'], axis=1)


    if CFG["USE_PCA"]:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(train_df.iloc[:, 1:-1])
        X_test = pca.transform(test_df.iloc[:, 1:])
        print(sum(pca.explained_variance_ratio_))
        print(X_train)
    else:
        print("차원축소를 사용하지 않습니다!")

    if CFG["PREPROCESS_FEATURE"]:
        if CFG["USE_PCA"]:
            X_train = X_scaler.fit_transform(X_train)
            X_test = X_scaler.transform(X_test)
        else:
            X_train = X_scaler.fit_transform(train_df.iloc[:, 1:-1])
            X_test = X_scaler.transform(test_df.iloc[:, 1:])

    if CFG["PREPROCESS_TARGET"]:
        y_train = y_scaler.fit_transform(train_df.iloc[:, [-1]])
    else:
        y_train = train_df.iloc[:, [-1]].values

    print("train features shape : ", X_train.shape)
    print("train labels shape : ", y_train.shape)
    print("test features shape ; ", X_test.shape)
    return X_train, y_train, X_test, X_scaler, y_scaler