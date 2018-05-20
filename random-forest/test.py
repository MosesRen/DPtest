from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
from scipy.stats import mode
from utilities import information_gain, entropy,shuffle_in_unison
from decisiontree import DecisionTreeClassifier
from randomforest import RandomForestClassifier
from sklearn.metrics import accuracy_score 


def load_traindata(filepath):
    #读取数据集

    csv = pd.read_csv(filepath,index_col = None,parse_dates = True,iterator = True)
    dataset = csv.read()
    # # 读取数据集
    return dataset


if __name__ == '__main__': 
    
    dataset = load_traindata(r'wine.txt')
    train_data = dataset.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values
    train_target = dataset.iloc[:,[13]].values
    # from sklearn.ensemble import RandomForestClassifier
    # model =  RandomForestClassifier(n_estimators=15, n_jobs=8)
    # fit_model = model.fit(X_train, y_train.ravel()) 
    # for i in range(10):
    # print ("the %s times"% i)
    X_train, X_test, y_train, y_test =  train_test_split(train_data, train_target,test_size=0.3, random_state=42)
    tree = RandomForestClassifier()
    tree.fit(X_train, y_train.ravel())
    y_pred_rf = tree.predict(X_test)
    y_true_rf = y_test.ravel()
    print( accuracy_score(y_true_rf, y_pred_rf))
        # y_pred_rf = tree.predict(X_test,0.2)
        # y_true_rf = y_test.ravel()
        # print( accuracy_score(y_true_rf, y_pred_rf))
        # y_pred_rf = tree.predict(X_test,0.5)
        # y_true_rf = y_test.ravel()
        # print( accuracy_score(y_true_rf, y_pred_rf))
        # y_pred_rf = tree.predict(X_test,1)
        # y_true_rf = y_test.ravel()
        # print(accuracy_score(y_true_rf, y_pred_rf))
        # y_pred_rf = tree.predict(X_test,5)
        # y_true_rf = y_test.ravel()
        # print(accuracy_score(y_true_rf, y_pred_rf))
        # y_pred_rf = tree.predict(X_test,10)
        # y_true_rf = y_test.ravel()
        # print(accuracy_score(y_true_rf, y_pred_rf))
        # y_pred_rf = tree.predict(X_test,0.0)
        # y_true_rf = y_test.ravel()
        # print(accuracy_score(y_true_rf, y_pred_rf))
        # y_pred = model.predict(X_test)
        # y_true = y_test.ravel()
        # print(accuracy_score(y_true, y_pred))