import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from LogisticRegression import RegressionModel
from svm_model import SVMModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    features = [
        'EMA10',
        'EMA100',
        'EMA12',
        'EMA20',
        'EMA26',
        'EMA50',
        'SMA10',
        'SMA100',
        'SMA15',
        'SMA20',
        'SMA5',
        'SMA50',
        'MACD',
        'Volume'
    ]

    label = ['Class']

    directory = 'CleanData'
    df_train = []
    df_valid = []
    df_test = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if (filename == "train_set.csv"):
            df_train = pd.read_csv(directory + '/' + filename)
        elif (filename == "validation_set.csv"):
            df_valid = pd.read_csv(directory + '/' + filename)
        elif (filename == "test_set.csv"):
            df_test = pd.read_csv(directory + '/' + filename)
        else:
            None

    Xtrain = df_train[features].values
    Ytrain = df_train[label].values.ravel()
    Xvalid = df_valid[features].values
    Yvalid = df_valid[label].values.ravel()
    Xtest = df_test[features].values
    Ytest = df_test[label].values.ravel()

    model = RegressionModel(1)
    model_ridge = RegressionModel(2)
    model_lasso = RegressionModel(3)
    model.train(Xtrain, Ytrain)
    model_ridge.train(Xtrain, Ytrain)
    model_lasso.train(Xtrain, Ytrain)

    threshold = 0.5

    Y_pred = np.array(model.predict(Xtest))
    Y_pred_ridge = np.array(model_ridge.predict(Xtest) > threshold)
    Y_pred_lasso = np.array(model_lasso.predict(Xtest) > threshold)

    #Evaluation
    accuracy_clf = accuracy_score(Ytest, Y_pred, normalize = True)
    accuracy_clf_ridge = accuracy_score(Ytest, Y_pred_ridge, normalize = True)
    accuracy_clf_lasso = accuracy_score(Ytest, Y_pred_lasso, normalize = True)

    print(accuracy_clf, accuracy_clf_ridge, accuracy_clf_lasso)

    f1_clf = f1_score(Ytest, np.array(Y_pred), average='weighted')
    f1_clf_ridge = f1_score(Ytest, np.array(Y_pred_ridge), average='weighted')
    f1_clf_lasso = f1_score(Ytest, np.array(Y_pred_lasso), average='weighted')

    svm_linear = SVMModel(1)
    svm_poly = SVMModel(2)
    svm_rbf = SVMModel(3)
    svm_sigmoid = SVMModel(4)
    svm_linear.train(Xtrain, Ytrain)
    svm_poly.train(Xtrain, Ytrain)
    svm_rbf.train(Xtrain, Ytrain)
    svm_sigmoid.train(Xtrain, Ytrain)

    Ypred_svm_linear = np.array(svm_linear.predict(Xtest))
    Ypred_svm_poly = np.array(svm_poly.predict(Xtest))
    Ypred_svm_rbf = np.array(svm_rbf.predict(Xtest))
    Ypred_svm_sigmoid = np.array(svm_sigmoid.predict(Xtest))

    print(Ypred_svm_sigmoid)

if __name__ == "__main__":
    main()
