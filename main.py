import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import evaluation as eval
import os
import warnings
from neural import NN_wrapper
from util import get_clean_data
from LogisticRegression import RegressionModel
from svm_model import SVMModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    features = [
        'EMA200',
        'EMA100',
        'EMA12',
        'EMA20',
        'EMA26',
        'EMA50',
        'SMA10',
        'SMA100',
        'SMA15',
        'SMA20',
        'SMA200',
        'SMA50',
        'MACD',
        'Volume',
        'EMA200Cross',
        'EMA100Cross',
        'EMA12Cross',
        'EMA20Cross',
        'EMA26Cross',
        'EMA50Cross',
        'SMA10Cross',
        'SMA100Cross',
        'SMA15Cross',
        'SMA20Cross',
        'SMA200Cross',
        'SMA50Cross'
    ]
    label = ['Class']

    #Get Data
    df_train, df_validation, df_test = get_clean_data()

    Xtrain = df_train[features].values
    Ytrain = df_train[label].values.ravel()
    Xvalid = df_valid[features].values
    Yvalid = df_valid[label].values.ravel()
    Xtest = df_test[features].values
    Ytest = df_test[label].values.ravel()

    '''
    Training
    '''
    #Neural Network
    NN = NN_wrapper(Xtrain.shape[1])
	NN.train(Xtrain,Ytrain)

    #Logistic Regression Model
    model = RegressionModel(1)
    model_ridge = RegressionModel(2)
    model_lasso = RegressionModel(3)
    model.train(Xtrain, Ytrain)
    model_ridge.train(Xtrain, Ytrain)
    model_lasso.train(Xtrain, Ytrain)

    #SVM Model
    svm_linear = SVMModel(1)
    svm_poly = SVMModel(2)
    svm_rbf = SVMModel(3)
    svm_sigmoid = SVMModel(4)
    svm_linear.train(Xtrain, Ytrain)
    svm_poly.train(Xtrain, Ytrain)
    svm_rbf.train(Xtrain, Ytrain)
    svm_sigmoid.train(Xtrain, Ytrain)

    '''
    Prediction
    '''
    nn_pred = np.array(NN.predict(Xtrain))
    logistic_pred = np.array(model.predict(Xtest))
    logistic_ridge_pred = np.array(model_ridge.predict(Xtest))
    logistic_lasso_pred = np.array(model_lasso.predict(Xtest))
    svm_linear_pred = np.array(svm_linear.predict(Xtest))
    svm_poly_pred = np.array(svm_poly.predict(Xtest))
    svm_rbf_pred = np.array(svm_rbf.predict(Xtest))
    svm_sigmoid_pred = np.array(svm_sigmoid.predict(Xtest))

    '''
    Evaluation
    '''
    accuracy_clf = accuracy_score(Ytest, Y_pred, normalize = True)
    accuracy_clf_ridge = accuracy_score(Ytest, Y_pred_ridge, normalize = True)
    accuracy_clf_lasso = accuracy_score(Ytest, Y_pred_lasso, normalize = True)
    f1_clf = f1_score(Ytest, np.array(Y_pred), average='weighted')
    f1_clf_ridge = f1_score(Ytest, np.array(Y_pred_ridge), average='weighted')
    f1_clf_lasso = f1_score(Ytest, np.array(Y_pred_lasso), average='weighted')

    '''
    Portfolio Generation
    '''


if __name__ == "__main__":
    main()
