import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import evaluation as eval
import os
import warnings
#from neural import NN_wrapper
from util import get_clean_data
from LogisticRegression import RegressionModel
from svm_model import SVMModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    features = [
        'EMA10','EMA12','EMA20','EMA26','EMA50','EMA100','EMA200',
        'SMA5','SMA10','SMA15','SMA20','SMA50','SMA100','SMA200',
        'EMA10Cross','EMA12Cross','EMA20Cross','EMA26Cross','EMA50Cross','EMA100Cross','EMA200Cross',
        'MACD','Volume','Price',
        'Up-Down5','Up-Down10','Up-Down15','Up-Down20','Up-Down50','Up-Down100',
        'SMA5Cross','SMA10Cross','SMA15Cross','SMA20Cross','SMA50Cross','SMA100Cross','SMA200Cross'
    ]
    regularized_features = [
        'SMA5','SMA15','SMA20','SMA200',
        'EMA10Cross','EMA20Cross','EMA26Cross','EMA50Cross','EMA100Cross','EMA200Cross',
        'MACD','Volume','Price',
        'Up-Down10','Up-Down15','Up-Down50','Up-Down100'
        'SMA5Cross','SMA10Cross','SMA15Cross','SMA20Cross','SMA50Cross','SMA100Cross','SMA200Cross'
    ]
    label = ['Class']

    #Get Data
    df_train, df_valid, df_test = get_clean_data()
    Xtrain = df_train[features].values
    Ytrain = df_train[label].values.ravel()
    Xvalid = df_valid[features].values
    Yvalid = df_valid[label].values.ravel()
    Xtest = df_test[features].values
    Ytest = df_test[label].values.ravel()

    '''
    Training
    '''

    #Logistic Regression Model
    logistic = RegressionModel(1)
    logistic.train(Xtrain, Ytrain)
    logistic_ridge = RegressionModel(2)
    logistic_ridge.train(Xtrain, Ytrain)
    logistic_lasso = RegressionModel(3)
    logistic_lasso.train(Xtrain, Ytrain)

    #Neural Network
    NN = NN_wrapper(Xtrain.shape[1])
    NN.train(Xtrain,Ytrain)
    print("Finish Neural Network")

    #SVM Model
    svm_linear = SVMModel(1)
    svm_poly = SVMModel(2)
    svm_rbf = SVMModel(3)
    svm_sigmoid = SVMModel(4)

    svm_linear.train(Xtrain, Ytrain)
    print("Finish SVM Linear")
    svm_poly.train(Xtrain, Ytrain)
    print("Finish SVM Poly")
    svm_rbf.train(Xtrain, Ytrain)
    print("Finish SVM Rbf")
    svm_sigmoid.train(Xtrain, Ytrain)
    print("Finish SVM Sigmoid")

    #LSTM Model

    look_back = 10
    def create_dataset(Xtrain, Ytrain, look_back):
        dataX, dataY = [], []
        for i in range(len(Xtrain)-look_back-1):
            a = Xtrain[i:(i+look_back)]
            dataX.append(a)
            dataY.append(Ytrain[i + look_back])
        return np.array(dataX), np.array(dataY)


    scale = MinMaxScaler(feature_range=(0, 1))
    Xtrain = scale.fit_transform(Xtrain)
    Ytrain = scale.fit_transform(Ytrain.reshape(-1,1))
    Xtest = scale.fit_transform(Xtest)
    Ytest = scale.fit_transform(Ytest.reshape(-1,1))

    #Transform dimensions
    trainX, trainY = create_dataset(Xtrain, Ytrain, look_back)
    testX, testY = create_dataset(Xtest, Ytest, look_back)

    lstm = LSTMModel()
    lstm.train(trainX, trainY)

    y_pred = lstm.predict(testX)
    acc = lstm.evaluate(testX, testY)
    print("Acuuracy (LSTM)" + str(-acc))

    y_true = Ytest[look_back+1:].ravel()
    print('AUC Score of', 'LSTM')
    print(roc_auc_score(np.array(y_true), np.array(y_pred.ravel())))


    '''
    Prediction
    '''
    all_pred = {}
    all_pred['logistic_pred'] = np.array(logistic.predict(Xvalid)[:,1])
    all_pred['nn_pred'] = np.array(NN.predict(Xvalid))
    all_pred['logistic_ridge_pred'] = np.array(logistic_ridge.predict(Xvalid))
    all_pred['logistic_lasso_pred'] = np.array(logistic_lasso.predict(Xvalid))
    all_pred['svm_linear_pred'] = np.array(svm_linear.predict(Xvalid))
    all_pred['svm_poly_pred'] = np.array(svm_poly.predict(Xvalid))
    all_pred['svm_rbf_pred'] = np.array(svm_rbf.predict(Xvalid))
    all_pred['svm_sigmoid_pred'] = np.array(svm_sigmoid.predict(Xvalid))

    '''
    Evaluations
    '''
    for key, pred in all_pred.items():
        print('Accuracy Score of', key)
        print(eval.accuracy(prediction = pred, true_class = Yvalid))
        print('F1 Score of', key)
        print(eval.f1score(prediction = pred, true_class = Yvalid, average='macro'))
        print('='*50)

    '''
    Portfolio Generation
    '''
    price = df_valid['Price'].values
    for key, pred in all_pred.items():
        portfolio = eval.portfolio_generator(principal = 10000.,prediction = pred, true_price = price,threshold = [0.499,0.501] ,leverage = 1 ,short = True)
        abs_profit, profit, sharpe = eval.profit_eval(portfolio)
        print('Annual Profit for', key)
        print(100*profit, '%')
        print('*'*50)

if __name__ == "__main__":
    main()
