import numpy as np
import pandas as pd
import evaluation as eval
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

TRANSACTION_COST = 0.109 #10.9 bps according to http://www.integrity-research.com/equity-commission-rates-remain-steady/

'''
Import File
'''
prediction = pd.read_csv('output/summary.csv')
prediction_dict = dict()
prediction_dict['SVC_pred'] = prediction['SVC'].values
prediction_dict['SVM_ploy'] = prediction['SVMPoly'].values
prediction_dict['SVM_Radial'] = prediction['SVMRadial'].values
prediction_dict['SVM_Sigmoid'] = prediction['SVMSigmoid'].values
prediction_dict['GRU'] = prediction['GRU'].values
prediction_dict['Single_LSTM'] = prediction['SingleLSTM'].values
prediction_dict['Multi_LSTM'] = prediction['MultiLSTM'].values
prediction_dict['price'] = prediction['Price'].values
prediction_dict['CNN'] = prediction['CNN'].values
prediction_dict['logistic_lasso'] = prediction['logistic_lasso_pred'].values
prediction_dict['logistic_ridge'] = prediction['logistic_ridge_pred'].values
prediction_dict['logistic'] = prediction['logistic_pred'].values

'''
Portfolio Generation
'''
result = dict()
for key, pred in prediction_dict.items():
    portfolio = eval.portfolio_generator(principal = 1000.,prediction = pred, true_price = price,threshold = [0.499,0.501] ,leverage = 1 ,short = True, transc = TRANSACTION_COST)
    abs_profit, profit, sharpe, profit_per_hr = eval.profit_eval(portfolio)
    result[str(key)] = portfolio
    print('Profit/hr', key)
    print(profit_per_hr)
    print('*'*50)


#Plot
# plt.plot(result['Baseline'], label = "Baseline Strategy")
# plt.plot(result['LSTM'], label = "LSTM")
# plt.plot(result['Linear'], label = "Linear Regression")
# plt.plot(result['Neural'], label = "Neural Network")
# plt.title("Portfolio Value Overtime of Different Strategies ", fontsize = 15)
# plt.axis([0, 450, 0.90, 1.40])
# plt.legend()
# plt.xlabel('Index: 1-Minute Interval')
# plt.ylabel('Portfolio Value')
# plt.show()

#Statistics

# print('Baseline', bs.sharpe_calc(result['Baseline']), bs.max_drawdown(result['Baseline']))
# print('LSTM', bs.sharpe_calc(result['LSTM']), bs.max_drawdown(result['LSTM']))
# print('Linear', bs.sharpe_calc(result['Linear']), bs.max_drawdown(result['Linear']))
# print('Neural', bs.sharpe_calc(result['Neural']), bs.max_drawdown(result['Neural']))





for key, pred in all_pred.items():
    print('Accuracy Score of', key)
    print(eval.accuracy(prediction = pred, true_class = Yvalid))
    print('F1 Score of', key)
    print(eval.f1score(prediction = pred, true_class = Yvalid, average='macro'))
    print('='*50)
