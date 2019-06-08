import numpy as np
import pandas as pd
import evaluation as eval
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


#Import
prediction = pd.read_csv('output/summary.csv')
SVC_pred = prediction['SVC']
SVM_ploy = prediction['SVM']
SVC_pred = prediction['SVC']
SVC_pred = prediction['SVC']
SVC_pred = prediction['SVC']
SVC	SVMPoly	SVMRadial	SVMSigmoid	GRU	SingleLSTM	MultiLSTM	Price	CNN

#Plot
plt.plot(result['Baseline'], label = "Baseline Strategy")
plt.plot(result['LSTM'], label = "LSTM")
plt.plot(result['Linear'], label = "Linear Regression")
plt.plot(result['Neural'], label = "Neural Network")
plt.title("Portfolio Value Overtime of Different Strategies ", fontsize = 15)
plt.axis([0, 450, 0.90, 1.40])
plt.legend()
plt.xlabel('Index: 1-Minute Interval')
plt.ylabel('Portfolio Value')
plt.show()

#Statistics

print('Baseline', bs.sharpe_calc(result['Baseline']), bs.max_drawdown(result['Baseline']))
print('LSTM', bs.sharpe_calc(result['LSTM']), bs.max_drawdown(result['LSTM']))
print('Linear', bs.sharpe_calc(result['Linear']), bs.max_drawdown(result['Linear']))
print('Neural', bs.sharpe_calc(result['Neural']), bs.max_drawdown(result['Neural']))





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
    abs_profit, profit, sharpe, profit_per_hr = eval.profit_eval(portfolio)
    print('Annual Profit for', key)
    print(100*profit, '%')
    print('*'*50)
