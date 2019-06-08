import numpy as np
import pandas as pd
from sklearn import metrics

def profit_eval(portfolio):
    """
    Output different evaluation metrics from portfolio value over time

    Parameters
    ----------
    portfolio : np.array
        portfolio value overtime
        length should be equal to nrow(test_set)

    Returns
    -------
    abs_profit: float
        Total profit in absolute term ($$)

    annual_prof: float
        Annualized profit in percentage

    sharpe: float
        sharpe ratio - calculated by sqrt[(num_trading_day)*(num_trading_minutes)]*np.mean(minute_return)/volatility
    """
    num_trading_day = 252.
    num_trading_minutes = 390.
    n = len(portfolio)-1
    return_array = np.zeros(n)
    for i in range(len(return_array)):
        return_array[i] = portfolio[i+1]/portfolio[i] - 1

    abs_profit = portfolio[len(portfolio)-1]-portfolio[0]
    power = num_trading_day*num_trading_minutes/len(portfolio)
    profit = (portfolio[len(portfolio)-1]/portfolio[0]) ** (power) - 1
    sharpe = np.sqrt(num_trading_day*num_trading_minutes)*np.mean(return_array)/np.std(return_array)
    profit_per_hour = (portfolio[n] - portfolio[0])*60/len(portfolio)

    return abs_profit, profit, sharpe, profit_per_hour

def accuracy(prediction,true_class):
    pred = np.where(prediction > 0.5, 1, 0)
    accuracy = np.mean(true_class == pred)
    return accuracy

def f1score(prediction,true_class, average='micro'):
    pred = np.where(prediction > 0.5, 1, 0)
    score = metrics.f1_score(pred, true_class, average = average)
    return score

def portfolio_generator(principal,prediction,true_price,threshold, leverage = 1,short = True, transc = 0.109):

    """
    Generate portfolio value over time from prediction

    Parameters
    ----------
    principal : float
        how much money to start off with

    prediction: np.array
        prediction from model such that each element is in [0,1]

    true_price: np.array
        stock price array

    threshold: np.array, len=2
        format = [selling threshold, buying threshold]
        baseline_case: [0.5,0.5]
        [0.2,0.7] means
            sell if pred in [0,0.2]
            hold if pred in [0.2,0.7]
            buy if pred in [0.7,1]

    leverage: float
        how much leverage to take

    short: boolean
        allowing shorting or not

    Returns
    -------
    value_over_time: np.array
        portfolio value over time // for plotting purposes
    """
    n = true_price.shape[0]
    value_over_time = np.zeros(n) #portfolio value over time
    cash = np.zeros(n) #cash value over time
    units = np.zeros(n) #shares owned over time
    cash[0] = principal*leverage
    units[0] = 0.0
    borrow = np.ones(n)*principal*(leverage-1) #amount borrowed
    cond = 1
    '''
    condition 1: all cash
    condition 2: no cash with positive # of shares
    condition 3: excess cash with negative # of shares
    '''
    for i in range(n):
        if short:
            #Entering position
            if cond == 1:
                if prediction[i] > threshold[1]:
                    if i != 0:
                        units[i] = (1-transc)*cash[i-1]/true_price[i]
                        cash[i] = 0
                        cond = 2
                    else:
                        units[i] = (1-transc)*cash[i]/true_price[i]
                        cash[i] = 0
                        cond = 2
                    #print('Enter Long from none')
                elif prediction[i] < threshold[0]:
                    if i != 0:
                        units[i] = -(1-transc)*cash[i-1]/true_price[i]
                        cash[i] = cash[i-1] - units[i]*true_price[i]
                        cond = 3
                    else:
                        units[i] = -(1-transc)*cash[i]/true_price[i]
                        cash[i] = cash[i] - units[i]*true_price[i]
                        cond = 3
                    #print('Enter Short from none')
                elif i == 0 and prediction[i] > threshold[0] and prediction[i] < threshold[1]:
                    cond = 1
                else:
                    cash[i] = cash[i-1]
                    units[i] = units[i-1]
                    cond = 1
            #Exiting long position
            elif cond == 2 and prediction[i] < threshold[0]:
                #Exit long
                cash[i] = cash[i-1] + (1-transc)*units[i-1]*true_price[i]
                units[i] = 0
                #print('Exit long')
                #Enter Short
                units[i] = -(1-transc)*cash[i]/true_price[i]
                cash[i] = cash[i] - units[i]*true_price[i]
                cond = 3
                #print('Enter short from long')
            #Exiting short position
            elif cond == 3 and prediction[i] > threshold[1]:
                #Exit short
                cash[i] = cash[i-1] + (1-transc)*units[i-1]*true_price[i]
                units[i] = 0
                #print('Exit Short')
                #Enter long
                units[i] = (1-transc)*cash[i]/true_price[i]
                cash[i] = 0
                cond = 2
                #print('Enter long from short')
            #Holding Condition
            else:
                cash[i] = cash[i-1]
                units[i] = units[i-1]
                #print('Holding')
        else:
            #Entering position
            if cond == 1 and prediction[i] > threshold[1]:
                units[i] = (1-transc)*cash[i]/true_price[i]
                cash[i] = 0
                cond = 2
                #print('Enter')
            #Exiting position
            elif cond == 2 and prediction[i] < threshold[0]:
                cash[i] = (1-transc)*true_price[i]*units[i-1]
                units[i] = 0
                cond = 1
                #print('Exit')
            #Holding Condition
            else:
                cash[i] = cash[i-1]
                units[i] = units[i-1]
                #print('Holding')

    value_over_time = cash + np.multiply(units,true_price) - borrow

    raw_data = {'Portfolio Value':value_over_time, 'Cash': cash, 'Units': units}
    pd.DataFrame(raw_data).to_csv("debug.csv")

    return value_over_time


#Testing
if False:
    price = np.array([100,99,101,102,105,110,115])
    money = 100
    pred = np.array([0.2,0.7,0.5,0.9,0.3,0.1,0.8])
    threshold = np.array([.4,.6])
    leverage = [1,3]

    for i in leverage:
        value_over_time, cash, units = portfolio_generator(principal = money, prediction = pred, true_price = price,
                                                            threshold = threshold, leverage = i, short = True)
        raw_data = {'Portfolio Value':value_over_time, 'Cash': cash, 'Units': units, 'Prediction': pred, 'True Price': price}
        pd.DataFrame(raw_data).to_csv("test"+str(i)+".csv")

if False:
    x = np.arange(90,100,0.01)
    profit_eval(x)
