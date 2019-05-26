import numpy as np
import pandas as pd
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
    return_array = np.zeros(len(portfolio-1))
    for i in range(return_array):
        return_array[i] = portfolio[i+1]/portfolio[i] - 1

    abs_profit = portfolio[len(portfolio-1)]-portfolio[0]
    annual_prof = ((portfolio[len(portfolio-1)]/portfolio[0]) ** (num_trading_day*num_trading_minutes/len(portfolio)) - 1)*100
    sharpe = np.sqrt(num_trading_day*num_trading_minutes)*np.mean(return_array)/np.std(return_array)

    return abs_profit, annual_prof, sharpe

def accuracy(prediction,true_class):

    prediction =np.array(prediction)
    true_class =np.array(true_class)

    pred = np.where(prediction > 0.5, 1, 0)
    accuracy = np.mean(true_class == pred)

    return accuracy

def portfolio_generator(principal,prediction,true_price,threshold, leverage = 1,short = True):

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
    n = len(true_price)
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
    condition 2: excess cash with negative # of shares
    '''
    for i in range(n):
        if short:
            #Entering position
            if cond == 1:
                if prediction[i] > threshold[1]:
                    units[i] = cash[i]/true_price[i]
                    cash[i] = 0
                    cond = 2
                    print('Enter Long from none')
                elif prediction[i] < threshold[0]:
                    units[i] = -cash[i]/true_price[i]
                    cash[i] = cash[i] - units[i]*true_price[i]
                    cond = 3
                    print('Enter Short from none')
            #Exiting long position
            elif cond == 2 and prediction[i] < threshold[0]:
                #Exit long
                cash[i] = cash[i-1] + units[i-1]*true_price[i]
                units[i] = 0
                print('Exit long')
                #Enter Short
                units[i] = -cash[i]/true_price[i]
                cash[i] = cash[i] - units[i]*true_price[i]
                cond = 3
                print('Enter short from long')
            #Exiting short position
            elif cond == 3 and prediction[i] > threshold[1]:
                #Exit short
                cash[i] = cash[i-1] + units[i-1]*true_price[i]
                units[i] = 0
                print('Exit Short')
                #Enter long
                units[i] = cash[i]/true_price[i]
                cash[i] = cash[i] - units[i]*true_price[i]
                cond = 2
                print('Enter long from short')
            #Holding Condition
            else:
                cash[i] = cash[i-1]
                units[i] = units[i-1]
                print('Holding')
        else:
            #Entering position
            if cond == 1 and prediction[i] > threshold[1]:
                units[i] = cash[i]/true_price[i]
                cash[i] = 0
                cond = 2
                print('Enter')
            #Exiting position
            elif cond == 2 and prediction[i] < threshold[0]:
                cash[i] = true_price[i]*units[i-1]
                units[i] = 0
                cond = 1
                print('Exit')
            #Holding Condition
            else:
                cash[i] = cash[i-1]
                units[i] = units[i-1]
                print('Holding')

    value_over_time = cash + units*true_price - borrow

    return value_over_time, cash, units

'''
#Testing
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
'''
