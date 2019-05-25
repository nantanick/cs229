import numpy as np

def profit_eval(principal,prediction,true_price,threshold):
    units = 0.0 #number of shares owned
    value_over_time = [] #portfolio value over time
    money = principal
    for i in range(len(prediction)):
        if prediction[i] > threshold:
            unit = money/true_price[i]
    
    abs_profit = 0
    annual_prof = 0
    volatility = 0
    sharpe = 0

    return abs_profit, annual_prof, volatility, sharpe

def f1score(prediction,true_class):
    score = 0
    return score

def accuracy(prediction,true_class):

    prediction =np.array(prediction)
    true_class =np.array(true_class)

    pred = np.where(prediction > 0.5, 1, 0)
    accuracy = np.mean(true_class == pred)

    return accuracy
