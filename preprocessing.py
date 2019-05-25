from util import get_raw_data
import pandas as pd
import numpy as np

def create_sma(price_array, window):
    if len(price_array) < window:
        return None
    sma = np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        sma[i] = np.sum(price_array[i-window:i])/float(window)
    return sma

def create_ema(price_array, sma, window):
    if len(price_array) < window:
        return None
    c = 2./float(window + 1)
    ema = np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        if i == window:
            ema[i] = sma[i]
        else:
            ema[i] = c*(price_array[i] - ema[i-1]) + ema[i-1]
    return ema

def create_mom(price_array, window):
    mom =  np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        mom[i] = price_array[i] - price_array[i-window]
    return mom

def create_macd(price_array, window = [12, 26]):
    sma_12 = create_sma(price_array, window[0])
    sma_26 = create_sma(price_array, window[1])
    ema_12 = create_ema(price_array, sma_12, window[0])
    ema_26 = create_ema(price_array, sma_26, window[1])
    diff_ema = ema_12 - ema_26
    sma_9 = create_sma(diff_ema, window = 9)
    v = create_ema(diff_ema, sma_9, window = 9)
    return diff_ema - v

def create_return(price_array, window):
    output = np.zeros(len(price_array))
    for i in range(window, len(price_array)):
        output[i] = float(price_array[i+1] - price_array[i+1-window])/float(price_array[i+1-window])
        if i+2 == len(price_array): break
    return output

def create_up_down(price_array, window):
    pastUD = np.zeros(len(price_array))
    for i in range(window+1, len(price_array)):
        pastUD[i] = window - 2*np.sum(price_array[i-window:i] < price_array[i-window-1:i-1])
    return pastUD

def create_day_since_cross(cross_array):
    day_since_cross = np.zeros(len(cross_array))
    num = 0
    for i in range(len(cross_array)):
        if cross_array[i] == 0:
            num += 1
        else:
            num = 0
        day_since_cross[i] = num
    return day_since_cross

def create_macd_cross(macd):
    macd_cross = np.zeros(len(macd))
    for i in range(1, len(macd)):
        if macd[i-1] < 0 and macd[i] > 0:
            macd_cross[i] = 1
        elif macd[i-1] > 0 and macd[i] < 0:
            macd_cross[i] = -1
        else:
            macd_cross[i] = 0
    return macd_cross

def create_ma_cross(ma, price_array):
    ma_cross = np.zeros(len(ma))
    for i in range(1, len(ma)):
        if ma[i-1] < price_array[i-1] and ma[i] > price_array[i]:
            ma_cross[i] = 1
        elif ma[i-1] > price_array[i-1] and ma[i] < price_array[i]:
            ma_cross[i] = -1
        else:
            ma_cross[i] = 0
    return ma_cross

def create_class(price_array):
    output = np.zeros(len(price_array))
    for i in range(len(price_array)):
        if price_array[i+1] > price_array[i]:
            output[i] = 1
        if i+2 == len(price_array): break
    return output


def main():

    #df = data[['Date','Settle', 'Volume']]
    data = get_raw_data()

    df = data

    window_sma = [5, 10, 15, 20, 50, 100, 200]
    window_ema = [10, 12, 20, 26, 50, 100, 200]

    price_val = np.array(df['average'])
    time_val = np.array(df['date'])
    daily_return = create_class(price_val)

    sma_map = {}
    ema_map = {}
    mom_map = {}
    sma_cross_map = {}
    ema_cross_map = {}
    up_down_map = {}
    for k, l in zip(window_sma, window_ema):
        sma_map["SMA" + str(k)] = create_sma(price_val, k)
        sma_map["SMA" + str(l)] = create_sma(price_val, l)
        ema_map["EMA" + str(l)] = create_ema(price_val, sma_map["SMA" + str(l)], l)
        mom_map["MOM" + str(k)] = create_mom(price_val, k)
        sma_cross_map["SMA_CROSS" + str(k)] = create_ma_cross(sma_map["SMA" + str(k)], price_val)
        ema_cross_map["EMA_CROSS" + str(l)] = create_ma_cross(ema_map["EMA" + str(l)], price_val)
        up_down_map["Up-Down" + str(k)] = create_up_down(price_val, l)

    macd_val = create_macd(price_val)
    macd_cross = create_macd_cross(macd_val)

    day_since_cross_map = {}
    for m,l in zip(sma_cross_map.keys(),ema_cross_map.keys()):
        day_since_cross_map["Day_Since_" + str(m)] = create_day_since_cross(sma_cross_map[m])
        day_since_cross_map["Day_Since_" + str(l)] = create_day_since_cross(ema_cross_map[l])

    raw_data = {'Date':time_val, 'Price': price_val, 'Minute':np.array(df['minute']),
    'Class': daily_return, 'Volume': np.array(df['volume']),'SMA5' : sma_map["SMA5"],
    'SMA10' : sma_map["SMA10"], 'SMA15' : sma_map["SMA15"], 'SMA20' : sma_map["SMA20"],
    'SMA50' : sma_map["SMA50"], 'SMA100' : sma_map["SMA100"], 'SMA200' : sma_map["SMA200"],
    'EMA10' : ema_map["EMA10"], 'EMA12' : ema_map["EMA12"], 'EMA20' : ema_map["EMA20"],
    'EMA26' : ema_map["EMA26"], 'EMA50' : ema_map["EMA50"], 'EMA100' : ema_map["EMA100"],
    'EMA200' : ema_map["EMA200"], 'MACD' : macd_val, 'MACD_Cross' : macd_cross,
    'SMA5Cross' : sma_cross_map["SMA_CROSS5"], 'SMA10Cross' : sma_cross_map["SMA_CROSS10"],
    'SMA15Cross' : sma_cross_map["SMA_CROSS15"], 'SMA20Cross' : sma_cross_map["SMA_CROSS20"],
    'SMA50Cross' : sma_cross_map["SMA_CROSS50"], 'SMA100Cross' : sma_cross_map["SMA_CROSS100"],
    'EMA12Cross' : ema_cross_map["EMA_CROSS12"], 'EMA10Cross' : ema_cross_map["EMA_CROSS10"],
    'EMA20Cross' : ema_cross_map["EMA_CROSS20"], 'EMA26Cross' : ema_cross_map["EMA_CROSS26"],
    'EMA50Cross' : ema_cross_map["EMA_CROSS50"], 'EMA100Cross' : ema_cross_map["EMA_CROSS100"],
    'SMA200Cross' : sma_cross_map["SMA_CROSS200"], 'EMA200Cross' : ema_cross_map["EMA_CROSS200"],
    'Up-Down5' : up_down_map["Up-Down5"],'Up-Down10' : up_down_map["Up-Down10"], 'Up-Down15' : up_down_map["Up-Down15"],
    'Up-Down20' : up_down_map["Up-Down20"],'Up-Down50' : up_down_map["Up-Down50"], 'Up-Down100' : up_down_map["Up-Down100"],
    'Day_Since_SMA5Cross' : day_since_cross_map["Day_Since_SMA_CROSS5"], 'Day_Since_SMA10Cross' : day_since_cross_map["Day_Since_SMA_CROSS10"],
    'Day_Since_SMA15Cross' : day_since_cross_map["Day_Since_SMA_CROSS15"], 'Day_Since_SMA20Cross' : day_since_cross_map["Day_Since_SMA_CROSS20"],
    'Day_Since_SMA50Cross' : day_since_cross_map["Day_Since_SMA_CROSS50"], 'Day_Since_SMA100Cross' : day_since_cross_map["Day_Since_SMA_CROSS100"],
    'Day_Since_EMA12Cross' : day_since_cross_map["Day_Since_EMA_CROSS12"], 'Day_Since_EMA10Cross' : day_since_cross_map["Day_Since_EMA_CROSS10"],
    'Day_Since_EMA20Cross' : day_since_cross_map["Day_Since_EMA_CROSS20"], 'Day_Since_EMA26Cross' : day_since_cross_map["Day_Since_EMA_CROSS26"],
    'Day_Since_EMA50Cross' : day_since_cross_map["Day_Since_EMA_CROSS50"], 'Day_Since_EMA100Cross' : day_since_cross_map["Day_Since_EMA_CROSS100"]
    }

    data = pd.DataFrame(raw_data)
    data[200:len(price_val)].to_csv("spy1min.csv")

if __name__ == "__main__":
    main()
