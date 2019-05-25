import pandas as pd
import glob

def get_data():
    data = []
    for f in sorted(glob.glob('./data/*.json')):
        temp = pd.read_json(f)
        data.append(temp)
    return pd.concat(data)


if __name__ == '__main__':
    print(get_data().head())
