import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data():
    df = pd.read_csv("data/dataset.csv")
    df.rename(columns={'Temp': 'load'}, inplace=True)

    data = df['load'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, scaler

def create_sequences(data, seq_length):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)