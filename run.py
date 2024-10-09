import pandas as pd
import os
import torch

from datavis import plotprices
from dataprep import Dataloader
from model import LSTMModel,GRUModel
from train import train_fn,evaluate_fn
import json

stocks = [
    "AAPL",
    "AMD"
]

hist = {}

# Base directory where the CSV files are located
base_dir = "C:/Users/15515/Documents/AN$/Stock Market Forecasting/data"

for stock in stocks:
    file_path = os.path.join(base_dir, f"{stock}.csv")
    hist[stock] = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Define the fixed period
#start_date = "2022-01-01"
#end_date = "2023-12-31"

#hists = {}
#for s in stocks:
#    hists[s] = yf.download(s,start_date,end_date)
#for stock, data in hists.items():
#    print(f"Data for {stock}:")
#    print(data.head())

# Ploting stock prices and displaying candlestick of stocks
plotprices(stocks, hist)

for stock in stocks:
    data_frame = pd.DataFrame(hist[stock].loc[:,"Close"])
    data_obj = Dataloader(data_frame)

    configs = json.load(open('config.json', 'r'))

    train, test = data_obj.train_test_split(configs['data']['trainsize'], data_obj.scaled_data)

    # Create sequences for training and testing
    X_train, y_train = data_obj.create_sequences(train, configs["data"]["seq_length"])
    X_test, y_test = data_obj.create_sequences(test, configs["data"]["seq_length"])

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    X_test_tensor = torch.Tensor(X_test)
    y_test_tensor = torch.Tensor(y_test)

    train_data, valid_data = data_obj.train_valid_split(configs["data"]["validsize"], X_train_tensor, y_train_tensor)
    train_data, valid_data, test_data = data_obj.batches(configs["data"]["batch_size"], train_data, valid_data, X_test_tensor, y_test_tensor)

    # Initialize and train the LSTM and GRU model
    lstm_model = LSTMModel(configs["model"]["input_size"][0], configs["model"]["hidden_size"][0])
    gru_model = GRUModel(configs["model"]["input_size"][1], configs["model"]["hidden_size"][1])

    
    print("LSTM Model")
    lstmmodel = train_fn(lstm_model, configs["model"]["epochs"][0], train_data, valid_data)
    evaluate_fn(stock, configs['model']['type'][0], lstmmodel, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, valid_data, data_obj.scaler)

    print("GRU Model")
    grumodel = train_fn(gru_model, configs["model"]["epochs"][1], train_data, valid_data)
    evaluate_fn(stock, configs['model']['type'][1], grumodel, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, valid_data, data_obj.scaler)















