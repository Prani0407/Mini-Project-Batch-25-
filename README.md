# Stock Market Forecasting using LSTM and GRU
## About the Project
This project involves predicting the prices of various stocks using historical stock price data. It utilizes deep learning models — **LSTM** (Long Short-Term Memory) and **GRU** (Gated Recurrent Unit)—both of which are variations of **RNN** (Recurrent Neural Network).

### Objectives
- **Visualization**: Visualizing stock prices over a period of time using candlestick charts and line graphs.
- **Data Collection and Preprocessing**: Collecting stock closing prices and preprocessing them for the purpose of training the models.
- **Model Implementation**: Implementing the architectures of both LSTM and GRU models.
- **Training and Evaluation**: Training the models and evaluating predicted stock prices using various metrics.
- **Results**: Visualizing the results via graphs and saving the predicted vs. actual stock prices for comparison.

## Usage
Go to the project directory and install the dependencies using the command:
```
pip install -r requirements.txt
```
To run the code, execute the following command:
```
python run.py
```

## Dataset
The dataset used in this project was downloaded from **Yahoo Finance**. It contains the following columns - **Date, Open, High, Low, Close, Adj Close, Volume**. Only the **Close** stock price was considered for forecasting.The **AAPL** and **AMD** stocks were used in this project. The stock prices of AAPL were downloaded for two years (2022-2023) and stock prices of AMD were downloaded for approximately seven years (2017-2023).

## Description
The files `datavis.py` and `dataprep.py` are used for visualizing the stock prices using various plots and preparing the stock prices data for training the models. 
The stock prices are normalized using the **MinMaxScaler**, and sequences of prices are created for training.

The models **LSTM** and **GRU** are defined in the file `model.py`. 
The models are then trained, and the results are evaluated based on various metrics like **RMSE**, **MSE**, **MAE**, **R<sup>2</sup>** in the `train.py` file. 

Various graphs are also plotted to better understand the results. Graphs of results for AAPL stock can be found in the `results_graphs` folder. 
The **actual vs predicted price values** of the test data are stored in the CSV files.

It was observed that the GRU model performed better in capturing the short-term changes in stock prices compared to LSTM.

## Future Work
While this project provides a strong foundation for stock price prediction, few of the potential areas for future work include:
- Using additional features like - Trading Volume, Low and High stock prices or Sentiment value of stock news.
- Exploring and experimenting with more advanced deep learning architectures like Transformers with attention mechanisms or hybrid models.
