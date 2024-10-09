import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plotprices(stocks, hist):
    plt.figure(figsize=(14, 7))

    for stock in stocks:
        temp_df = hist[stock]
        plt.plot(temp_df.index, temp_df['Close'], label=stock)

    plt.title('Stock Closing Prices Over the Last 2 Years')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Candlestick chart of all the stocks
    for stock in stocks:
        temp_df = hist[stock]
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=temp_df.index,
                    open=temp_df["Open"],
                    high=temp_df["High"],
                    low=temp_df["Low"],
                    close=temp_df["Close"],
                )
            ]
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
            height=300,
            paper_bgcolor="LightSteelBlue",
            title=stock,
        )
        fig.show()