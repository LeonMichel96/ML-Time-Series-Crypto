# 📈 Time Series Prediction for Cryptocurrencies with Prophet

This project is an interactive tool designed to forecast the future price behavior of cryptocurrencies using time series models. It leverages the public [CoinGecko API](https://www.coingecko.com/en/api), which provides access to up to **one year of historical price data** for the most relevant cryptocurrencies on the market.

### 🧠 How does it work?

- The **user selects** one of the top cryptocurrencies through an intuitive interface.
- The **CoinGecko API** is queried to automatically retrieve the last year of daily historical prices.
- The data is split: **90% is used for training and testing**, and the remaining portion is reserved for evaluation.
- The model used for forecasting is **Facebook Prophet**, due to its ability to capture trends, seasonality, and anomalies in time series data.


Link to web app : https://pq2afeeg7q9dfdvihzvrbd.streamlit.app/
