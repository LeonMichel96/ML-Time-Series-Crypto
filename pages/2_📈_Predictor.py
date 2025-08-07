from utils import today
from utils import get_data
from utils import cleaning
from utils import plot_raw
from utils import train_test
from utils import model_forecast
from utils import plot_forecast

import numpy as np
import streamlit as st
import requests
import datetime
from plotly import graph_objects as go

### Page
st.header('Predictor 📈')

## User inputs
st.subheader('User Input')
currencies = ('bitcoin', 'ethereum', 'solana', 'polkadot', 'binancecoin')

st.info("""
        Data Source is coingecko api, which allows to fetch free historical prices one year in the past.

        - The model will use as train set 90% of this historical data.
        - The model will predict the test set + 365 days in the future.
        - The user can select how many historic and future data want to see in the tables shown below.
        """)

with st.form('Inputs'):
    coin = st.selectbox('Select the cryptocurrency you want to predict ', currencies)
    last_days = st.number_input(label='How many days of historic data you want to see? (in the table)', min_value=1, max_value=365, value=15, key='days')
    future_days = st.number_input(label='How many days of future data you want to see? (in the table)', min_value=1, max_value=365, value=15, key='future_days')
    submitted = st.form_submit_button("Submit")

# Complete params and data needed for API call
if submitted:
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range'

    past_date = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp())

    params = {'vs_currency': 'usd',
                'from': past_date,
                'to': today,
                'precision': 2}

## API call and cleaning
    df_original = get_data(url, params)
    df = cleaning(df_original)

## Show last values and plot historic data
    st.markdown(f"""
    ## Raw data
    Cryptocurrency {coin} values in the last {last_days} days and data plot over time.
    """)

    st.dataframe(df.tail(last_days))
    raw_chart = plot_raw(df)
    st.plotly_chart(raw_chart)

## Show Forecast values and plot actuals (train and test) vs forecast
    st.markdown(f"""
        ## Forecast Data

        Cryptocurrency {coin} forecast values for the following {future_days} days
    """)

    train, test = train_test(df)
    model, forecast, forecast_original = model_forecast(train, test)

    future_index = len(train)+len(test)

    st.dataframe(forecast.iloc[future_index:future_index+future_days])

    st.write('''
        ## Forecast Plot

        Forecast vs Actuals plot: this graph contains real values (of train and test set), and forecast values,
        you can use the test set to compare real values and dorecast values of the model.''')




    forecast_fig = plot_forecast(train, test, forecast)

    st.plotly_chart(forecast_fig)
