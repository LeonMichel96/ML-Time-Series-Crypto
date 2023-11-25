from utils import today
from utils import get_data
from utils import cleaning
from utils import plot_raw
import streamlit as st
import requests
import datetime

st.markdown("""
    # Cryptocurrency Prediction App
    ## Introduction
    This project tends to deliver a prediction in time for a selected layer one cryptocurrency using a Machine Learning approach,
    more specific, this app is using Facebook prediciton model for time series 'Prophet'. We are also using coingecko public API to get the data.
""")
st.write("  ")

# User inputs
st.subheader('User Input')
currencies = ('bitcoin', 'ethereum', 'solana', 'polkadot', 'binancecoin')
coin = st.selectbox('''First select the cryptocurrency you want to predict (Note: the name is the coingecko API id for it, it may not
             match the official cryptocurrency or L1 name''', currencies)
st.write("  ")

years = st.slider('''Select the number of years in the past you want to retrieve of price history, this will be used
                     by the model to train itself and learn :''',1,4)

# Complete params and data needed for API call
url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range'

past_date = int((datetime.datetime.now() - datetime.timedelta(days=365 * years)).timestamp())

params = {'vs_currency': 'usd',
          'from': past_date,
          'to': today,
          'precision': 2}

df_original = get_data(url, params)
df = cleaning(df_original)

st.markdown("""
    ## Raw data

      Cryptocurrency values in the last 10 days and data plot over time.
            """)

st.write(df.tail(10))
raw_chart = plot_raw(df)
#st.text(print(type(raw_chart)))
st.plotly_chart(raw_chart)
