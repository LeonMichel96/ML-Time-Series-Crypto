from utils import today
from utils import get_data
from utils import cleaning
from utils import plot_raw
from utils import train_test
from utils import model_forecast
from utils import plot_forecast
import streamlit as st
import requests
import datetime
from plotly import graph_objects as go

st.markdown("""
    # Cryptocurrency Prediction App
    ## Introduction
    This app intends to deliver a prediction in time of prices (one year in the future) for a selected Layer One cryptocurrency using a Machine Learning approach,
    more specific, this app is using Facebook prediciton model for time series 'Prophet'. We are also using coingecko public API to get the data.
""")
st.write("  ")

# User inputs
st.subheader('User Input')
currencies = ('bitcoin', 'ethereum', 'solana', 'polkadot', 'binancecoin')
with st.form('Inputs'):
    coin = st.selectbox('''First select the cryptocurrency you want to predict (Note: the name is the coingecko API id for it, it may not
                match the official cryptocurrency or L1 name''', currencies)

    st.write("  ")

    years = st.slider('''Select the number of years in the past you want to retrieve of price history, this will be used
                        by the model to train itself and learn :''',1,1)

    submitted = st.form_submit_button("Submit")

# Complete params and data needed for API call
if submitted:
    url = f'https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range'

    past_date = int((datetime.datetime.now() - datetime.timedelta(days=365 * years)).timestamp())

    params = {'vs_currency': 'usd',
                'from': past_date,
                'to': today,
                'precision': 2}

    df_original = get_data(url, params)
    df = cleaning(df_original)

# Show last values and plot historic data
    st.markdown(f"""
                ## Raw data

                Cryptocurrency {coin} values in the last 15 days and data plot over time.
                        """)

    st.write(df.tail(15))
    raw_chart = plot_raw(df)
    st.plotly_chart(raw_chart)

# Show Forecast values and plot actuals (train and test) vs forecast
    st.markdown(f"""
        ## Forecast Data

        Cryptocurrency {coin} forecast values for the following 15 days

                """)

    train, test = train_test(df)
    model, forecast, forecast_original = model_forecast(train, test)

    future_index = len(train)+len(test)
    st.write(forecast.iloc[future_index:future_index+15])

    st.write("  ")

    st.write('''Forecast vs Actuals plot: this graph contains real values (of train and test set), and forecast values,
         tou can use the test set to compare real values and dorecast values of the model.''')

    forecast_chart = plot_forecast(model, forecast_original)
    forecast_chart.update_layout(showlegend= True, legend=dict(traceorder='reversed'))
    forecast_chart.data[0].name = 'Train Set (Actual Values)'
    forecast_chart.data[3].name = 'Upper Bound'
    forecast_chart.data[1].name = 'Lower Bound'
    test_trace = go.Scatter(x=test['ds'], y=test['y'], mode='markers', name='Test Set (Actual Values)')
    forecast_chart.add_trace(test_trace)


    st.plotly_chart(forecast_chart)
