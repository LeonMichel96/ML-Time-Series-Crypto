import requests
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import streamlit as st
from plotly import graph_objects as go
import plotly.express as px
from prophet.plot import plot_plotly

# API params
today = int(datetime.datetime.now().timestamp())

# Get data function
@st.cache_data
def get_data(url, params):
    ''' Returns a df from the coingecko API, url and params will be input from the user'''
    response = requests.get(url, params=params).json()
    df = pd.DataFrame(response["prices"], columns=["Timestamp", "Price"])
    return df

# Cleaning function
def cleaning(df):
    '''Returns a clean df changing timestamp format to standard yyyy-mm-dd'''
    df['Date']= df['Timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime("%Y-%m-%d"))
    df['Timestamp'] = df['Date']
    df.drop(columns=['Date'], inplace=True)
    df.rename(columns={'Timestamp': 'Date'}, inplace=True)
    df['Date']=pd.to_datetime(df['Date'], format="%Y-%m-%d")
    return df

# Plot raw data function
def plot_raw(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines+markers',
        name='Price'
    ))

    # Add a slider to modify the x-axis (date)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True,
            ),
            type="date"
        )
    )
    return fig

# Train & Test split
def train_test(df):
    '''Returns a train and test df (the train percentage will be 90% of the original data) in the needed format for prophet'''
    df.rename(columns={'Date': 'ds',
                  'Price': 'y'}, inplace=True)

    train_index = int(0.9*len(df))
    train = df.iloc[:train_index]
    test = df.iloc[train_index:]
    return train, test

# Instantiate and train model
def model_forecast(train_df, test_df):
    '''Returns a trained Prophet model based on a train dataframe provided and a forecast dataframe of one year'''
    model = Prophet()
    model.fit(train_df)
    future = model.make_future_dataframe(periods=(365+len(test_df)))
    forecast_original = model.predict(future)
    forecast_df = forecast_original[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()

    forecast_df.rename(columns={'ds':'Date',
                                'yhat':'Forecast',
                                'yhat_lower':'Lower Limit',
                                'yhat_upper':'Upper Limit'}, inplace=True)
    return model, forecast_df, forecast_original

# Plot Forecast
def plot_forecast(train, test, forecast):
    fig = go.Figure()

    # train trace
    fig.add_trace(go.Scatter(
        x=train['ds'],
        y=train['y'],
        mode= 'markers',
        name= 'Train Set'
    ))

    # test trace
    fig.add_trace(go.Scatter(
        x=test['ds'],
        y=test['y'],
        mode= 'markers',
        name= 'Test Set'
    ))

    # forecast trace
    fig.add_trace(go.Scatter(
        x=forecast['Date'],
        y=forecast['Forecast'],
        mode= 'lines',
        name= 'Forecast'
    ))

    # Upper bound
    fig.add_trace(go.Scatter(
        x=forecast['Date'],
        y=forecast['Upper Limit'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),  # invisible line
        showlegend=False
    ))

#   Lower bound
    fig.add_trace(go.Scatter(
        x=forecast['Date'],
        y=forecast['Lower Limit'],
        mode='lines',
        name='Confidence Interval',
        fill='tonexty',  # fill area between this and last trace
        fillcolor='rgba(19, 113, 255, 0.15)',
        line=dict(width=0),  # invisible line
    ))

    fig.update_layout(
        title = 'Forecast vs Actuals',
        xaxis=dict(
            rangeslider=dict(
                visible=True,
            ),
            type="date"
        )
    )

    return fig
