import requests
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import streamlit as st
from plotly import graph_objects as go
import plotly.express as px

# API params
today = int(datetime.datetime.now().timestamp())

# Get data function
@st.cache
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
    df['Date']=pd.to_datetime(df['Date'])
    return df

# Plot raw data function
def plot_raw(df):
    fig = px.scatter(df, x="Date", y="Price", title="Historic data")
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
