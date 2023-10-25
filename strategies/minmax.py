import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

class Minmax_strategy:
    
    def __init__(self, ticker, data_period='1y', period_to_buy=89, period_to_sell=13):
        self.ticker = ticker
        self.period = data_period
        self.period_to_buy = period_to_buy
        self.period_to_sell = period_to_sell
    
    def backtester(self, logs=False):
        self.df = self.get_ticker_values()
        
        self.df["Max"] = self.df["High"].rolling(window=self.period_to_buy).max()
        self.df["Min"] = self.df["Low"].rolling(window=self.period_to_sell).min()
        
        max_list = self.df[self.df['Max']<=self.df["High"]]
        min_list = self.df[self.df['Min']>=self.df["Low"]]
        
        sell_date = '2000-01-01'
        profit = 0
        cont = True
        
        while cont:

            try:
                buy_price = max_list[max_list.index>sell_date].iloc[0]["Close"]
                buy_date = max_list[max_list.index>sell_date].index[0]
                
                if logs:
                    print("Bought ", buy_date, " for ", buy_price)
                    
            except:
                break

            try:
                sell_price = min_list[min_list.index>buy_date].iloc[0]["Close"]
                sell_date = min_list[min_list.index>buy_date].index[0]
                profit += (sell_price - buy_price) / buy_price
                
                if logs:
                    print("Sold ", sell_date, " for ", sell_price)

            except:
                profit += (self.df["Close"][-1] - buy_price) / buy_price
                cont = False
        
        classic_profit = self.calculate_classic_profit()    
        
        return profit, classic_profit
    
    def predict(self):
        
        self.df = self.get_ticker_values()
        self.df["Max"] = self.df["High"].rolling(window=self.period_to_buy).max()
        self.df["Min"] = self.df["Low"].rolling(window=self.period_to_sell).min()
        
        if self.df.iloc[-1]["High"] >= self.df.iloc[-1]["Max"]:
            return "Buy"
        elif self.df.iloc[-1]["Low"] <= self.df.iloc[-1]["Min"]:
            return "Sell"
        return "Hold"
         
    
    def get_df(self):
        return self.df
    
        
    def calculate_classic_profit(self):
        classic = (self.df['Close'][-1] - self.df['Open'][0]) / self.df['Open'][0]
        return classic
        
    def get_ticker_values(self):
        df = yf.Ticker(self.ticker).history(period = self.period)
        return df