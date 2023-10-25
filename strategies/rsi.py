import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress

    
class RSI_strategy:
    def __init__(self, ticker, rsi_top_border=60, rsi_bot_border=40, period='1y', rsi_period=14):
        self.ticker = ticker
        self.period = period
        self.rsi_top_border = rsi_top_border
        self.rsi_bot_border = rsi_bot_border
    
    def get_ticker_signals(self, period, distance):
        
        self.df = self.get_ticker_values()[-(period + distance):]
        self.rsi = self.rsi_14_days()
        #self.rsi = self.calculate_rsi()
        self.MACD, self.signal = self.calculate_macd_and_signal()        
        self.hist = self.MACD - self.signal 
        signals = pd.DataFrame()
        
        self.df["Rolling Price"] = self.df["Close"].rolling(window=distance).mean()

        
        for x in range(1, period+1):
            
            rsi_break_lower = True if (self.df['RSI'][-x] >= self.rsi_bot_border) and (self.df['RSI'][-x-1] < self.rsi_bot_border) else False
            rsi_break_top = True if (self.df['RSI'][-x] <= self.rsi_top_border) and (self.df['RSI'][-x-1] > self.rsi_top_border) else False
            rsi_over_mid = True if self.df['RSI'][-x] > 50 else False
            macd_positive = True if self.MACD[-x] > 0 else False
            macd_stronger_signal = True if self.signal[-x] <= self.MACD[-x] else False
            intersection_buy = True if self.signal[-x] < self.MACD[-x] and self.signal[-x-1] > self.MACD[-x-1] else False
            intersection_sell = True if self.signal[-x] > self.MACD[-x] and self.signal[-x-1] < self.MACD[-x-1] else False
            hist_axis = "Positive" if self.hist[-x] > 0 else "Negative"
            hist_trend = "Positive" if self.hist[-x] > self.hist[-x-1] else "Negative"
            over_rolling_price = True if self.df["Close"][-x] >= self.df["Rolling Price"][-x] else False
            
            signals = signals.append({"Date": self.df.index[-x], "Ticker": self.ticker, "Over rolling price": over_rolling_price, "Break RSI bot": rsi_break_lower, "Break RSI top": rsi_break_top, "RSI OVER 50": rsi_over_mid, "MACD positive": macd_positive, "MACD > SIGNAL": macd_stronger_signal, "MACD intersection over SIGNAL": intersection_buy, "SIGNAL intersection over MACD": intersection_sell, "Histogram axis": hist_axis, "Histogram trend": hist_trend}, ignore_index=True)
                
        return signals
    
    def predict_operations(self, signals):
        
        operations = pd.DataFrame()
        
        trend = self.build_trend_line(5)
        
        for x in range(0, len(signals)):
            if signals["Over rolling price"][x] == True and signals["MACD > SIGNAL"][x] == True and signals["RSI OVER 50"][x] == True and signals["Histogram axis"][x] == "Positive" and signals["Histogram trend"][x] == "Positive":
                operations = operations.append({"Date": signals["Date"][x], "Ticker": signals["Ticker"][x], "Recommendation": "Buy", "Base Line": self.MACD[-(x+1)], "Signal Line": self.signal[-(x+1)], "RSI": self.df["RSI"][-(x+1)], "Histogram": self.hist[-(x+1)], "Trend line slope": trend}, ignore_index=True)
            elif signals["Over rolling price"][x] == False and signals["MACD > SIGNAL"][x] == False and signals["RSI OVER 50"][x] == False and signals["Histogram axis"][x] == "Negative":
                operations = operations.append({"Date": signals["Date"][x], "Ticker": signals["Ticker"][x], "Recommendation": "Sell", "Base Line": self.MACD[-(x+1)], "Signal Line": self.signal[-(x+1)], "RSI": self.df["RSI"][-(x+1)], "Histogram": self.hist[-(x+1)], "Trend line slope": trend}, ignore_index=True)
        return operations
    
    def build_trend_line(self, period):
        result = linregress(range(len(self.df[-period:])), self.df['Close'][-period:])
        return result.slope
    
    def strategy(self):
        self.df = self.get_ticker_values()
        self.rsi = self.rsi_14_days(self.df)
        self.MACD, self.signal = self.calculate_macd_and_signal()
        self.operations = pd.DataFrame()
        
        for x in range(15, len(self.df)):

            # to buy asset
            if (self.df['RSI'][x] >= self.rsi_bot_border) and (self.df['RSI'][x-1] < self.rsi_bot_border) and  (self.MACD[x] > 0 or self.signal[x] <= self.MACD[x]):
                    self.operations = self.operations.append({"Date": self.df.index[x], "Ticker": self.ticker, "RSI": self.df["RSI"][x], "MACD": self.MACD[x], "Signal": self.signal[x], "Operation": "Buy", "Recommendation": self.recomendation}, ignore_index=True)
                    

            # to sell asset
            elif ((self.df['RSI'][x] <= self.rsi_top_border) & (self.df['RSI'][x-1] > self.rsi_top_border)) and (self.signal[x] > self.MACD[x] or self.MACD[x]<0):
                    self.operations = self.operations.append({"Date": self.df.index[x], "Ticker": self.ticker, "RSI": self.df["RSI"][x], "MACD": self.MACD[x], "Signal": self.signal[x], "Operation": "Sell", "Recommendation": self.recomendation}, ignore_index=True)
        
    def backtester(self, period, logs=False):
        signals = self.get_ticker_signals(period, 50)
        operations = self.predict_operations(signals)

        state = "to_buy"
        profit = 0
        
        classic_profit = (self.df[self.df.index==operations["Date"][0]]["Close"].values[0] - self.df[self.df.index==operations["Date"][len(operations)-1]]["Close"].values[0]) / self.df[self.df.index==operations["Date"][len(operations)-1]]["Close"].values[0]
        
        for i in range(len(operations)-1, 0, -1):
            if state == "to_buy":
                if operations["Recommendation"][i] == "Buy":
                    state = "to_sell"
                    bought_price = self.df[self.df.index==operations["Date"][i]]["Close"].values[0]
                    
                    if logs:
                        print("Bought ", operations["Date"][i], " for ", bought_price)
            else:
                if operations["Recommendation"][i] == "Sell":
                    state = "to_buy"
                    sold_price = self.df[self.df.index==operations["Date"][i]]["Close"].values[0]
                    profit += (sold_price - bought_price) / bought_price
                    
                    if logs:
                        print("Sold ", operations["Date"][i], " for ", sold_price)
                    
        return profit, classic_profit

    def predict(self):
        signals = self.get_ticker_signals(1, 50)
        operations = self.predict_operations(signals)
        return operations
    
    def return_results(self):
        return self.ticker, self.classic_profit, self.profit
    
    def get_rsi(self):
        return self.rsi
    
    def get_macd(self):
        return self.MACD 
    
    def get_signal(self):
        return self.signal
    
    def get_df(self):
        return self.df
    
    def create_graph(self):

        # Create a plot with multiple lines
        fig = go.Figure()

        # Add the first line
        fig.add_trace(go.Scatter(x=self.MACD.index, y=self.MACD, mode='lines', name='MACD'))
        fig.add_trace(go.Scatter(x=self.signal.index, y=self.signal, mode='lines', name='Signal'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['RSI'], mode='lines', name='RSI'))

        # Customize the layout
        fig.update_layout(title='RSI + MACD + Signal')
        fig.update_layout(hovermode='x unified')

        # Show the interactive plot
        fig.show()
    
    def create_graph_v2(self):

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        # Add the first line
        
        fig.add_trace(go.Scatter(x=self.MACD.index, y=self.MACD, mode='lines', name='Main'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.signal.index, y=self.signal, mode='lines', name='Signal'), row=3, col=1)
        
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Close'], mode='lines', name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df["Rolling Price"], mode='lines', name='Rolling Price'), row=1, col=1)
        
        
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['RSI'], mode='lines', name='RSI'), row=2, col=1)
        fig.add_shape(type='line', x0=min(self.signal.index), x1=max(self.signal.index), y0=self.rsi_top_border, y1=self.rsi_top_border,
              line=dict(color='black', width=1), row=2, col=1)
        fig.add_shape(type='line', x0=min(self.signal.index), x1=max(self.signal.index), y0=self.rsi_bot_border, y1=self.rsi_bot_border,
              line=dict(color='black', width=1), row=2, col=1)
        
        colors = ['green' if val >= 0 else 'red' for val in self.hist]
        fig.add_trace(go.Bar(x=self.hist.index, y=self.hist, name="Hist", marker_color=colors), row=3, col=1)
        # Customize the layout
        fig.update_layout(title='RSI + MACD + Signal')
        fig.update_layout(hovermode='x unified')
        fig.update_layout(height=1200, width=1000)

        # Show the interactive plot
        fig.show()
    
    
    def get_ticker_values(self):
        df = yf.Ticker(self.ticker).history(period = self.period)
        return df
    
    def calculate_classic_profit(self):
        classic = (self.df['Close'][-1] - self.df['Open'][0]) / self.df['Open'][0]
        return classic
   
    def calculate_rsi(self, period=14):
        # Calculate daily price changes
        self.df['Price Change'] = self.df['Close'].diff()

        # Calculate gains and losses
        self.df['Gain'] = self.df['Price Change'].apply(lambda x: x if x > 0 else 0)
        self.df['Loss'] = self.df['Price Change'].apply(lambda x: -x if x < 0 else 0)

        # Calculate the average gain and average loss
        avg_gain = self.df['Gain'].rolling(window=period).mean()
        avg_loss = self.df['Loss'].rolling(window=period).mean()

        # Calculate the relative strength (RS)
        rs = avg_gain / avg_loss

        # Calculate the RSI
        self.df['RSI'] = 100 - (100 / (1 + rs))
        #return rsi
    
    def rsi_14_days(self):
        ## 14_Day RSI
        self.df['Up Move'] = np.nan
        self.df['Down Move'] = np.nan
        self.df['Average Up'] = np.nan
        self.df['Average Down'] = np.nan

        self.df['Adj Close'] = self.df['Close']
        # Relative Strength
        self.df['RS'] = np.nan
        # Relative Strength Index
        self.df['RSI'] = np.nan
        ## Calculate Up Move & Down Move
        for x in range(1, len(self.df)):
            self.df['Up Move'][x] = 0
            self.df['Down Move'][x] = 0

            if self.df['Adj Close'][x] > self.df['Adj Close'][x-1]:
                self.df['Up Move'][x] = self.df['Adj Close'][x] - self.df['Adj Close'][x-1]

            if self.df['Adj Close'][x] < self.df['Adj Close'][x-1]:
                self.df['Down Move'][x] = abs(self.df['Adj Close'][x] - self.df['Adj Close'][x-1])  

        ## Calculate initial Average Up & Down, RS and RSI
        self.df['Average Up'][14] = self.df['Up Move'][1:15].mean()
        self.df['Average Down'][14] = self.df['Down Move'][1:15].mean()
        self.df['RS'][14] = self.df['Average Up'][14] / self.df['Average Down'][14]
        self.df['RSI'][14] = 100 - (100/(1+self.df['RS'][14]))
        ## Calculate rest of Average Up, Average Down, RS, RSI
        for x in range(15, len(self.df)):
            self.df['Average Up'][x] = (self.df['Average Up'][x-1]*13+self.df['Up Move'][x])/14
            self.df['Average Down'][x] = (self.df['Average Down'][x-1]*13+self.df['Down Move'][x])/14
            self.df['RS'][x] = self.df['Average Up'][x] / self.df['Average Down'][x]
            self.df['RSI'][x] = 100 - (100/(1+self.df['RS'][x]))
            
        return self.df
    
    def calculate_macd_and_signal(self, short=12, long=26, signal_v=9):
        ## Calculate the MACD and Signal Line indicators
        ## Calculate the Short Term Exponential Moving Average
        ShortEMA = self.df.Close.ewm(span=short, adjust=False).mean() 
        ## Calculate the Long Term Exponential Moving Average
        LongEMA = self.df.Close.ewm(span=long, adjust=False).mean() 
        ## Calculate the Moving Average Convergence/Divergence (MACD)
        MACD = ShortEMA - LongEMA
        ## Calcualte the signal line
        signal = MACD.ewm(span=signal_v, adjust=False).mean()
        
        return MACD, signal
    
     