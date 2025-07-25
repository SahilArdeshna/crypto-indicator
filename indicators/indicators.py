import os
import pandas as pd
import requests
from datetime import datetime
import logging
import numpy as np

def get_crypto_symbols_from_env():
    cryptos = os.getenv("CRYPTOS", "BTC").split(",")
    return [c.strip().upper() + "USDT" if not c.strip().upper().endswith("USDT") else c.strip().upper() for c in cryptos]

def get_interval_from_env():
    return os.getenv("INTERVAL", "4h")

def fetch_ohlcv(symbol, interval=None, limit=100):
    if interval is None:
        interval = get_interval_from_env()
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        res = requests.get(url).json()
        df = pd.DataFrame(res, columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df[["open", "high", "low", "close", "volume"]]
    except Exception as e:
        logging.error(f"Error fetching {symbol} OHLCV from Binance: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI manually to avoid pandas_ta issues"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(prices, period=20):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_ema(prices, period=50):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period).mean()

def get_indicators(symbol, interval=None, limit=100):
    if interval is None:
        interval = get_interval_from_env()
    try:
        df = fetch_ohlcv(symbol, interval, limit)
        if df is None:
            logging.warning(f"Failed to fetch {symbol} OHLCV data. Indicators cannot be calculated.")
            return None
        
        # Calculate indicators manually to avoid pandas_ta bugs
        df['RSI_14'] = calculate_rsi(df['close'], 14)
        df['SMA_20'] = calculate_sma(df['close'], 20)
        df['EMA_50'] = calculate_ema(df['close'], 50)
        
        # Simple VWAP calculation
        df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Simple Bollinger Bands
        sma_20 = df['SMA_20']
        std_20 = df['close'].rolling(window=20).std()
        df['BBU_20_2.0'] = sma_20 + (std_20 * 2)
        df['BBL_20_2.0'] = sma_20 - (std_20 * 2)
        
        # MACD calculation
        ema_12 = calculate_ema(df['close'], 12)
        ema_26 = calculate_ema(df['close'], 26)
        df['MACD_12_26_9'] = ema_12 - ema_26
        df['MACDs_12_26_9'] = calculate_ema(df['MACD_12_26_9'], 9)
        
        # Simple Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['STOCHk_14_3_3'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(window=3).mean()
        
        # Simple ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATRr_14'] = tr.rolling(window=14).mean()
        
        # Simple OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        # Simple ADX approximation (trend strength)
        df['ADX_14'] = df['close'].rolling(window=14).std() / df['close'].rolling(window=14).mean() * 100

        latest = df.iloc[-1]

        def classify(value, indicator):
            if pd.isna(value):
                return "Neutral"
            if indicator == "RSI":
                return "Bullish" if value < 70 and value > 50 else "Bearish" if value < 30 else "Neutral"
            elif indicator == "MACD":
                return "Bullish" if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] else "Bearish"
            elif indicator == "Stoch":
                return "Bullish" if latest['STOCHk_14_3_3'] > latest['STOCHd_14_3_3'] else "Bearish"
            elif indicator == "ADX":
                return "Strong" if value > 25 else "Weak"
            else:
                return "Neutral"

        indicators = {
            "RSI": classify(latest['RSI_14'], "RSI"),
            "MACD": classify(None, "MACD"),
            "Stochastic": classify(None, "Stoch"),
            "BB": "Bullish" if 'BBL_20_2.0' in latest and not pd.isna(latest['BBL_20_2.0']) and latest['close'] > latest['BBL_20_2.0'] else "Neutral",
            "SMA20": "Bullish" if not pd.isna(latest['SMA_20']) and latest['close'] > latest['SMA_20'] else "Bearish",
            "EMA50": "Bullish" if not pd.isna(latest['EMA_50']) and latest['close'] > latest['EMA_50'] else "Bearish",
            "VWAP": "Bullish" if not pd.isna(latest['VWAP']) and latest['close'] > latest['VWAP'] else "Bearish",
            "ADX": classify(latest['ADX_14'], "ADX"),
            "ATR": f"{latest['ATRr_14']:.2f}" if not pd.isna(latest['ATRr_14']) else "N/A",
            "OBV": f"{latest['OBV']:.2f}" if not pd.isna(latest['OBV']) else "N/A"
        }
        return indicators
    except Exception as e:
        logging.error(f"Error calculating indicators for {symbol}: {e}")
        return None
