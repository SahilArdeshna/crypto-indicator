import os
import pandas as pd
import pandas_ta as ta
import requests
import logging

def get_crypto_symbols_from_env():
    cryptos = os.getenv("CRYPTOS", "BTC").split(",")
    return [c.strip().upper() + "USDT" if not c.strip().upper().endswith("USDT") else c.strip().upper() for c in cryptos]

def get_interval_from_env():
    return os.getenv("INTERVAL", "4h")

def fetch_ohlcv(symbol, interval, limit=100):
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

def get_indicators(symbol, interval, limit=100):
    try:
        df = fetch_ohlcv(symbol, interval, limit)
        if df is None:
            logging.warning(f"Failed to fetch {symbol} OHLCV data. Indicators cannot be calculated.")
            return None
        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.stoch(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.vwap(append=True)
        df.ta.adx(append=True)
        df.ta.atr(length=14, append=True)
        df.ta.obv(append=True)

        latest = df.iloc[-1]

        def classify(value, indicator):
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
            "BB": "Bullish" if 'BBL_20_2.0' in latest and latest['close'] > latest['BBL_20_2.0'] else "Neutral",
            "SMA20": "Bullish" if latest['close'] > latest['SMA_20'] else "Bearish",
            "EMA50": "Bullish" if latest['close'] > latest['EMA_50'] else "Bearish",
            "VWAP": "Bullish" if latest['close'] > latest['VWAP_D'] else "Bearish",
            "ADX": classify(latest['ADX_14'], "ADX"),
            "ATR": f"{latest['ATRr_14']:.2f}",
            "OBV": f"{latest['OBV']:.2f}"
        }
        return indicators
    except Exception as e:
        logging.error(f"Error calculating indicators for {symbol}: {e}")
        return None
