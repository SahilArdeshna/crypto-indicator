import os
import pandas as pd
import requests
import logging
import time

def get_crypto_symbols_from_env():
    cryptos = os.getenv("CRYPTOS", "BTC").split(",")
    return [c.strip().upper() for c in cryptos]

def fetch_full_ohlcv(symbol='BTC', currency='USD', total_hours=8400):
    """
    Fetch historical hourly OHLCV data with pagination to get enough data for weekly indicators
    """
    logging.info(f"Fetching {total_hours} hourly candles for {symbol}/{currency}...")
    all_data = []
    limit = 2000  # Maximum limit allowed by CryptoCompare
    toTs = None

    while len(all_data) < total_hours:
        try:
            url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol}&tsym={currency}&limit={limit}"
            if toTs:
                url += f"&toTs={toTs}"
            
            response = requests.get(url)
            data = response.json()
            
            if data.get("Response") != "Success" or not data.get("Data", {}).get("Data"):
                logging.error(f"Error fetching data: {data.get('Message', 'Unknown error')}")
                break
                
            batch_data = data['Data']['Data']
            if not batch_data:
                break
                
            all_data = batch_data + all_data
            toTs = batch_data[0]['time'] - 1  # Go back in time
            
            logging.info(f"Fetched {len(batch_data)} candles, total: {len(all_data)}/{total_hours}")
            
            # Be nice to the API
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error during pagination: {e}")
            break

    # Create DataFrame from collected data
    df = pd.DataFrame(all_data[:total_hours])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('timestamp', inplace=True)
    
    # Rename columns to match our expected format
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volumefrom': 'volume'
    })
    
    # Select only the columns we need
    return df[['open', 'high', 'low', 'close', 'volume']]

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

def get_indicators(symbol):
    try:
        # Use the paginated fetch to get enough data for weekly indicators
        df = fetch_full_ohlcv(symbol=symbol, currency='USD', total_hours=8400)
        if df is None or df.empty:
            logging.warning(f"Failed to fetch {symbol} OHLCV data. Indicators cannot be calculated.")
            return None
        
        # Resample to weekly OHLCV (weekly candles starting from Monday)
        weekly_df = df.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logging.info(f"Created {len(weekly_df)} weekly candles from {len(df)} hourly candles for {symbol}")
        
        # Calculate indicators manually on weekly data
        weekly_df['RSI_14'] = calculate_rsi(weekly_df['close'], 14)
        weekly_df['SMA_20'] = calculate_sma(weekly_df['close'], 20)
        weekly_df['EMA_50'] = calculate_ema(weekly_df['close'], 50)
        
        # Simple VWAP calculation
        weekly_df['VWAP'] = (weekly_df['close'] * weekly_df['volume']).cumsum() / weekly_df['volume'].cumsum()
        
        # Simple Bollinger Bands
        sma_20 = weekly_df['SMA_20']
        std_20 = weekly_df['close'].rolling(window=20).std()
        weekly_df['BBU_20_2.0'] = sma_20 + (std_20 * 2)
        weekly_df['BBL_20_2.0'] = sma_20 - (std_20 * 2)
        
        # MACD calculation
        ema_12 = calculate_ema(weekly_df['close'], 12)
        ema_26 = calculate_ema(weekly_df['close'], 26)
        weekly_df['MACD_12_26_9'] = ema_12 - ema_26
        weekly_df['MACDs_12_26_9'] = calculate_ema(weekly_df['MACD_12_26_9'], 9)
        
        # Simple Stochastic
        low_14 = weekly_df['low'].rolling(window=14).min()
        high_14 = weekly_df['high'].rolling(window=14).max()
        weekly_df['STOCHk_14_3_3'] = 100 * ((weekly_df['close'] - low_14) / (high_14 - low_14))
        weekly_df['STOCHd_14_3_3'] = weekly_df['STOCHk_14_3_3'].rolling(window=3).mean()
        
        # Simple ATR
        tr1 = weekly_df['high'] - weekly_df['low']
        tr2 = abs(weekly_df['high'] - weekly_df['close'].shift(1))
        tr3 = abs(weekly_df['low'] - weekly_df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        weekly_df['ATRr_14'] = tr.rolling(window=14).mean()
        
        # Simple OBV
        obv = [0]
        for i in range(1, len(weekly_df)):
            if weekly_df['close'].iloc[i] > weekly_df['close'].iloc[i-1]:
                obv.append(obv[-1] + weekly_df['volume'].iloc[i])
            elif weekly_df['close'].iloc[i] < weekly_df['close'].iloc[i-1]:
                obv.append(obv[-1] - weekly_df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        weekly_df['OBV'] = obv
        
        # Simple ADX approximation (trend strength)
        weekly_df['ADX_14'] = weekly_df['close'].rolling(window=14).std() / weekly_df['close'].rolling(window=14).mean() * 100

        # Get the latest weekly data point
        latest = weekly_df.iloc[-1]

        def classify(value, indicator):
            if pd.isna(value):
                return "🟡"
            if indicator == "RSI":
                return "🟢" if value > 60 else "🔴" if value < 40 else "🟡"
            elif indicator == "MACD":
                return "🟢" if latest['MACD_12_26_9'] > latest['MACDs_12_26_9'] else "🔴"
            elif indicator == "Stoch":
                return "🟢" if latest['STOCHk_14_3_3'] > latest['STOCHd_14_3_3'] else "🔴"
            elif indicator == "ADX":
                return "Strong" if value > 25 else "Weak"
            else:
                return "🟡"

        indicators = {
            "RSI": classify(latest['RSI_14'], "RSI"),
            "MACD": classify(latest['MACD_12_26_9'], "MACD"),
            "Stochastic": classify(latest['STOCHk_14_3_3'], "Stoch"),
            "BB": "🟢" if 'BBL_20_2.0' in latest and not pd.isna(latest['BBL_20_2.0']) and latest['close'] > latest['BBL_20_2.0'] else "🟡",
            "SMA20": "🟢" if not pd.isna(latest['SMA_20']) and latest['close'] > latest['SMA_20'] else "🔴",
            "EMA50": "🟢" if not pd.isna(latest['EMA_50']) and latest['close'] > latest['EMA_50'] else "🔴",
            "VWAP": "🟢" if not pd.isna(latest['VWAP']) and latest['close'] > latest['VWAP'] else "🔴",
            "ADX": classify(latest['ADX_14'], "ADX"),
            "ATR": f"{latest['ATRr_14']:.2f}" if not pd.isna(latest['ATRr_14']) else "N/A",
            "OBV": f"{latest['OBV']:.2f}" if not pd.isna(latest['OBV']) else "N/A"
        }
        return indicators
    except Exception as e:
        logging.error(f"Error calculating indicators for {symbol}: {e}")
        return None
