import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from threading import Thread
import ta
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingBot')

class TradingBot:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.load_config()
        self.running = False
        self.connected = False
        self.data_cache = {}
        self.open_positions = {}
        self.account_info = None
        self.failed_symbols = {}  # Track symbols that fail to retrieve data
        self.symbol_mapping = {
            "US30": ["US30", "U30USD", "US30Index", "USA30", "USA30Cash", "DJ30", "DOW30", "USTEC"],
            "NASDAQ": ["NASDAQ", "NASDAQ100", "NAS100", "NASUSD", "USTECH", "USTECH100", "NDX"],
            "SPX500": ["SPX500", "SPXUSD", "S&P500", "SP500", "SPX", "US500"]
        }
    
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)
            logger.info("Configuration loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.error(f"Error loading PENI: {e}")
            self.config = self.get_default_config()
            return False
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_path, 'w') as file:
                json.dump(self.config, file, indent=4)
            logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get_default_config(self):
        """Return default configuration if config file not found"""
        return {
            "account": {
                "login": 0,
                "password": "",
                "server": "",
                "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
            },
            "trading": {
                "symbols": ["EURUSD", "GBPUSD", "XAUUSD", "US30", "NASDAQ", "SPX500"],
                "timeframes": ["M5", "M15", "H1"],
                "risk_percent": 2.0,
                "max_risk_percent": 20.0,
                "min_risk_reward": 2.0,
                "max_open_trades": 5,
                "use_trailing_stop": True,
                "trailing_stop_activation": 1.0,
                "trailing_stop_distance": 0.5
            },
            "strategy": {
                "price_action": {
                    "enabled": True,
                    "min_candle_size": 0.0010,
                    "rejection_level": 0.3,
                    "confirmation_candles": 2
                },
                "volume": {
                    "enabled": True,
                    "threshold": 1.5
                },
                "indicators": {
                    "rsi": {
                        "enabled": True,
                        "period": 14,
                        "overbought": 70,
                        "oversold": 30
                    },
                    "macd": {
                        "enabled": True,
                        "fast": 12,
                        "slow": 26,
                        "signal": 9
                    },
                    "support_resistance": {
                        "enabled": True,
                        "lookback": 100,
                        "threshold": 5
                    }
                }
            },
            "gui": {
                "theme": "dark",
                "update_interval": 1.0,
                "chart_periods": ["1H", "4H", "1D"]
            }
        }
    
    def connect(self):
        """Connect to MetaTrader 5 terminal"""
        if self.connected:
            return True
        
        try:
            # Initialize MT5 connection
            if not mt5.initialize(path=self.config["account"]["path"]):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login if account details are provided
            if self.config["account"]["login"] > 0:
                login_result = mt5.login(
                    self.config["account"]["login"],
                    self.config["account"]["password"],
                    self.config["account"]["server"]
                )
                
                if not login_result:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
            
            self.connected = True
            self.account_info = mt5.account_info()
            logger.info(f"Connected to MetaTrader 5. Account: {self.account_info.balance} {self.account_info.currency}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5 terminal"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MetaTrader 5")
    
    def get_market_data(self, symbol, timeframe, bars=500):
        """Get market data from MT5"""
        try:
            # Check if this symbol has consistently failed
            if symbol in self.failed_symbols:
                fail_info = self.failed_symbols[symbol]
                # If we've had too many consecutive failures, skip with less logging
                if fail_info['count'] > 5:
                    # Only log once every 10 attempts to reduce log spam
                    if fail_info['count'] % 10 == 0:
                        logger.warning(f"Skipping {symbol} {timeframe} - failed {fail_info['count']} consecutive times")
                    # Increment count but return None
                    self.failed_symbols[symbol]['count'] += 1
                    return None
                
                # Apply exponential backoff for retry
                last_attempt = fail_info['last_attempt']
                if (datetime.now() - last_attempt).total_seconds() < min(300, 5 * (2 ** min(fail_info['count'], 5))):
                    return None  # Skip this attempt based on backoff
            
            # Convert string timeframe to MT5 timeframe constant
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
            }
            
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Try with the original symbol name first
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            # If failed and symbol is one of our problem indices, try alternative names
            if (rates is None or len(rates) == 0) and symbol in self.symbol_mapping:
                # Try each alternative symbol name
                for alt_symbol in self.symbol_mapping[symbol]:
                    if alt_symbol == symbol:
                        continue  # Skip the original name we already tried
                    
                    logger.info(f"Trying alternative symbol name: {alt_symbol} for {symbol}")
                    alt_rates = mt5.copy_rates_from_pos(alt_symbol, tf, 0, bars)
                    
                    if alt_rates is not None and len(alt_rates) > 0:
                        # If successful, update the mapping preference
                        logger.info(f"Successfully retrieved data using alternate name: {alt_symbol}")
                        # Move the successful name to the front of the list for future attempts
                        self.symbol_mapping[symbol].remove(alt_symbol)
                        self.symbol_mapping[symbol].insert(0, alt_symbol)
                        rates = alt_rates
                        break
            
            # If we still don't have data, record the failure
            if rates is None or len(rates) == 0:
                err_msg = mt5.last_error()
                if symbol not in self.failed_symbols:
                    self.failed_symbols[symbol] = {
                        'count': 1,
                        'last_attempt': datetime.now(),
                        'last_error': err_msg
                    }
                else:
                    self.failed_symbols[symbol]['count'] += 1
                    self.failed_symbols[symbol]['last_attempt'] = datetime.now()
                    self.failed_symbols[symbol]['last_error'] = err_msg
                
                logger.error(f"Failed to get data for {symbol} {timeframe}: {err_msg}")
                return None
            
            # If successful, reset failure tracking for this symbol
            if symbol in self.failed_symbols:
                del self.failed_symbols[symbol]
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Cache the data
            key = f"{symbol}_{timeframe}"
            self.data_cache[key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol} {timeframe}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators on the provided DataFrame with enhanced indicators"""
        if df is None or len(df) < 50:
            return None
        
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # --- TREND INDICATORS ---
            
            # Moving Averages
            df['ma20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ma50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ma200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # EMA - Exponential Moving Averages
            df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # ADX - Average Directional Index for trend strength
            adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx_indicator.adx()
            df['di_plus'] = adx_indicator.adx_pos()
            df['di_minus'] = adx_indicator.adx_neg()
            
            # --- MOMENTUM INDICATORS ---
            
            # RSI
            if self.config["strategy"]["indicators"]["rsi"]["enabled"]:
                period = self.config["strategy"]["indicators"]["rsi"]["period"]
                df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
                
                # RSI Divergence detection (simple)
                df['rsi_slope'] = df['rsi'].rolling(window=5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
                df['price_slope'] = df['close'].rolling(window=5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
                df['rsi_divergence'] = np.where((df['rsi_slope'] > 0) & (df['price_slope'] < 0), 1,  # Bullish divergence
                                    np.where((df['rsi_slope'] < 0) & (df['price_slope'] > 0), -1, 0))  # Bearish divergence
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            if self.config["strategy"]["indicators"]["macd"]["enabled"]:
                fast = self.config["strategy"]["indicators"]["macd"]["fast"]
                slow = self.config["strategy"]["indicators"]["macd"]["slow"]
                signal = self.config["strategy"]["indicators"]["macd"]["signal"]
                
                macd = ta.trend.MACD(
                    df['close'], 
                    window_fast=fast, 
                    window_slow=slow, 
                    window_sign=signal
                )
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_histogram'] = macd.macd_diff()
                
                # MACD Divergence
                df['macd_slope'] = df['macd'].rolling(window=5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])
                df['macd_divergence'] = np.where((df['macd_slope'] > 0) & (df['price_slope'] < 0), 1,
                                        np.where((df['macd_slope'] < 0) & (df['price_slope'] > 0), -1, 0))
            
            # --- VOLATILITY INDICATORS ---
            
            # ATR - Average True Range
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bollinger_high'] = bollinger.bollinger_hband()
            df['bollinger_mid'] = bollinger.bollinger_mavg()
            df['bollinger_low'] = bollinger.bollinger_lband()
            df['bollinger_width'] = (df['bollinger_high'] - df['bollinger_low']) / df['bollinger_mid']
            
            # Enhanced Price Action features
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_pct'] = df['body_size'] / (df['high'] - df['low'])  # Body as percentage of range
            df['upper_wick'] = df.apply(lambda row: max(row['high'] - row['close'], row['high'] - row['open']), axis=1)
            df['lower_wick'] = df.apply(lambda row: max(row['open'] - row['low'], row['close'] - row['low']), axis=1)
            df['candle_range'] = df['high'] - df['low']
            df['candle_rel_size'] = df['candle_range'] / df['candle_range'].rolling(20).mean()
            
            # Candle classification
            df['is_bullish'] = df['close'] > df['open']
            df['is_doji'] = df['body_size'] < df['candle_range'] * 0.1
            df['is_hammer'] = (df['body_size'] < df['candle_range'] * 0.4) & (df['lower_wick'] > df['body_size'] * 2) & (df['lower_wick'] > df['upper_wick'] * 2)
            df['is_shooting_star'] = (df['body_size'] < df['candle_range'] * 0.4) & (df['upper_wick'] > df['body_size'] * 2) & (df['upper_wick'] > df['lower_wick'] * 2)
            df['is_engulfing'] = df['body_size'] > df['body_size'].shift(1) * 1.5
            
            # Inside/Outside bars
            df['is_inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
            df['is_outside_bar'] = (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
            
            # Advanced Support and resistance levels
            if self.config["strategy"]["indicators"]["support_resistance"]["enabled"]:
                lookback = self.config["strategy"]["indicators"]["support_resistance"]["lookback"]
                threshold = self.config["strategy"]["indicators"]["support_resistance"]["threshold"]
                
                # Swing high/low detection using fractals (enhanced)
                df['is_high'] = False
                df['is_low'] = False
                
                for i in range(2, len(df)-2):
                    # Swing high
                    if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                        df['high'].iloc[i] > df['high'].iloc[i-2] and
                        df['high'].iloc[i] > df['high'].iloc[i+1] and
                        df['high'].iloc[i] > df['high'].iloc[i+2]):
                        df.at[df.index[i], 'is_high'] = True
                    
                    # Swing low
                    if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                        df['low'].iloc[i] < df['low'].iloc[i-2] and
                        df['low'].iloc[i] < df['low'].iloc[i+1] and
                        df['low'].iloc[i] < df['low'].iloc[i+2]):
                        df.at[df.index[i], 'is_low'] = True
                
                # Calculate proximity to support/resistance
                highs = df[df['is_high']]['high'].tail(lookback).tolist()
                lows = df[df['is_low']]['low'].tail(lookback).tolist()
                
                df['dist_to_resistance'] = df.apply(lambda row: min([abs(row['close'] - lvl) for lvl in highs], default=float('inf')), axis=1)
                df['dist_to_support'] = df.apply(lambda row: min([abs(row['close'] - lvl) for lvl in lows], default=float('inf')), axis=1)
                
                # Normalize distances by ATR
                if 'atr' in df.columns and df['atr'].iloc[-1] > 0:
                    df['dist_to_resistance_atr'] = df['dist_to_resistance'] / df['atr']
                    df['dist_to_support_atr'] = df['dist_to_support'] / df['atr']
                
            # Volume analysis
            if 'tick_volume' in df.columns and self.config["strategy"]["volume"]["enabled"]:
                df['volume_ma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
                df['volume_trend'] = df['volume_ratio'].rolling(5).mean()
                
                # Volume climax detection
                df['vol_climax'] = (df['volume_ratio'] > 2.0) & (df['volume_ratio'] > df['volume_ratio'].shift(1) * 1.5)
            
            # Trend direction and strength
            df['trend_direction'] = np.where(df['ma50'] > df['ma200'], 1, -1)
            df['above_ma200'] = np.where(df['close'] > df['ma200'], 1, -1)
            df['trend_strength'] = df['adx']  # ADX measures trend strength
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_price_action(self, df):
        """Enhanced price action analysis using advanced patterns and multi-factor confirmation"""
        if df is None or len(df) < 20:
            return None, None
        
        try:
            # Use the last few candles for analysis
            recent_df = df.tail(5).copy()
            current = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            # Initialize signals with more detailed structure
            signals = {
                'buy': False, 
                'sell': False,
                'strength': 0,      # 0-10 signal strength
                'patterns': [],
                'confirmations': [],
                'warnings': []
            }
            
            # Extract settings from config
            min_candle_size = self.config["strategy"]["price_action"]["min_candle_size"]
            rejection_level = self.config["strategy"]["price_action"]["rejection_level"]
            
            # --- TREND ANALYSIS ---
            trend_direction = 1 if float(current['ma50']) > float(current['ma200']) else -1
            short_trend = 1 if float(current['ema20']) > float(current['ema50']) else -1
            
            # Trend strength from ADX
            trend_strong = current['adx'] > 25
            signals['confirmations'].append(f"ADX: {current['adx']:.1f} - {'Strong' if trend_strong else 'Weak'} trend")
            
            # --- CANDLESTICK PATTERNS ---
            
            # Bullish engulfing
            if (current['is_bullish'] and not prev['is_bullish'] and  # Current bullish, prev bearish
                current['is_engulfing'] and                          # Current engulfs previous
                current['open'] <= prev['close'] and                 # Current opens below prev close
                current['close'] > prev['open']):                    # Current closes above prev open
                signals['buy'] = True
                signals['strength'] += 2
                signals['patterns'].append('bullish_engulfing')
            
            # Bearish engulfing
            if (not current['is_bullish'] and prev['is_bullish'] and # Current bearish, prev bullish
                current['is_engulfing'] and                          # Current engulfs previous
                current['open'] >= prev['close'] and                 # Current opens above prev close
                current['close'] < prev['open']):                    # Current closes below prev open
                signals['sell'] = True
                signals['strength'] += 2
                signals['patterns'].append('bearish_engulfing')
            
            # Hammer (bullish reversal)
            if current['is_hammer'] and not current['is_bullish']:
                bullish_context = current['close'] < current['ma50'] and trend_direction > 0
                if bullish_context:
                    signals['buy'] = True
                    signals['strength'] += 2.5
                    signals['patterns'].append('hammer')
            
            # Shooting Star (bearish reversal)
            if current['is_shooting_star'] and current['is_bullish']:
                bearish_context = current['close'] > current['ma50'] and trend_direction < 0
                if bearish_context:
                    signals['sell'] = True
                    signals['strength'] += 2.5
                    signals['patterns'].append('shooting_star')
            
            # Doji - potential reversal signal depending on context
            if current['is_doji']:
                signals['patterns'].append('doji')
                
                # Doji at resistance in uptrend - bearish
                if current['close'] > current['ma20'] and current['dist_to_resistance_atr'] < 1:
                    signals['sell'] = True
                    signals['strength'] += 1.5
                    signals['patterns'].append('doji_at_resistance')
                
                # Doji at support in downtrend - bullish
                elif current['close'] < current['ma20'] and current['dist_to_support_atr'] < 1:
                    signals['buy'] = True
                    signals['strength'] += 1.5
                    signals['patterns'].append('doji_at_support')
            
            # Inside Bar - consolidation/continuation pattern
            if current['is_inside_bar']:
                signals['patterns'].append('inside_bar')
                
                # Inside bar in uptrend - potential bullish continuation
                if trend_direction > 0 and current['close'] > current['ma50']:
                    signals['buy'] = True
                    signals['strength'] += 1
                    signals['patterns'].append('inside_bar_bullish')
                
                # Inside bar in downtrend - potential bearish continuation
                elif trend_direction < 0 and current['close'] < current['ma50']:
                    signals['sell'] = True
                    signals['strength'] += 1
                    signals['patterns'].append('inside_bar_bearish')
            
            # Outside Bar - volatility/reversal pattern
            if current['is_outside_bar']:
                signals['patterns'].append('outside_bar')
                
                # Look for context to determine if reversal or continuation
                if current['is_bullish'] and prev2['is_bullish'] and not prev['is_bullish']:
                    # Potential bullish reversal of a single bearish candle
                    signals['buy'] = True
                    signals['strength'] += 2
                    signals['patterns'].append('outside_bar_bullish')
                elif not current['is_bullish'] and not prev2['is_bullish'] and prev['is_bullish']:
                    # Potential bearish reversal of a single bullish candle
                    signals['sell'] = True
                    signals['strength'] += 2
                    signals['patterns'].append('outside_bar_bearish')
            
            # --- SUPPORT/RESISTANCE ANALYSIS ---
            
            # Near Support/Resistance analysis
            near_resistance = 'dist_to_resistance_atr' in df.columns and current['dist_to_resistance_atr'] < 0.5
            near_support = 'dist_to_support_atr' in df.columns and current['dist_to_support_atr'] < 0.5
            
            if near_resistance:
                signals['warnings'].append('near_resistance')
                # Enhance sell signals at resistance
                if signals['sell']:
                    signals['strength'] += 1.5
                    signals['confirmations'].append('at_resistance')
                # Weaken buy signals at resistance
                if signals['buy']:
                    signals['strength'] -= 1
                    signals['warnings'].append('buying_at_resistance')
            
            if near_support:
                signals['warnings'].append('near_support')
                # Enhance buy signals at support
                if signals['buy']:
                    signals['strength'] += 1.5
                    signals['confirmations'].append('at_support')
                # Weaken sell signals at support
                if signals['sell']:
                    signals['strength'] -= 1
                    signals['warnings'].append('selling_at_support')
            
            # --- INDICATOR CONFIRMATIONS ---
            
            # RSI Analysis
            # Fix for potential Series comparison issues in the RSI analysis section
            # Replace the oversold/overbought conditions with:

            # RSI Analysis
            if 'rsi' in df.columns:
                current_rsi = current['rsi']
                rsi_oversold = self.config["strategy"]["indicators"]["rsi"]["oversold"]
                rsi_overbought = self.config["strategy"]["indicators"]["rsi"]["overbought"]
                
                # Oversold/Overbought conditions - use scalar values, not Series comparisons
                oversold = float(current_rsi) < rsi_oversold
                overbought = float(current_rsi) > rsi_overbought
                
                # RSI Divergence - make sure we're checking single values
                bullish_divergence = 'rsi_divergence' in df.columns and float(current['rsi_divergence']) > 0
                bearish_divergence = 'rsi_divergence' in df.columns and float(current['rsi_divergence']) < 0
                
                if oversold and signals['buy']:
                    signals['strength'] += 1.5
                    signals['confirmations'].append('rsi_oversold')
                
                if overbought and signals['sell']:
                    signals['strength'] += 1.5
                    signals['confirmations'].append('rsi_overbought')
                
                if bullish_divergence:
                    signals['buy'] = True
                    signals['strength'] += 2
                    signals['confirmations'].append('rsi_bullish_divergence')
                
                if bearish_divergence:
                    signals['sell'] = True
                    signals['strength'] += 2
                    signals['confirmations'].append('rsi_bearish_divergence')
            
            # MACD Analysis
# Fix for MACD section - ensure we're using scalar values:
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                # MACD Crossovers - use scalar values
                macd_bullish_cross = (float(current['macd']) > float(current['macd_signal']) and 
                                    float(df['macd'].iloc[-2]) <= float(df['macd_signal'].iloc[-2]))
                
                macd_bearish_cross = (float(current['macd']) < float(current['macd_signal']) and 
                                    float(df['macd'].iloc[-2]) >= float(df['macd_signal'].iloc[-2]))
                
                # MACD Divergence - ensure scalar values
                bullish_macd_divergence = 'macd_divergence' in df.columns and float(current['macd_divergence']) > 0
                bearish_macd_divergence = 'macd_divergence' in df.columns and float(current['macd_divergence']) < 0
                
                if macd_bullish_cross:
                    if signals['buy']:
                        signals['strength'] += 1.5
                        signals['confirmations'].append('macd_bullish_cross')
                    elif not signals['sell']:  # Don't override a sell signal
                        signals['buy'] = True
                        signals['strength'] += 1
                        signals['patterns'].append('macd_bullish_cross')
                
                if macd_bearish_cross:
                    if signals['sell']:
                        signals['strength'] += 1.5
                        signals['confirmations'].append('macd_bearish_cross')
                    elif not signals['buy']:  # Don't override a buy signal
                        signals['sell'] = True
                        signals['strength'] += 1
                        signals['patterns'].append('macd_bearish_cross')
                
                if bullish_macd_divergence and current['macd'] < 0:
                    signals['buy'] = True
                    signals['strength'] += 2
                    signals['confirmations'].append('macd_bullish_divergence')
                
                if bearish_macd_divergence and current['macd'] > 0:
                    signals['sell'] = True
                    signals['strength'] += 2
                    signals['confirmations'].append('macd_bearish_divergence')
            
            # Volume Confirmation
            if 'volume_ratio' in df.columns:
                high_volume = current['volume_ratio'] > self.config["strategy"]["volume"]["threshold"]
                volume_climax = 'vol_climax' in df.columns and current['vol_climax']
                
                if high_volume:
                    # High volume confirms the signal
                    if signals['buy'] or signals['sell']:
                        signals['strength'] += 1
                        signals['confirmations'].append('high_volume')
                
                if volume_climax:
                    # Volume climax often indicates exhaustion
                    if signals['buy'] and trend_direction < 0:
                        signals['strength'] += 1.5
                        signals['confirmations'].append('volume_climax_reversal')
                    elif signals['sell'] and trend_direction > 0:
                        signals['strength'] += 1.5
                        signals['confirmations'].append('volume_climax_reversal')
            
            # Bollinger Band Analysis
# Fix for the "The truth value of a Series is ambiguous" error in analyze_price_action function
# Replace the bb_squeeze line and subsequent conditional with:

# Bollinger Band Analysis
            if all(col in df.columns for col in ['bollinger_high', 'bollinger_low', 'bollinger_width']):
                # Bollinger Band Squeeze (low volatility, potential breakout setup)
                # Get the current bollinger width value only
                current_bb_width = current['bollinger_width']
                bb_width_mean = df['bollinger_width'].rolling(20).mean().iloc[-1]
                bb_squeeze = current_bb_width < bb_width_mean * 0.85
                
                # Price near bands - check single values, not Series
                near_upper_band = abs(current['close'] - current['bollinger_high']) / current['atr'] < 0.5
                near_lower_band = abs(current['close'] - current['bollinger_low']) / current['atr'] < 0.5
                
                # Band breakouts - compare single values
                upper_breakout = (current['close'] > current['bollinger_high'] and
                                df['close'].iloc[-2] <= df['bollinger_high'].iloc[-2])
                lower_breakout = (current['close'] < current['bollinger_low'] and
                                df['close'].iloc[-2] >= df['bollinger_low'].iloc[-2])
                
                if bb_squeeze:
                    signals['confirmations'].append('bollinger_squeeze')
                    # Strength depends on direction
                    if signals['buy'] and trend_direction > 0:
                        signals['strength'] += 1
                    elif signals['sell'] and trend_direction < 0:
                        signals['strength'] += 1
                
                if upper_breakout and trend_direction > 0:
                    if signals['buy']:
                        signals['strength'] += 1.5
                        signals['confirmations'].append('bollinger_upper_breakout')
                    else:
                        signals['buy'] = True
                        signals['strength'] += 1
                        signals['patterns'].append('bollinger_upper_breakout')
                
                if lower_breakout and trend_direction < 0:
                    if signals['sell']:
                        signals['strength'] += 1.5
                        signals['confirmations'].append('bollinger_lower_breakout')
                    else:
                        signals['sell'] = True
                        signals['strength'] += 1
                        signals['patterns'].append('bollinger_lower_breakout')
                
                # Mean reversion signals
                if near_upper_band and trend_direction < 0:
                    if signals['sell']:
                        signals['strength'] += 1
                        signals['confirmations'].append('overbought_at_band')
                
                if near_lower_band and trend_direction > 0:
                    if signals['buy']:
                        signals['strength'] += 1
                        signals['confirmations'].append('oversold_at_band')
            
            # --- TREND ALIGNMENT ADJUSTMENT ---
            
            # Strengthen signals in the direction of the stronger trend
            if signals['buy'] and trend_direction > 0 and short_trend > 0:
                signals['strength'] += 1
                signals['confirmations'].append('with_uptrend')
            
            if signals['sell'] and trend_direction < 0 and short_trend < 0:
                signals['strength'] += 1
                signals['confirmations'].append('with_downtrend')
            
            # Weaken counter-trend signals
            if signals['buy'] and trend_direction < 0 and short_trend < 0:
                signals['strength'] -= 1
                signals['warnings'].append('against_downtrend')
            
            if signals['sell'] and trend_direction > 0 and short_trend > 0:
                signals['strength'] -= 1
                signals['warnings'].append('against_uptrend')
            
            # Fix for the "buy_strength" KeyError in analyze_price_action function
            # Replace the conflicting signals section with:

            # --- FINAL SIGNAL ADJUSTMENT ---

            # Track separate strength values for buy and sell signals
            buy_strength = signals['strength'] if signals['buy'] else 0
            sell_strength = signals['strength'] if signals['sell'] else 0

            # Avoid conflicting signals by taking the stronger one
            if signals['buy'] and signals['sell']:
                if buy_strength > sell_strength:
                    signals['sell'] = False
                    signals['patterns'] = [p for p in signals['patterns'] if not p.startswith('bearish_') and not p.startswith('sell_')]
                else:
                    signals['buy'] = False
                    signals['patterns'] = [p for p in signals['patterns'] if not p.startswith('bullish_') and not p.startswith('buy_')]

            # Cap strength at 10
            signals['strength'] = min(10, max(0, signals['strength']))
            
            # --- CALCULATE ENTRY/EXIT LEVELS ---
            
            entry_price = None
            stop_loss = None
            take_profit = None
            
            if signals['buy']:
                entry_price = current['close']
                
                # Dynamic stop loss based on ATR and recent swing lows
                if 'atr' in df.columns:
                    atr_stop = current['low'] - current['atr'] * 1.5
                    
                    # Look for recent swing lows for stop placement
                    recent_lows = df[df['is_low'] == True]['low'].tail(3).min()
                    # If we found a valid swing low, use it if it's reasonable
                    if not pd.isna(recent_lows) and recent_lows < current['low']:
                        swing_stop = recent_lows - (0.1 * current['atr'])  # Small buffer below swing
                        
                        # Take the higher of the two stops (less risk)
                        stop_loss = max(atr_stop, swing_stop)
                    else:
                        stop_loss = atr_stop
                else:
                    # Fallback to the original method
                    stop_loss = min(df['low'].tail(3).min(), current['low'] - current['candle_range']*0.5)
                
                           # Take profit based on risk-reward ratio and potential resistance
                risk = entry_price - stop_loss
                base_tp = entry_price + (risk * self.config["trading"]["min_risk_reward"])
                
                # Check if there's resistance before the base TP
                if 'dist_to_resistance' in df.columns:
                    # Find nearby resistance levels
                    resistances = df[df['is_high']]['high'].tail(5).tolist()
                    # Filter resistance levels between entry and base TP
                    relevant_res = [r for r in resistances if entry_price < r < base_tp]
                    
                    if relevant_res:
                        # Use the closest resistance as TP, with a small buffer
                        closest_res = min(relevant_res)
                        take_profit = closest_res - (0.1 * current['atr'])  # Small buffer before resistance
                    else:
                        take_profit = base_tp
                else:
                    take_profit = base_tp
            
            elif signals['sell']:
                entry_price = current['close']
                
                # Dynamic stop loss based on ATR and recent swing highs
                if 'atr' in df.columns:
                    atr_stop = current['high'] + current['atr'] * 1.5
                    
                    # Look for recent swing highs for stop placement
                    recent_highs = df[df['is_high'] == True]['high'].tail(3).max()
                    # If we found a valid swing high, use it if it's reasonable
                    if not pd.isna(recent_highs) and recent_highs > current['high']:
                        swing_stop = recent_highs + (0.1 * current['atr'])  # Small buffer above swing
                        
                        # Take the lower of the two stops (less risk)
                        stop_loss = min(atr_stop, swing_stop)
                    else:
                        stop_loss = atr_stop
                else:
                    # Fallback to the original method
                    stop_loss = max(df['high'].tail(3).max(), current['high'] + current['candle_range']*0.5)
                
                # Take profit based on risk-reward ratio and potential support
                risk = stop_loss - entry_price
                base_tp = entry_price - (risk * self.config["trading"]["min_risk_reward"])
                
                # Check if there's support before the base TP
                if 'dist_to_support' in df.columns:
                    # Find nearby support levels
                    supports = df[df['is_low']]['low'].tail(5).tolist()
                    # Filter support levels between entry and base TP
                    relevant_sup = [s for s in supports if base_tp < s < entry_price]
                    
                    if relevant_sup:
                        # Use the closest support as TP, with a small buffer
                        closest_sup = max(relevant_sup)
                        take_profit = closest_sup + (0.1 * current['atr'])  # Small buffer above support
                    else:
                        take_profit = base_tp
                else:
                    take_profit = base_tp
            
            levels = {
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            return signals, levels
            
        except Exception as e:
            logger.error(f"Error analyzing price action: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return empty signals to avoid further errors
            return {'buy': False, 'sell': False, 'strength': 0, 'patterns': [], 'confirmations': [], 'warnings': []}, {'entry': None, 'stop_loss': None, 'take_profit': None}

    def _get_pip_size(self, symbol, digits):
        """Get proper pip size based on symbol characteristics"""
        # Standard forex pairs have 4 or 5 decimal places
        if symbol.endswith('JPY'):
            return 0.01 if digits == 2 else 0.001
        elif any(x in symbol for x in ["USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF"]):
            return 0.0001 if digits == 4 else 0.00001
        # Metals, indices, etc.
        elif symbol in ["XAUUSD", "XAGUSD"]:  # Gold, Silver
            return 0.01
        elif symbol in ["US30", "NASDAQ", "SPX500"]:  # Indices
            return 0.1 if digits == 1 else 1.0
        else:
            # Default based on digits
            if digits == 2:
                return 0.01
            elif digits == 3:
                return 0.001
            elif digits == 4:
                return 0.0001
            elif digits == 5:
                return 0.00001
            else:
                return 0.01

    def _get_symbol_exposure(self, symbol):
        """Calculate total exposure for a given symbol"""
        if not self.connected:
            return 0.0
        
        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions is None or len(positions) == 0:
                return 0.0
                
            total_lots = sum(pos.volume for pos in positions)
            return total_lots
        except Exception as e:
            logger.error(f"Error calculating exposure for {symbol}: {e}")
            return 0.0
        
    def calculate_position_size(self, symbol, entry, stop_loss):
        """Enhanced position size calculation with improved risk management"""
        if not self.connected or entry is None or stop_loss is None:
            return 0.0
        
        try:
            # Get account info and symbol info
            account_info = mt5.account_info()
            if account_info is None:
                return 0.0
                
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return 0.0
            
            # Risk calculation with account equity instead of balance for better risk management
            equity = account_info.equity
            balance = account_info.balance
            
            # Use the lower of equity or balance to be conservative with risk
            risk_capital = min(equity, balance)
            
            # Get risk percentage from config
            risk_percent = min(self.config["trading"]["risk_percent"], self.config["trading"]["max_risk_percent"])
            
            # Dynamic risk adjustment based on current drawdown
            if equity < balance:
                drawdown = (balance - equity) / balance
                # Reduce risk if in drawdown
                if drawdown > 0.05:  # More than 5% drawdown
                    risk_percent = max(0.5, risk_percent * (1 - drawdown))  # Reduce risk based on drawdown severity
                    logger.info(f"Reducing risk due to {drawdown:.2%} drawdown: now risking {risk_percent:.2f}%")
            
            # Calculate risk amount
            risk_amount = risk_capital * (risk_percent / 100.0)
            
            # Calculate pip value based on symbol characteristics
            pip_size = self._get_pip_size(symbol, symbol_info.digits)
            
            # Calculate stop loss distance in pips
            stop_distance_price = abs(entry - stop_loss)
            stop_distance_pips = stop_distance_price / pip_size
            
            # Validate stop distance is reasonable
            if stop_distance_pips < 5:
                logger.warning(f"{symbol}: Stop distance too small: {stop_distance_pips:.1f} pips")
                stop_distance_pips = max(stop_distance_pips, 5)  # Minimum 5 pip stop
            
            if stop_distance_pips > 200:
                logger.warning(f"{symbol}: Stop distance too large: {stop_distance_pips:.1f} pips")
                stop_distance_pips = min(stop_distance_pips, 200)  # Maximum 200 pip stop for most instruments
            
            # Calculate position size
            if stop_distance_pips > 0:
                position_size = risk_amount / (stop_distance_pips * pip_size * 10000)
                
                # Convert to lots - standard lot is 100,000 units
                lots = position_size / 100000
                
                # Adjust for contract size and tick value
                contract_size = symbol_info.trade_contract_size
                tick_value = symbol_info.trade_tick_value
                
                if contract_size > 0:
                    lots = lots / (contract_size / 100000)
                
                # Check maximum exposure per symbol
                open_exposure = self._get_symbol_exposure(symbol)
                if open_exposure + lots > 10.0:  # Maximum 10 lots per symbol
                    lots = max(0, 10.0 - open_exposure)
                    logger.warning(f"Limiting position size on {symbol} due to existing exposure")
                
                # Round to 2 decimals and ensure minimum lot size
                min_lot = symbol_info.volume_min
                lot_step = symbol_info.volume_step
                
                # Round to nearest lot step
                lots = round(max(min_lot, min(lots, 100.0)) / lot_step) * lot_step
                
                return lots
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def execute_trade(self, symbol, trade_type, lot_size, entry_price=0, stop_loss=0, take_profit=0):
        """Execute a trade on MT5"""
        if not self.connected:
            return None
        
        try:
            # Prepare trade request
            if trade_type.lower() not in ["buy", "sell"]:
                logger.error(f"Invalid trade type: {trade_type}")
                return None
                
            # Set up trade type
            trade_type_map = {
                "buy": mt5.ORDER_TYPE_BUY,
                "sell": mt5.ORDER_TYPE_SELL
            }
            
            order_type = trade_type_map[trade_type.lower()]
            
            # Get symbol information for proper price formatting
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            # Check if symbol is available for trading
            if not symbol_info.visible or not symbol_info.trade_allowed:
                logger.error(f"Symbol {symbol} is not available for trading")
                return None
            
            price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(lot_size),
                "type": order_type,
                "price": price,
                "sl": stop_loss,
                "tp": take_profit,
                "deviation": 10,  # Maximum price deviation in points
                "magic": 12345,   # Magic number for identification
                "comment": "TradingBot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK
            }
            
            # Execute the trade
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade execution failed: {result.retcode}, {result.comment}")
                return None
            else:
                logger.info(f"Trade executed successfully: {trade_type} {lot_size} {symbol} at {price}")
                return result
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def manage_open_positions(self):
        """Manage open positions (trailing stops, etc.)"""
        if not self.connected:
            return
            
        try:
            # Get open positions
            positions = mt5.positions_get()
            if positions is None:
                return
            
            for position in positions:
                # Check if trailing stop is enabled
                if not self.config["trading"]["use_trailing_stop"]:
                    continue
                    
                symbol = position.symbol
                position_id = position.ticket
                position_type = "buy" if position.type == mt5.POSITION_TYPE_BUY else "sell"
                
                # Get current price
                price_info = mt5.symbol_info_tick(symbol)
                if price_info is None:
                    continue
                    
                current_price = price_info.bid if position_type == "buy" else price_info.ask
                
                # Calculate profit in pips
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    continue
                    
                pip_size = 0.0001 if symbol_info.digits == 4 else 0.00001
                if "JPY" in symbol:
                    pip_size = 0.01 if symbol_info.digits == 2 else 0.001
                
                # For indices and metals, get the appropriate pip size
                if symbol in ["XAUUSD", "XAGUSD", "US30", "NASDAQ", "SPX500"]:
                    pip_size = 0.1 if symbol_info.digits == 1 else 0.01
                
                profit_price = abs(current_price - position.price_open)
                profit_pips = profit_price / pip_size
                
                # Check if profit exceeds activation level
                activation_pips = self.config["trading"]["trailing_stop_activation"]
                if profit_pips > activation_pips:
                    # Calculate new stop loss level
                    trailing_distance = self.config["trading"]["trailing_stop_distance"] * pip_size
                    current_sl = position.sl
                    
                    if position_type == "buy":
                        new_sl = current_price - trailing_distance
                        # Only move stop loss if it's higher than current
                        if current_sl == 0 or new_sl > current_sl:
                            self.modify_position(position_id, new_sl, position.tp)
                    else:  # sell
                        new_sl = current_price + trailing_distance
                        # Only move stop loss if it's lower than current
                        if current_sl == 0 or new_sl < current_sl:
                            self.modify_position(position_id, new_sl, position.tp)
        
        except Exception as e:
            logger.error(f"Error managing positions: {e}")
    
    def modify_position(self, ticket, sl, tp):
        """Modify an existing position (change SL/TP)"""
        if not self.connected:
            return False
            
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": sl,
                "tp": tp
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to modify position {ticket}: {result.retcode}, {result.comment}")
                return False
            else:
                logger.info(f"Modified position {ticket}: SL={sl}, TP={tp}")
                return True
                
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False
    
    def analyze_symbol(self, symbol):
        """Analyze a single symbol across timeframes and return trading signals"""
        if not self.connected:
            return None
            
        try:
            signals = []
            
            for timeframe in self.config["trading"]["timeframes"]:
                # Get and process market data
                df = self.get_market_data(symbol, timeframe)
                if df is None:
                    continue
                    
                # Calculate indicators
                df = self.calculate_indicators(df)
                if df is None:
                    continue
                
                # Analyze price action
                pa_signals, levels = self.analyze_price_action(df)
                if pa_signals is None:
                    continue
                
                # If we have a signal with sufficient strength
                if (pa_signals["buy"] or pa_signals["sell"]) and pa_signals["strength"] >= 5:
                    signal = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "action": "buy" if pa_signals["buy"] else "sell",
                        "strength": pa_signals["strength"],
                        "patterns": pa_signals["patterns"],
                        "entry": levels["entry"],
                        "stop_loss": levels["stop_loss"],
                        "take_profit": levels["take_profit"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Calculate position size
                    lot_size = self.calculate_position_size(
                        symbol, levels["entry"], levels["stop_loss"]
                    )
                    signal["lot_size"] = lot_size
                    
                    signals.append(signal)
            
            return signals
                
        except Exception as e:
            logger.error(f"Error analyzing symbol {symbol}: {e}")
            return None
        
    def _analyze_market_condition(self):
        """Analyze overall market condition using benchmark indices"""
        market_condition = {
            'bias': 'neutral',  # 'bullish', 'bearish', or 'neutral'
            'volatility': 'normal',  # 'high', 'low', or 'normal'
            'correlated_pairs': []  # List of correlated pairs
        }
        
        try:
            # Use major indices or USD index as benchmarks
            benchmark_symbols = ["US30", "SPX500", "EURUSD", "USDJPY"]
            benchmark_data = {}
            
            for symbol in benchmark_symbols:
                # Try to get the data, use alternative names if needed
                for alt_symbol in [symbol] + self.symbol_mapping.get(symbol, []):
                    df = self.get_market_data(alt_symbol, "H1", 50)
                    if df is not None and len(df) > 0:
                        benchmark_data[symbol] = df
                        break
            
            # Need at least some benchmark data
            if len(benchmark_data) < 2:
                return market_condition
            
            # Calculate overall market bias
            bullish_count = 0
            bearish_count = 0
            
            for symbol, df in benchmark_data.items():
                # Calculate short and medium term trends
                df = df.copy()
                df['ma20'] = df['close'].rolling(20).mean()
                df['ma50'] = df['close'].rolling(50).mean()
                
                # Get the last row as a single value, not a Series
                last_close = float(df['close'].iloc[-1])
                last_ma20 = float(df['ma20'].iloc[-1])
                last_ma50 = float(df['ma50'].iloc[-1])
                
                # Price above both MAs suggests bullish bias
                if last_close > last_ma20 and last_close > last_ma50:
                    bullish_count += 1
                # Price below both MAs suggests bearish bias
                elif last_close < last_ma20 and last_close < last_ma50:
                    bearish_count += 1
                # Short term MA above long term MA suggests bullish
                elif last_ma20 > last_ma50:
                    bullish_count += 0.5
                # Short term MA below long term MA suggests bearish
                elif last_ma20 < last_ma50:
                    bearish_count += 0.5
            
            # Determine overall bias
            if bullish_count > bearish_count + 1:
                market_condition['bias'] = 'bullish'
            elif bearish_count > bullish_count + 1:
                market_condition['bias'] = 'bearish'
            
            # Analyze volatility
            volatility_scores = []
            for symbol, df in benchmark_data.items():
                if 'atr' not in df.columns:
                    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
                
                # Compare current ATR to its average - use scalar values
                current_atr = float(df['atr'].iloc[-1])
                avg_atr = float(df['atr'].rolling(20).mean().iloc[-1])
                
                if not pd.isna(current_atr) and not pd.isna(avg_atr) and avg_atr > 0:
                    volatility_ratio = current_atr / avg_atr
                    volatility_scores.append(volatility_ratio)
            
            # Set volatility based on average score
            if volatility_scores:
                avg_volatility = sum(volatility_scores) / len(volatility_scores)
                if avg_volatility > 1.5:
                    market_condition['volatility'] = 'high'
                elif avg_volatility < 0.7:
                    market_condition['volatility'] = 'low'
            
            return market_condition
            
        except Exception as e:
            logger.error(f"Error analyzing market condition: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return market_condition
    
    def run_trading_cycle(self):
        """Enhanced trading cycle with improved signal filtering"""
        if not self.connected:
            logger.info("Not connected to MT5. Attempting to connect...")
            if not self.connect():
                return []
        
        all_signals = []
        
        # Filter out symbols that have consistently failed (more than 20 times)
        current_symbols = [s for s in self.config["trading"]["symbols"] 
                        if s not in self.failed_symbols or self.failed_symbols[s]['count'] <= 20]
        
        if len(current_symbols) < len(self.config["trading"]["symbols"]):
            disabled_symbols = set(self.config["trading"]["symbols"]) - set(current_symbols)
            logger.warning(f"Skipping analysis for consistently failing symbols: {', '.join(disabled_symbols)}")
        
        # Calculate market condition metrics first (overall market state)
        market_condition = self._analyze_market_condition()
        
        # Track how many signals per direction we're finding
        buy_signals = 0
        sell_signals = 0
        
        for symbol in current_symbols:
            signals = self.analyze_symbol(symbol)
            if signals:
                # Count signal directions
                for signal in signals:
                    if signal["action"] == "buy":
                        buy_signals += 1
                    else:
                        sell_signals += 1
                
                # Add market condition info to each signal
                for signal in signals:
                    signal["market_condition"] = market_condition
                    all_signals.append(signal)
        
        # Adjust signal strengths based on correlation with market condition
        if market_condition.get('bias') == 'bullish':
            # Strengthen buy signals, weaken sell signals
            for signal in all_signals:
                if signal["action"] == "buy":
                    signal["strength"] = min(10, signal["strength"] + 0.5)
                    signal["confirmations"].append("aligns_with_bullish_market")
                else:
                    signal["strength"] = max(0, signal["strength"] - 0.5)
                    signal["warnings"].append("against_bullish_market")
        
        elif market_condition.get('bias') == 'bearish':
            # Strengthen sell signals, weaken buy signals
            for signal in all_signals:
                if signal["action"] == "sell":
                    signal["strength"] = min(10, signal["strength"] + 0.5)
                    signal["confirmations"].append("aligns_with_bearish_market")
                else:
                    signal["strength"] = max(0, signal["strength"] - 0.5)
                    signal["warnings"].append("against_bearish_market")
        
        # Sort signals by strength
        all_signals.sort(key=lambda x: x["strength"], reverse=True)
        
        # Filter signals with too many warnings or insufficient strength
        filtered_signals = []
        for signal in all_signals:
            # Only consider strong signals (strength >= 6)
            if signal["strength"] >= 6:
                # Skip signals with too many warnings
                if len(signal.get("warnings", [])) <= 2:
                    filtered_signals.append(signal)
        
        # Log signal summary
        if filtered_signals:
            logger.info(f"Found {len(filtered_signals)} strong trading signals after filtering")
            for i, signal in enumerate(filtered_signals[:3]):  # Log top 3 signals
                logger.info(f"Signal {i+1}: {signal['symbol']} {signal['timeframe']} {signal['action'].upper()} (Strength: {signal['strength']:.1f})")
        
        return filtered_signals
    
    def auto_trade(self, signals):
        """Automatically execute trades based on signals"""
        if not self.connected or not signals:
            return []
        
        # Check how many positions are already open
        positions = mt5.positions_get()
        open_positions_count = len(positions) if positions is not None else 0
        max_positions = self.config["trading"]["max_open_trades"]
        
        executed_trades = []
        
        # If we have room for more positions
        if open_positions_count < max_positions:
            available_slots = max_positions - open_positions_count
            
            # Take the strongest signals first, limited by available slots
            for signal in signals[:available_slots]:
                # Execute the trade
                result = self.execute_trade(
                    signal["symbol"],
                    signal["action"],
                    signal["lot_size"],
                    signal["entry"],
                    signal["stop_loss"],
                    signal["take_profit"]
                )
                
                if result is not None:
                    executed_trades.append({
                        "ticket": result.order,
                        "symbol": signal["symbol"],
                        "action": signal["action"],
                        "lot_size": signal["lot_size"],
                        "entry": signal["entry"],
                        "stop_loss": signal["stop_loss"],
                        "take_profit": signal["take_profit"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        return executed_trades
    
    def start(self):
        """Start the trading bot in a separate thread"""
        if self.running:
            logger.warning("Trading bot is already running")
            return False
        
        self.running = True
        self.thread = Thread(target=self._trading_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Trading bot started")
        return True
    
    def stop(self):
        """Stop the trading bot"""
        if not self.running:
            logger.warning("Trading bot is not running")
            return False
        
        self.running = False
        logger.info("Trading bot stopped")
        return True
    
    # Update the _trading_loop method to be more responsive
    def _trading_loop(self):
        """Main trading loop running in a separate thread"""
        logger.info("Trading loop started")
        
        last_analysis_time = datetime.now() - timedelta(minutes=5)  # Force first analysis immediately
        
        while self.running:
            try:
                # Make sure we're connected
                if not self.connected:
                    if not self.connect():
                        logger.error("Failed to connect to MT5, retrying in 30 seconds")
                        time.sleep(5)  # Check more frequently but with shorter sleeps
                        continue
                
                # Manage existing positions - do this more frequently
                self.manage_open_positions()
                
                # Only run full analysis periodically to avoid hammering the API
                time_since_last = (datetime.now() - last_analysis_time).total_seconds()
                if time_since_last >= 60:  # Run analysis once per minute at most
                    # Run analysis and get trading signals
                    signals = self.run_trading_cycle()
                    last_analysis_time = datetime.now()
                    
                    # Execute trades if auto-trading is enabled
                    if self.config.get("auto_trading", True) and signals:
                        logger.info(f"Found {len(signals)} potential trading opportunities")
                        executed_trades = self.auto_trade(signals)
                        if executed_trades:
                            logger.info(f"Executed {len(executed_trades)} trades")
                
                # Quick sleep to keep the thread responsive
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Shorter sleep after error
        
        logger.info("Trading loop stopped")
    
    def get_account_summary(self):
        """Get account summary information"""
        if not self.connected:
            return None
        
        try:
            account = mt5.account_info()
            if account is None:
                return None
            
            return {
                "balance": account.balance,
                "equity": account.equity,
                "profit": account.profit,
                "margin": account.margin,
                "free_margin": account.margin_free,
                "margin_level": account.margin_level,
                "currency": account.currency
            }
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None
    
    def get_open_positions(self):
        """Get all open positions"""
        if not self.connected:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                    "volume": pos.volume,
                    "open_price": pos.price_open,
                    "current_price": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "open_time": datetime.fromtimestamp(pos.time).strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return result
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
    
    def close_position(self, ticket):
        """Close a specific position by ticket number"""
        if not self.connected:
            return False
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                logger.error(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            
            # Create opposite request to close the position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 12345,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position {ticket}: {result.retcode}, {result.comment}")
                return False
            else:
                logger.info(f"Position {ticket} closed successfully")
                return True
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False


# Main execution
if __name__ == "__main__":
    # Create the bot instance
    bot = TradingBot()
    
    # Try to connect to MT5
    if bot.connect():
        logger.info("Connected to MetaTrader 5")
        
        # Start the bot
        bot.start()
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping bot...")
            bot.stop()
            bot.disconnect()
    else:
        logger.error("Failed to connect to MetaTrader 5")