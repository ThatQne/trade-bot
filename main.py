import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from threading import Thread
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ta
import os
import matplotlib.dates as mdates

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
        """Get market data from MT5 with simplified symbol handling"""
        try:
            # Convert string timeframe to MT5 timeframe constant
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1
            }
            
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Try to get the market data
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            
            # Track failures
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
            
            # Reset failure tracking on success
            if symbol in self.failed_symbols:
                del self.failed_symbols[symbol]
            
            # Process data
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
        
    def get_available_symbols(self):
        """Get all available symbols using visibility as the tradability indicator"""
        if not self.connected:
            if not self.connect():
                return []
        
        try:
            # Get all available symbols
            all_symbols = mt5.symbols_get()
            if all_symbols is None:
                logger.error("Failed to get symbols from broker")
                return []
            
            # Process all symbols
            all_symbol_data = []
            visible_symbols_data = []
            
            for symbol in all_symbols:
                symbol_name = symbol.name
                symbol_info = mt5.symbol_info(symbol_name)
                
                if symbol_info is None:
                    continue
                
                # Use visibility as the tradability indicator (as requested)
                is_visible = bool(symbol_info.visible)
                # Considering visible symbols as tradable
                is_tradable = is_visible
                
                # Get the latest tick to calculate real-time spread
                tick = mt5.symbol_info_tick(symbol_name)
                pip_size = self._get_pip_size(symbol_name, symbol_info.digits)
                
                spread_in_pips = 0
                spread_percentage = 0
                
                if tick and tick.ask > 0 and tick.bid > 0:
                    # Calculate real-time spread
                    real_spread = tick.ask - tick.bid
                    # Convert to pips
                    spread_in_pips = real_spread / pip_size * 10
                    # Calculate percentage spread for better comparison across instruments
                    spread_percentage = (real_spread / tick.bid) * 100 if tick.bid > 0 else 0
                else:
                    # Fallback to symbol info spread
                    spread_in_pips = symbol_info.spread / 10
                
                # Create symbol data record
                symbol_data = {
                    'name': symbol_name,
                    'spread': spread_in_pips,
                    'spread_percentage': spread_percentage,
                    'digits': symbol_info.digits,
                    'pip_value': pip_size,
                    'volume_min': getattr(symbol_info, 'volume_min', 0.01),
                    'volume_step': getattr(symbol_info, 'volume_step', 0.01),
                    'type': self._classify_symbol_type(symbol_name),
                    'is_tradable': is_tradable,
                    'is_visible': is_visible
                }
                
                # Store in appropriate lists
                all_symbol_data.append(symbol_data)
                if is_visible:  # Using visibility as tradability
                    visible_symbols_data.append(symbol_data)
            
            # Sort all symbols by spread percentage for better comparison across instruments
            all_symbol_data.sort(key=lambda x: x['spread_percentage'])
            
            # Count visible (tradable) symbols
            visible_count = len(visible_symbols_data)
            
            # Log summary
            logger.info(f"Found {len(all_symbol_data)} available symbols from broker ({visible_count} visible/tradable)")
            
            # Store lists for different purposes
            all_names = [s['name'] for s in all_symbol_data]
            visible_names = [s['name'] for s in visible_symbols_data]
            
            # Store visible symbols list for auto-trading (treating visible as tradable)
            self.tradable_symbols = visible_names
            
            # Log the top symbols with comprehensive info
            for i, symbol in enumerate(all_symbol_data[:30]):
                visibility_text = "visible/tradable" if symbol['is_visible'] else "hidden/non-tradable"
                logger.info(f"Top {i+1}: {symbol['name']} - "
                        f"Spread: {symbol['spread']:.2f} pips "
                        f"({symbol['spread_percentage']:.5f}%) - "
                        f"Type: {symbol['type']} - {visibility_text}")
            
            # Prepare priority symbols for analysis - crucial to include ALL tradable/visible symbols first
            symbols_to_analyze = self.config.get("analysis", {}).get("symbols_to_analyze", 30)
            
            # Start with ALL visible/tradable symbols
            priority_symbols = list(visible_names)
            
            # Then add additional non-visible symbols until we reach the limit
            remaining_slots = symbols_to_analyze - len(priority_symbols)
            if remaining_slots > 0:
                non_visible_symbols = [s['name'] for s in all_symbol_data if s['name'] not in priority_symbols]
                priority_symbols.extend(non_visible_symbols[:remaining_slots])
            
            # If we need to trim down, make sure to keep ALL visible/tradable symbols first
            if len(priority_symbols) > symbols_to_analyze:
                # Only trim non-visible symbols if needed
                visible_set = set(visible_names)
                non_visible_priority = [s for s in priority_symbols if s not in visible_set]
                
                # Keep all visible plus enough non-visible symbols to reach the limit
                remaining_slots = symbols_to_analyze - len(visible_names)
                if remaining_slots > 0:
                    priority_symbols = visible_names + non_visible_priority[:remaining_slots]
                else:
                    priority_symbols = visible_names[:symbols_to_analyze]  # In case we have too many visible symbols
            
            # Store the priority symbols
            self.analysis_priority_symbols = priority_symbols
            
            logger.info(f"Analysis priority list includes {len(visible_names)} visible/tradable symbols + "
                    f"{len(priority_symbols) - len(visible_names)} additional symbols")
            
            return all_names  # Return all symbols for comprehensive data
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    def _classify_trade_type(self, timeframe, signal_strength):
        """Enhanced trade type classification with more detailed information"""
        # Base classification on timeframe
        if timeframe == "M1":
            trade_type = "Scalp (ultra-short)"
        elif timeframe == "M5":
            trade_type = "Scalp"
        elif timeframe == "M15":
            trade_type = "Short-term"
        elif timeframe == "M30":
            trade_type = "Short-term swing"
        elif timeframe == "H1":
            trade_type = "Swing"
        elif timeframe == "H4":
            trade_type = "Medium-term"
        elif timeframe == "D1":
            trade_type = "Position"
        else:
            trade_type = "Long-term"
        
        # Adjust based on signal strength
        if signal_strength >= 9:
            confidence = "Very Strong"
        elif signal_strength >= 7:
            confidence = "Strong"
        elif signal_strength >= 5:
            confidence = "Moderate"
        else:
            confidence = "Tentative"
            
        return f"{confidence} {trade_type}"

    def run_trading_cycle(self):
        """Enhanced trading cycle with improved signal collection across all timeframes"""
        if not self.connected:
            logger.info("Not connected to MT5. Attempting to connect...")
            if not self.connect():
                return []
        
        all_signals = []
        
        # Get analysis parameters from config
        symbols_to_analyze = self.config.get("analysis", {}).get("symbols_to_analyze", 30)
        min_signal_strength = self.config.get("analysis", {}).get("min_signal_strength", 4.0)
        max_warnings = self.config.get("analysis", {}).get("max_warnings", 3)
        
        # Get available symbols if we haven't already
        if not hasattr(self, 'analysis_priority_symbols') or not self.analysis_priority_symbols:
            self.get_available_symbols()
            
        # Use the priority symbols determined during symbol discovery
        top_symbols = self.analysis_priority_symbols[:symbols_to_analyze] if hasattr(self, 'analysis_priority_symbols') else []
        
        # Filter out symbols that have consistently failed
        top_symbols = [s for s in top_symbols 
                    if s not in self.failed_symbols or self.failed_symbols[s]['count'] <= 20]
        
        # Calculate market condition metrics first
        market_condition = self._analyze_market_condition()
        
        # Create analysis tasks using asyncio
        try:
            # Randomize to avoid always analyzing the same symbols first
            import random
            random.shuffle(top_symbols)
            
            # Define all timeframes to analyze (both short and long term)
            all_timeframes = ["M5", "M15", "M30", "H1", "H4", "D1"]
            
            # Create tasks for all symbol/timeframe combinations
            async def analyze_all():
                # Create a thread pool with limited concurrency to avoid overloading MT5
                with ThreadPoolExecutor(max_workers=5) as executor:
                    loop = asyncio.get_event_loop()
                    
                    # First batch - analysis tasks for top symbols across all timeframes
                    priority_tasks = []
                    for symbol in top_symbols:
                        # Analyze all timeframes for each symbol for comprehensive analysis
                        for timeframe in all_timeframes:
                            priority_tasks.append(loop.run_in_executor(
                                executor,
                                self._analyze_symbol_timeframe,
                                symbol,
                                timeframe
                            ))
                    
                    # Run priority tasks first (top symbols)
                    logger.info(f"Analyzing {len(priority_tasks)} symbol/timeframe combinations")
                    start_time = time.time()
                    
                    # Run tasks in smaller batches to maintain responsiveness
                    batch_size = 10
                    for i in range(0, len(priority_tasks), batch_size):
                        batch = priority_tasks[i:i+batch_size]
                        batch_results = await asyncio.gather(*batch)
                        
                        # Collect signals from this batch
                        for signals in batch_results:
                            if signals:
                                all_signals.extend(signals)
                                    
                        # Add short pause between batches to keep interface responsive
                        await asyncio.sleep(0.1)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Analysis completed in {elapsed:.2f}s, found {len(all_signals)} signals")
                    
                    return all_signals
                
            # Run the asyncio event loop with a timeout to prevent hanging
            if hasattr(asyncio, 'run'):  # Python 3.7+
                signals_list = asyncio.run(analyze_all())
            else:  # Python 3.6 and below
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                signals_list = loop.run_until_complete(analyze_all())
                loop.close()
            
            logger.info(f"Async analysis completed, found {len(signals_list)} initial signals")
            
        except Exception as e:
            logger.error(f"Error in asyncio analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to simple approach
            logger.warning("Falling back to sequential analysis")
            signals_list = []
            
            for symbol in top_symbols[:10]:  # Analyze fewer symbols in fallback mode
                for timeframe in self.config["trading"]["timeframes"]:
                    signals = self._analyze_symbol_timeframe(symbol, timeframe)
                    if signals:
                        signals_list.extend(signals)
        
        # Adjust signals based on market condition and add trade type classification
        for signal in signals_list:
            # Ensure all signals have standard fields
            if "confirmations" not in signal:
                signal["confirmations"] = []
            if "warnings" not in signal:
                signal["warnings"] = []
            
            # Add trade type classification
            if "trade_type" not in signal:
                signal["trade_type"] = self._classify_trade_type(signal["timeframe"], signal["strength"])
            
            # Add market condition adjustment
            if market_condition.get('bias') == 'bullish':
                if signal["action"] == "buy":
                    signal["strength"] = min(10, signal["strength"] + 0.5)
                    signal["confirmations"].append("aligns_with_bullish_market")
                else:
                    signal["strength"] = max(0, signal["strength"] - 0.5)
                    signal["warnings"].append("against_bullish_market")
            
            elif market_condition.get('bias') == 'bearish':
                if signal["action"] == "sell":
                    signal["strength"] = min(10, signal["strength"] + 0.5)
                    signal["confirmations"].append("aligns_with_bearish_market")
                else:
                    signal["strength"] = max(0, signal["strength"] - 0.5)
                    signal["warnings"].append("against_bearish_market")
        
        # Sort signals by strength and filter
        signals_list.sort(key=lambda x: x["strength"], reverse=True)
        
        # Filter to get signals with configurable strength and warnings
        filtered_signals = [
            s for s in signals_list 
            if s["strength"] >= min_signal_strength and len(s.get("warnings", [])) <= max_warnings
        ]
        
        # Log signal summary
        if filtered_signals:
            logger.info(f"Found {len(filtered_signals)} trading signals after filtering")
            for i, signal in enumerate(filtered_signals[:5]):  # Log top 5 signals
                trade_type = signal.get('trade_type', '')
                logger.info(f"Signal {i+1}: {signal['symbol']} {signal['timeframe']} {signal['action'].upper()} "
                        f"(Strength: {signal['strength']:.1f}) - {trade_type}")
        
        return filtered_signals

    def _classify_symbol_type(self, symbol_name):
        """Classify symbol into categories for better filtering"""
        symbol_upper = symbol_name.upper()
        
        # Check for forex pairs
        if (len(symbol_name) == 6 and 
            all(c in "EURUSDGBPCADAUDNZDJPYCHFSGDSEK" for c in symbol_upper)):
            return "forex"
            
        # Check for indices
        elif any(index in symbol_upper for index in ["SPX", "US30", "US500", "NDX", "NAS", "DJ", "DAX", "FTSE"]):
            return "index"
            
        # Check for metals
        elif any(metal in symbol_upper for metal in ["XAU", "GOLD", "XAG", "SILVER"]):
            return "metal"
            
        # Check for crypto
        elif any(crypto in symbol_upper for crypto in ["BTC", "ETH", "LTC", "XRP"]):
            return "crypto"
            
        # Check for commodities
        elif any(commodity in symbol_upper for commodity in ["OIL", "BRENT", "GAS", "NG"]):
            return "commodity"
            
        # Default
        else:
            return "other"
    
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

            # Avoid conflicting signals by taking the stronger one
            if signals['buy'] and signals['sell']:
                # Calculate strength for each direction based on patterns and confirmations
                buy_patterns = len([p for p in signals['patterns'] if 'bullish' in p or 'buy' in p])
                sell_patterns = len([p for p in signals['patterns'] if 'bearish' in p or 'sell' in p])
                
                buy_confirmations = len([c for c in signals.get('confirmations', []) if 'bullish' in c or 'buy' in c or 'support' in c])
                sell_confirmations = len([c for c in signals.get('confirmations', []) if 'bearish' in c or 'sell' in c or 'resistance' in c])
                
                # Calculate total strength for each direction
                buy_total = buy_patterns + buy_confirmations
                sell_total = sell_patterns + sell_confirmations
                
                # Keep only the stronger signal
                if buy_total > sell_total:
                    signals['sell'] = False
                else:
                    signals['buy'] = False

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
        """Position size calculation with improved stop distance handling"""
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
            
            # Use the lower of equity or balance to be conservative with risk
            risk_capital = min(account_info.equity, account_info.balance)
            
            # Get risk percentage from config
            risk_percent = self.config["trading"]["risk_percent"]
            max_risk_percent = self.config["trading"]["max_risk_percent"]
            risk_percent = min(risk_percent, max_risk_percent)
            
            # Calculate risk amount
            risk_amount = risk_capital * (risk_percent / 100.0)
            
            # Get pip size for this symbol
            pip_size = self._get_pip_size(symbol, symbol_info.digits)
            
            # Calculate stop loss distance in price units
            stop_distance_price = abs(entry - stop_loss)
            
            # Get ATR for dynamic stop sizing if available
            atr_value = None
            try:
                df = self.get_market_data(symbol, "H1", 50)
                if df is not None and len(df) > 20:
                    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
                    atr_value = atr.iloc[-1]
            except:
                pass
            
            # Determine reasonable stop distance based on symbol type
            symbol_type = self._classify_symbol_type(symbol)
            
            # Calculate minimum and maximum stop distances based on symbol type and ATR
            if symbol_type == "forex":
                if "JPY" in symbol:
                    min_stop_pips = 10
                    typical_stop_pips = 30 if atr_value is None else (atr_value / pip_size) * 1.5
                    max_stop_pips = 150
                else:
                    min_stop_pips = 5
                    typical_stop_pips = 20 if atr_value is None else (atr_value / pip_size) * 1.5
                    max_stop_pips = 100
            elif symbol_type == "metal":
                min_stop_pips = 20
                typical_stop_pips = 100 if atr_value is None else (atr_value / pip_size) * 1.5
                max_stop_pips = 300
            elif symbol_type == "index":
                min_stop_pips = 10
                typical_stop_pips = 50 if atr_value is None else (atr_value / pip_size) * 1.5
                max_stop_pips = 300
            elif symbol_type == "crypto":
                min_stop_pips = 50
                typical_stop_pips = 300 if atr_value is None else (atr_value / pip_size) * 1.5
                max_stop_pips = 1000
            else:
                min_stop_pips = 10
                typical_stop_pips = 50 if atr_value is None else (atr_value / pip_size) * 1.5
                max_stop_pips = 200
            
            # Convert stop loss distance to pips
            stop_distance_pips = stop_distance_price / pip_size
            
            # Check if provided stop is reasonable, otherwise use typical value
            if stop_distance_pips < min_stop_pips:
                logger.warning(f"{symbol}: Stop distance too small ({stop_distance_pips:.1f} pips). Using {min_stop_pips} pips.")
                stop_distance_pips = min_stop_pips
            elif stop_distance_pips > max_stop_pips:
                logger.warning(f"{symbol}: Stop distance too large ({stop_distance_pips:.1f} pips). Using {max_stop_pips} pips.")
                stop_distance_pips = max_stop_pips
            
            # Calculate position size based on risk and stop distance
            risk_per_pip = risk_amount / stop_distance_pips
            
            # Convert to standard lots
            position_size = risk_per_pip / (pip_size * 10000)
            
            # Apply broker-specific adjustments
            contract_size = symbol_info.trade_contract_size
            if contract_size > 0:
                position_size = position_size / (contract_size / 100000)
            
            # Check symbol exposure limits
            open_exposure = self._get_symbol_exposure(symbol)
            max_exposure = 10.0  # Maximum 10 lots per symbol
            
            if open_exposure + position_size > max_exposure:
                position_size = max(0, max_exposure - open_exposure)
                logger.warning(f"Limiting position size on {symbol} due to existing exposure")
            
            # Round to broker's lot step
            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step
            position_size = round(max(min_lot, position_size) / lot_step) * lot_step
            
            # Apply maximum limit
            position_size = min(position_size, 100.0)  # Cap at 100 lots for safety
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.01  # Return minimum lot size as fallback
    
    def execute_trade(self, symbol, trade_type, lot_size, entry_price=0, stop_loss=0, take_profit=0):
        """Execute a trade on MT5 with better error handling for non-tradable symbols"""
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
            is_visible = bool(symbol_info.visible)
            is_tradable = bool(is_visible and getattr(symbol_info, 'trade_mode', 0) == 0)
            
            if not is_tradable:
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
                        "confirmations": pa_signals.get("confirmations", []),  # Use get with default
                        "warnings": pa_signals.get("warnings", []),  # Use get with default
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
            # Use major forex pairs as benchmarks
            benchmark_symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
            benchmark_data = {}
            
            # First try to get data for major pairs
            for symbol in benchmark_symbols:
                df = self.get_market_data(symbol, "H1", 50)
                if df is not None and len(df) > 0:
                    benchmark_data[symbol] = df
            
            # If we don't have enough benchmark data, use any available symbol data
            if len(benchmark_data) < 2 and hasattr(self, 'analysis_priority_symbols'):
                for symbol in self.analysis_priority_symbols[:5]:  # Try top 5 symbols
                    if symbol not in benchmark_data:
                        df = self.get_market_data(symbol, "H1", 50)
                        if df is not None and len(df) > 0:
                            benchmark_data[symbol] = df
                            if len(benchmark_data) >= 3:  # Stop once we have enough data
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
    
    # Fix for the run_trading_cycle function - ensure all signal dictionaries have consistent keys

    def _analyze_symbol_timeframe(self, symbol, timeframe):
        """Analyze a single symbol and timeframe for signals"""
        try:
            # Get and process market data
            df = self.get_market_data(symbol, timeframe)
            if df is None or len(df) == 0:
                return []
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            if df is None:
                return []
            
            # Analyze price action
            pa_signals, levels = self.analyze_price_action(df)
            if pa_signals is None:
                return []
            
            # If we have a signal with sufficient strength
            if (pa_signals["buy"] or pa_signals["sell"]) and pa_signals["strength"] >= 5:
                signal = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "action": "buy" if pa_signals["buy"] else "sell",
                    "strength": pa_signals["strength"],
                    "patterns": pa_signals["patterns"],
                    "confirmations": pa_signals.get("confirmations", []),
                    "warnings": pa_signals.get("warnings", []),
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
                
                return [signal]
            return []
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
            return []

    def run_trading_cycle(self):
        """Enhanced trading cycle with improved signal collection across all timeframes"""
        if not self.connected:
            logger.info("Not connected to MT5. Attempting to connect...")
            if not self.connect():
                return []
        
        all_signals = []
        
        # Get analysis parameters from config
        symbols_to_analyze = self.config.get("analysis", {}).get("symbols_to_analyze", 30)
        min_signal_strength = self.config.get("analysis", {}).get("min_signal_strength", 4.0)
        max_warnings = self.config.get("analysis", {}).get("max_warnings", 3)
        
        # Get available symbols if we haven't already
        if not hasattr(self, 'analysis_priority_symbols') or not self.analysis_priority_symbols:
            self.get_available_symbols()
        
        # Use the priority symbols determined during symbol discovery
        top_symbols = self.analysis_priority_symbols if hasattr(self, 'analysis_priority_symbols') else []
        
        # Filter out symbols that have consistently failed
        top_symbols = [s for s in top_symbols 
                    if s not in self.failed_symbols or self.failed_symbols[s]['count'] <= 20]
        
        # Calculate market condition metrics
        market_condition = self._analyze_market_condition()
        
        # Create analysis tasks using asyncio
        try:
            # Randomize to avoid always analyzing the same symbols first
            import random
            random.shuffle(top_symbols)
            
            # Create tasks for all symbol/timeframe combinations
            async def analyze_all():
                # Create a thread pool with limited concurrency to avoid overloading MT5
                with ThreadPoolExecutor(max_workers=5) as executor:
                    loop = asyncio.get_event_loop()
                    
                    # First batch - analysis tasks for top symbols across all timeframes
                    priority_tasks = []
                    
                    # Make sure we have timeframes defined
                    all_timeframes = self.config["trading"]["timeframes"]
                    if not all_timeframes:
                        all_timeframes = ["M5", "M15", "H1"]
                    
                    for symbol in top_symbols:
                        # Analyze all timeframes for each symbol
                        for timeframe in all_timeframes:
                            priority_tasks.append(loop.run_in_executor(
                                executor,
                                self._analyze_symbol_timeframe,
                                symbol,
                                timeframe
                            ))
                    
                    # Run priority tasks first (top symbols)
                    logger.info(f"Analyzing {len(priority_tasks)} symbol/timeframe combinations")
                    start_time = time.time()
                    
                    # Run tasks in smaller batches to maintain responsiveness
                    batch_size = 10
                    for i in range(0, len(priority_tasks), batch_size):
                        batch = priority_tasks[i:i+batch_size]
                        batch_results = await asyncio.gather(*batch)
                        
                        # Collect signals from this batch
                        for signals in batch_results:
                            if signals:
                                all_signals.extend(signals)
                        
                        # Add short pause between batches to keep interface responsive
                        await asyncio.sleep(0.1)
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Analysis completed in {elapsed:.2f}s, found {len(all_signals)} signals")
                    
                    return all_signals
                
            # Run the asyncio event loop with a timeout to prevent hanging
            if hasattr(asyncio, 'run'):  # Python 3.7+
                signals_list = asyncio.run(analyze_all())
            else:  # Python 3.6 and below
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                signals_list = loop.run_until_complete(analyze_all())
                loop.close()
            
            logger.info(f"Async analysis completed, found {len(signals_list)} initial signals")
            
        except Exception as e:
            logger.error(f"Error in asyncio analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to simple approach
            logger.warning("Falling back to sequential analysis")
            signals_list = []
            
            for symbol in top_symbols[:10]:  # Analyze fewer symbols in fallback mode
                for timeframe in self.config["trading"]["timeframes"]:
                    signals = self._analyze_symbol_timeframe(symbol, timeframe)
                    if signals:
                        signals_list.extend(signals)
        
        # Adjust signals based on market condition and add trade type classification
        for signal in signals_list:
            # Ensure all signals have standard fields
            if "confirmations" not in signal:
                signal["confirmations"] = []
            if "warnings" not in signal:
                signal["warnings"] = []
            
            # Add trade type classification
            if "trade_type" not in signal:
                signal["trade_type"] = self._classify_trade_type(signal["timeframe"], signal["strength"])
            
            # Add market condition adjustment
            if market_condition.get('bias') == 'bullish':
                if signal["action"] == "buy":
                    signal["strength"] = min(10, signal["strength"] + 0.5)
                    signal["confirmations"].append("aligns_with_bullish_market")
                else:
                    signal["strength"] = max(0, signal["strength"] - 0.5)
                    signal["warnings"].append("against_bullish_market")
            
            elif market_condition.get('bias') == 'bearish':
                if signal["action"] == "sell":
                    signal["strength"] = min(10, signal["strength"] + 0.5)
                    signal["confirmations"].append("aligns_with_bearish_market")
                else:
                    signal["strength"] = max(0, signal["strength"] - 0.5)
                    signal["warnings"].append("against_bearish_market")
        
        # Sort signals by strength and filter
        signals_list.sort(key=lambda x: x["strength"], reverse=True)
        
        # Filter to get signals with configurable strength and warnings
        filtered_signals = [
            s for s in signals_list 
            if s["strength"] >= min_signal_strength and len(s.get("warnings", [])) <= max_warnings
        ]
        
        # Log signal summary
        if filtered_signals:
            logger.info(f"Found {len(filtered_signals)} trading signals after filtering")
            for i, signal in enumerate(filtered_signals[:5]):  # Log top 5 signals
                trade_type = signal.get('trade_type', '')
                logger.info(f"Signal {i+1}: {signal['symbol']} {signal['timeframe']} {signal['action'].upper()} "
                        f"(Strength: {signal['strength']:.1f}) - {trade_type}")
        
        return filtered_signals
    
    def auto_trade(self, signals):
        """Automatically execute trades based on signals - only for visible/tradable symbols"""
        if not self.connected or not signals:
            return []
        
        # Check how many positions are already open
        positions = mt5.positions_get()
        open_positions_count = len(positions) if positions is not None else 0
        max_positions = self.config["trading"]["max_open_trades"]
        
        # Make sure we have the list of tradable symbols (visible symbols)
        if not hasattr(self, 'tradable_symbols'):
            self.get_available_symbols()
                
        # Filter signals to only tradable (visible) symbols
        tradable_signals = [signal for signal in signals 
                        if signal["symbol"] in getattr(self, 'tradable_symbols', [])]
        
        if len(tradable_signals) < len(signals):
            logger.info(f"Filtered out {len(signals) - len(tradable_signals)} non-visible symbols from auto-trading")
        
        if not tradable_signals:
            logger.info("No tradable signals available for auto-trading")
            return []
        
        executed_trades = []
        
        # If we have room for more positions
        if open_positions_count < max_positions:
            available_slots = max_positions - open_positions_count
            
            # Take the strongest signals first, limited by available slots
            for signal in tradable_signals[:available_slots]:
                try:
                    # Apply risk percentage from config
                    # This ensures the risk setting is properly used for every trade
                    lot_size = self.calculate_position_size(
                        signal["symbol"],
                        signal["entry"],
                        signal["stop_loss"]
                    )
                    
                    # Update lot size in the signal
                    signal["lot_size"] = lot_size
                    
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
                except Exception as e:
                    logger.error(f"Error executing trade for {signal['symbol']}: {e}")
        
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
        """Main trading loop with more frequent symbol updates and better scheduling"""
        logger.info("Trading loop started")
        
        # Get intervals from config
        analysis_interval = self.config.get("analysis", {}).get("analysis_interval", 60)
        symbol_refresh_interval = self.config.get("analysis", {}).get("symbol_refresh_interval", 600)
        repeat_interval = self.config.get("analysis", {}).get("repeat_analysis_interval", 600)
        
        last_analysis_time = datetime.now() - timedelta(minutes=5)
        last_symbol_refresh_time = datetime.now() - timedelta(minutes=30)  # Force immediate symbol refresh
        
        # Track analyzed symbols and their signals
        recently_analyzed = {}  # {symbol: last_analyzed_time}
        
        while self.running:
            try:
                # Make sure we're connected
                if not self.connected:
                    if not self.connect():
                        logger.error("Failed to connect to MT5, retrying in 5 seconds")
                        time.sleep(5)
                        continue
                
                # Regularly refresh available symbols list
                current_time = datetime.now()
                time_since_symbol_refresh = (current_time - last_symbol_refresh_time).total_seconds()
                if time_since_symbol_refresh >= symbol_refresh_interval:
                    logger.info("Refreshing available symbols list")
                    self.get_available_symbols()  # This will update analysis_priority_symbols
                    last_symbol_refresh_time = current_time
                
                # Manage existing positions frequently
                self.manage_open_positions()
                
                # Run full analysis periodically
                time_since_last = (current_time - last_analysis_time).total_seconds()
                if time_since_last >= analysis_interval:
                    # Update recently analyzed symbols - remove expired entries
                    recently_analyzed = {
                        symbol: timestamp for symbol, timestamp in recently_analyzed.items()
                        if (current_time - timestamp).total_seconds() < repeat_interval
                    }
                    
                    # Run analysis on the most current symbol list
                    signals = self.run_trading_cycle()
                    last_analysis_time = current_time
                    
                    # Execute trades if auto-trading is enabled
                    if self.config.get("auto_trading", True) and signals:
                        # Filter out recently analyzed symbols
                        new_signals = [s for s in signals if s['symbol'] not in recently_analyzed]
                        
                        # Mark these symbols as recently analyzed
                        for signal in new_signals:
                            recently_analyzed[signal['symbol']] = current_time
                        
                        if new_signals:
                            executed_trades = self.auto_trade(new_signals)
                            if executed_trades:
                                logger.info(f"Executed {len(executed_trades)} trades")
                        else:
                            logger.info("No new trading opportunities after filtering recently analyzed symbols")
                    
                    # Adaptive analysis interval based on market activity
                    if signals:
                        # More signals = more market activity = analyze more frequently (min 30 seconds)
                        analysis_interval = max(30, 90 - len(signals) * 5)
                    else:
                        # No signals = less market activity = analyze less frequently (max 2 minutes)
                        analysis_interval = min(120, analysis_interval + 10)
                    
                    # Save updated analysis interval to config if we've made an adjustment
                    if "analysis" not in self.config:
                        self.config["analysis"] = {}
                    self.config["analysis"]["analysis_interval"] = analysis_interval
                
                # Brief sleep to keep the thread responsive
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(5)  # Short sleep after error
        
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