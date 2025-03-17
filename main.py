import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime
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
        """Calculate technical indicators on the provided DataFrame"""
        if df is None or len(df) < 50:
            return None
        
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # RSI
            if self.config["strategy"]["indicators"]["rsi"]["enabled"]:
                period = self.config["strategy"]["indicators"]["rsi"]["period"]
                df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
            
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
            
            # Price action features
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df.apply(
                lambda row: max(row['high'] - row['close'], row['high'] - row['open']), 
                axis=1
            )
            df['lower_wick'] = df.apply(
                lambda row: max(row['open'] - row['low'], row['close'] - row['low']), 
                axis=1
            )
            df['candle_range'] = df['high'] - df['low']
            
            # Support and resistance levels
            if self.config["strategy"]["indicators"]["support_resistance"]["enabled"]:
                lookback = self.config["strategy"]["indicators"]["support_resistance"]["lookback"]
                threshold = self.config["strategy"]["indicators"]["support_resistance"]["threshold"]
                
                # Simple swing high/low detection
                df['is_high'] = False
                df['is_low'] = False
                
                for i in range(2, len(df)-2):
                    if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                        df['high'].iloc[i] > df['high'].iloc[i-2] and
                        df['high'].iloc[i] > df['high'].iloc[i+1] and
                        df['high'].iloc[i] > df['high'].iloc[i+2]):
                        df.loc[i, 'is_high'] = True
                    
                    if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                        df['low'].iloc[i] < df['low'].iloc[i-2] and
                        df['low'].iloc[i] < df['low'].iloc[i+1] and
                        df['low'].iloc[i] < df['low'].iloc[i+2]):
                        df.loc[i, 'is_low'] = True
            
            # Volume analysis
            if 'tick_volume' in df.columns and self.config["strategy"]["volume"]["enabled"]:
                df['volume_ma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_ma']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_price_action(self, df):
        """Analyze price action patterns in the data"""
        if df is None or len(df) < 10:
            return None, None
        
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Initialize signals
            signals = {
                'buy': False, 
                'sell': False,
                'strength': 0,  # 0-10 signal strength
                'patterns': []
            }
            
            min_candle_size = self.config["strategy"]["price_action"]["min_candle_size"]
            rejection_level = self.config["strategy"]["price_action"]["rejection_level"]
            
            # Bullish engulfing
            if (current['open'] < current['close'] and  # Current candle is bullish
                prev['open'] > prev['close'] and        # Previous candle is bearish
                current['open'] <= prev['close'] and    # Current opens below prev close
                current['close'] > prev['open'] and     # Current closes above prev open
                current['body_size'] > min_candle_size):
                signals['buy'] = True
                signals['strength'] += 2
                signals['patterns'].append('bullish_engulfing')
            
            # Bearish engulfing
            if (current['open'] > current['close'] and  # Current candle is bearish
                prev['open'] < prev['close'] and        # Previous candle is bullish
                current['open'] >= prev['close'] and    # Current opens above prev close
                current['close'] < prev['open'] and     # Current closes below prev open
                current['body_size'] > min_candle_size):
                signals['sell'] = True
                signals['strength'] += 2
                signals['patterns'].append('bearish_engulfing')
            
            # Bullish pin bar / hammer
            if (current['body_size'] < current['candle_range'] * 0.4 and  # Small body
                current['lower_wick'] > current['body_size'] * 2 and      # Long lower wick
                current['lower_wick'] > current['upper_wick'] * 3):       # Lower wick much longer than upper
                signals['buy'] = True
                signals['strength'] += 3
                signals['patterns'].append('hammer')
            
            # Bearish pin bar / shooting star
            if (current['body_size'] < current['candle_range'] * 0.4 and  # Small body
                current['upper_wick'] > current['body_size'] * 2 and      # Long upper wick
                current['upper_wick'] > current['lower_wick'] * 3):       # Upper wick much longer than lower
                signals['sell'] = True
                signals['strength'] += 3
                signals['patterns'].append('shooting_star')
            
            # Dojis - indecision
            if current['body_size'] < current['candle_range'] * 0.1:
                signals['patterns'].append('doji')
                signals['strength'] += 1  # Lower strength for doji alone
                
            # Check if price is at support/resistance
            recent_highs = df[df['is_high'] == True]['high'].tail(5).tolist()
            recent_lows = df[df['is_low'] == True]['low'].tail(5).tolist()
            
            current_price = current['close']
            
            # Check proximity to recent supports/resistances
            for level in recent_highs:
                if abs(current_price - level) / current_price < 0.002:  # Within 0.2% of resistance
                    if signals['sell']:
                        signals['strength'] += 2
                    if signals['buy']:  # Potential breakout - more risky
                        signals['strength'] -= 1
                    signals['patterns'].append('at_resistance')
            
            for level in recent_lows:
                if abs(current_price - level) / current_price < 0.002:  # Within 0.2% of support
                    if signals['buy']:
                        signals['strength'] += 2
                    if signals['sell']:  # Potential breakdown - more risky
                        signals['strength'] -= 1
                    signals['patterns'].append('at_support')
            
            # Check indicator confirmations
            if 'rsi' in df.columns:
                current_rsi = current['rsi']
                if current_rsi < self.config["strategy"]["indicators"]["rsi"]["oversold"] and signals['buy']:
                    signals['strength'] += 1
                    signals['patterns'].append('rsi_oversold')
                elif current_rsi > self.config["strategy"]["indicators"]["rsi"]["overbought"] and signals['sell']:
                    signals['strength'] += 1
                    signals['patterns'].append('rsi_overbought')
            
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                if current['macd'] > current['macd_signal'] and signals['buy']:
                    signals['strength'] += 1
                    signals['patterns'].append('macd_bullish')
                elif current['macd'] < current['macd_signal'] and signals['sell']:
                    signals['strength'] += 1
                    signals['patterns'].append('macd_bearish')
            
            # Volume confirmation
            if 'volume_ratio' in df.columns:
                if current['volume_ratio'] > self.config["strategy"]["volume"]["threshold"]:
                    signals['strength'] += 1
                    signals['patterns'].append('high_volume')
            
            # Cap strength at 10
            signals['strength'] = min(10, signals['strength'])
            
            # Get potential entry and stop levels based on price action
            entry_price = None
            stop_loss = None
            take_profit = None
            
            if signals['buy']:
                entry_price = current['close']
                # Stop loss below the recent low
                stop_loss = min(df['low'].tail(3).min(), current['low'] - current['candle_range']*0.5)
                # Take profit based on risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.config["trading"]["min_risk_reward"])
            
            elif signals['sell']:
                entry_price = current['close']
                # Stop loss above the recent high
                stop_loss = max(df['high'].tail(3).max(), current['high'] + current['candle_range']*0.5)
                # Take profit based on risk-reward ratio
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * self.config["trading"]["min_risk_reward"])
            
            levels = {
                'entry': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            return signals, levels
            
        except Exception as e:
            logger.error(f"Error analyzing price action: {e}")
            return None, None
    
    def calculate_position_size(self, symbol, entry, stop_loss):
        """Calculate position size based on account balance and risk settings"""
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
            
            # Risk calculation
            balance = account_info.balance
            risk_percent = min(self.config["trading"]["risk_percent"], self.config["trading"]["max_risk_percent"])
            risk_amount = balance * (risk_percent / 100.0)
            
            # Calculate pip value
            pip_size = 0.0001 if symbol_info.digits == 4 else 0.00001
            if "JPY" in symbol:
                pip_size = 0.01 if symbol_info.digits == 2 else 0.001
            
            # For indices and metals, get the appropriate pip size
            if symbol in ["XAUUSD", "XAGUSD", "US30", "NASDAQ", "SPX500"]:
                pip_size = 0.1 if symbol_info.digits == 1 else 0.01
            
            # Calculate stop loss distance in pips
            stop_distance_price = abs(entry - stop_loss)
            stop_distance_pips = stop_distance_price / pip_size
            
            # Calculate position size
            if stop_distance_pips > 0:
                position_size = risk_amount / stop_distance_price
                
                # Convert to lots - standard lot is 100,000 units
                lots = position_size / 100000
                
                # Adjust for contract size and tick value
                contract_size = symbol_info.trade_contract_size
                tick_value = symbol_info.trade_tick_value
                
                if contract_size > 0:
                    lots = lots / (contract_size / 100000)
                
                # Round to 2 decimals for mini lots
                lots = round(max(0.01, min(lots, 100.0)), 2)
                
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
    
    def run_trading_cycle(self):
        """Run one complete trading analysis cycle across all symbols"""
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
        
        for symbol in current_symbols:
            signals = self.analyze_symbol(symbol)
            if signals:
                all_signals.extend(signals)
        
        # Sort signals by strength
        all_signals.sort(key=lambda x: x["strength"], reverse=True)
        
        return all_signals
    
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
    
    def _trading_loop(self):
        """Main trading loop running in a separate thread"""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Make sure we're connected
                if not self.connected:
                    if not self.connect():
                        logger.error("Failed to connect to MT5, retrying in 30 seconds")
                        time.sleep(30)
                        continue
                
                # Manage existing positions
                self.manage_open_positions()
                
                # Run analysis and get trading signals
                signals = self.run_trading_cycle()
                
                # Execute trades if auto-trading is enabled
                if self.config.get("auto_trading", True) and signals:
                    logger.info(f"Found {len(signals)} potential trading opportunities")
                    executed_trades = self.auto_trade(signals)
                    if executed_trades:
                        logger.info(f"Executed {len(executed_trades)} trades")
                
                # Wait before next cycle
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(30)  # Wait longer after an error
        
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