import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import logging

# Import the trading bot
from main import TradingBot, logger

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MetaTrader 5 Trading Bot")
        self.root.geometry("1200x800")
        
        # Set application icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Initialize the trading bot
        self.bot = TradingBot()
        
        # Create GUI variables
        self.account_connected = tk.BooleanVar(value=False)
        self.bot_running = tk.BooleanVar(value=False)
        self.selected_symbol = tk.StringVar(value="EURUSD")
        self.selected_timeframe = tk.StringVar(value="H1")
        self.auto_trading = tk.BooleanVar(value=True)
        self.risk_percent = tk.DoubleVar(value=2.0)
        
        # Load theme preference
        self.theme = tk.StringVar(value="dark")
        self.load_theme()
        
        # Create main layout
        self.create_menu()
        self.create_notebook()
        self.create_status_bar()
        
        # Initialize chart data
        self.chart_data = None
        self.chart_canvas = None
        
        # Start update threads
        self.gui_update_running = True
        self.update_thread = threading.Thread(target=self.update_gui_data, daemon=True)
        self.update_thread.start()
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Connect to MT5", command=self.connect_to_mt5)
        file_menu.add_command(label="Disconnect", command=self.disconnect_from_mt5)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Bot menu
        bot_menu = tk.Menu(menubar, tearoff=0)
        bot_menu.add_command(label="Start Bot", command=self.start_bot)
        bot_menu.add_command(label="Stop Bot", command=self.stop_bot)
        bot_menu.add_separator()
        bot_menu.add_checkbutton(label="Auto Trading", variable=self.auto_trading, 
                                command=self.update_auto_trading)
        menubar.add_cascade(label="Bot", menu=bot_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Edit Settings", command=self.edit_settings)
        
        # Theme submenu
        theme_menu = tk.Menu(settings_menu, tearoff=0)
        theme_menu.add_radiobutton(label="Light", variable=self.theme, value="light", 
                                  command=self.apply_theme)
        theme_menu.add_radiobutton(label="Dark", variable=self.theme, value="dark", 
                                  command=self.apply_theme)
        settings_menu.add_cascade(label="Theme", menu=theme_menu)
        
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Documentation", command=self.open_documentation)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_notebook(self):
        """Create the notebook with tabs"""
        self.notebook = ttk.Notebook(self.root)
        
        # Dashboard tab
        self.dashboard_frame = ttk.Frame(self.notebook)
        self.create_dashboard_tab()
        self.notebook.add(self.dashboard_frame, text="Dashboard")
        
        # Trading tab
        self.trading_frame = ttk.Frame(self.notebook)
        self.create_trading_tab()
        self.notebook.add(self.trading_frame, text="Trading")
        
        # Charts tab
        self.charts_frame = ttk.Frame(self.notebook)
        self.create_charts_tab()
        self.notebook.add(self.charts_frame, text="Charts")
        
        # Positions tab
        self.positions_frame = ttk.Frame(self.notebook)
        self.create_positions_tab()
        self.notebook.add(self.positions_frame, text="Positions")
        
        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.create_logs_tab()
        self.notebook.add(self.logs_frame, text="Logs")
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.create_settings_tab()
        self.notebook.add(self.settings_frame, text="Settings")
        
        self.notebook.pack(expand=1, fill="both", padx=5, pady=5)
    
    def create_dashboard_tab(self):
        """Create the dashboard tab"""
        # Account frame
        account_frame = ttk.LabelFrame(self.dashboard_frame, text="Account Information")
        account_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.account_labels = {}
        account_fields = ["Server", "Login", "Balance", "Equity", "Profit", "Margin", "Free Margin", "Margin Level"]
        
        for i, field in enumerate(account_fields):
            ttk.Label(account_frame, text=f"{field}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.account_labels[field.lower().replace(" ", "_")] = ttk.Label(account_frame, text="--")
            self.account_labels[field.lower().replace(" ", "_")].grid(row=i, column=1, sticky="w", padx=5, pady=2)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(self.dashboard_frame, text="Trading Statistics")
        stats_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.stats_labels = {}
        stats_fields = ["Total Trades", "Winning Trades", "Win Rate", "Profit Factor", "Average Win", "Average Loss"]
        
        for i, field in enumerate(stats_fields):
            ttk.Label(stats_frame, text=f"{field}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.stats_labels[field.lower().replace(" ", "_")] = ttk.Label(stats_frame, text="--")
            self.stats_labels[field.lower().replace(" ", "_")].grid(row=i, column=1, sticky="w", padx=5, pady=2)
        
        # Recent signals frame
        signals_frame = ttk.LabelFrame(self.dashboard_frame, text="Recent Trading Signals")
        signals_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Signals table
        columns = ("Time", "Symbol", "Action", "Timeframe", "Strength", "Entry", "SL", "TP")
        self.signals_tree = ttk.Treeview(signals_frame, columns=columns, show="headings")
        
        for col in columns:
            self.signals_tree.heading(col, text=col)
            self.signals_tree.column(col, width=80)
        
        self.signals_tree.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        signals_scrollbar = ttk.Scrollbar(signals_frame, orient="vertical", command=self.signals_tree.yview)
        signals_scrollbar.grid(row=0, column=1, sticky="ns")
        self.signals_tree.configure(yscrollcommand=signals_scrollbar.set)
        
        # Quick controls
        controls_frame = ttk.LabelFrame(self.dashboard_frame, text="Quick Controls")
        controls_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Connect button
        self.connect_button = ttk.Button(controls_frame, text="Connect to MT5", command=self.connect_to_mt5)
        self.connect_button.grid(row=0, column=0, padx=5, pady=5)
        
        # Start/Stop bot button
        self.bot_button = ttk.Button(controls_frame, text="Start Bot", command=self.toggle_bot)
        self.bot_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Auto trading checkbox
        self.auto_trading_check = ttk.Checkbutton(
            controls_frame, text="Auto Trading", variable=self.auto_trading, 
            command=self.update_auto_trading
        )
        self.auto_trading_check.grid(row=0, column=2, padx=5, pady=5)
        
        # Risk slider
        ttk.Label(controls_frame, text="Risk %:").grid(row=1, column=0, padx=5, pady=5)
        risk_slider = ttk.Scale(
            controls_frame, from_=0.5, to=20.0, orient="horizontal", 
            variable=self.risk_percent, command=self.update_risk
        )
        risk_slider.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.risk_value_label = ttk.Label(controls_frame, text="2.0%")
        self.risk_value_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Configure grid weights
        self.dashboard_frame.columnconfigure(0, weight=1)
        self.dashboard_frame.columnconfigure(1, weight=1)
        self.dashboard_frame.rowconfigure(1, weight=1)
        signals_frame.columnconfigure(0, weight=1)
        signals_frame.rowconfigure(0, weight=1)
    
    def create_trading_tab(self):
        """Create the trading tab"""
        # Symbol selection frame
        symbol_frame = ttk.LabelFrame(self.trading_frame, text="Symbol Selection")
        symbol_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(symbol_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.symbol_combo = ttk.Combobox(symbol_frame, textvariable=self.selected_symbol)
        self.symbol_combo.grid(row=0, column=1, padx=5, pady=5)
        self.symbol_combo['values'] = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NASDAQ", "SPX500"]
        
        ttk.Label(symbol_frame, text="Timeframe:").grid(row=1, column=0, padx=5, pady=5)
        self.timeframe_combo = ttk.Combobox(symbol_frame, textvariable=self.selected_timeframe)
        self.timeframe_combo.grid(row=1, column=1, padx=5, pady=5)
        self.timeframe_combo['values'] = ["M5", "M15", "M30", "H1", "H4", "D1"]
        
        ttk.Button(symbol_frame, text="Analyze", command=self.analyze_symbol).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Analysis results frame
        analysis_frame = ttk.LabelFrame(self.trading_frame, text="Analysis Results")
        analysis_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.analysis_text = scrolledtext.ScrolledText(analysis_frame, width=40, height=15)
        self.analysis_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Manual trading frame
        manual_frame = ttk.LabelFrame(self.trading_frame, text="Manual Trading")
        manual_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        ttk.Label(manual_frame, text="Action:").grid(row=0, column=0, padx=5, pady=5)
        self.action_combo = ttk.Combobox(manual_frame, values=["Buy", "Sell"])
        self.action_combo.grid(row=0, column=1, padx=5, pady=5)
        self.action_combo.current(0)
        
        ttk.Label(manual_frame, text="Lot Size:").grid(row=1, column=0, padx=5, pady=5)
        self.lot_entry = ttk.Entry(manual_frame)
        self.lot_entry.insert(0, "0.01")
        self.lot_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(manual_frame, text="Stop Loss:").grid(row=2, column=0, padx=5, pady=5)
        self.sl_entry = ttk.Entry(manual_frame)
        self.sl_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(manual_frame, text="Take Profit:").grid(row=3, column=0, padx=5, pady=5)
        self.tp_entry = ttk.Entry(manual_frame)
        self.tp_entry.grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Button(manual_frame, text="Execute Trade", command=self.execute_manual_trade).grid(row=4, column=0, columnspan=2, padx=5, pady=5)
        
        # Configure grid weights
        self.trading_frame.columnconfigure(0, weight=1)
        self.trading_frame.columnconfigure(1, weight=2)
        self.trading_frame.rowconfigure(0, weight=1)
        self.trading_frame.rowconfigure(1, weight=1)
    
    def create_charts_tab(self):
        """Create the charts tab"""
        # Controls frame
        controls_frame = ttk.Frame(self.charts_frame)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(controls_frame, text="Symbol:").grid(row=0, column=0, padx=5, pady=5)
        self.chart_symbol_combo = ttk.Combobox(controls_frame)
        self.chart_symbol_combo.grid(row=0, column=1, padx=5, pady=5)
        self.chart_symbol_combo['values'] = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NASDAQ", "SPX500"]
        self.chart_symbol_combo.current(0)
        
        ttk.Label(controls_frame, text="Timeframe:").grid(row=0, column=2, padx=5, pady=5)
        self.chart_tf_combo = ttk.Combobox(controls_frame)
        self.chart_tf_combo.grid(row=0, column=3, padx=5, pady=5)
        self.chart_tf_combo['values'] = ["M5", "M15", "M30", "H1", "H4", "D1"]
        self.chart_tf_combo.current(3)  # Default to H1
        
        ttk.Label(controls_frame, text="Bars:").grid(row=0, column=4, padx=5, pady=5)
        self.bars_combo = ttk.Combobox(controls_frame)
        self.bars_combo.grid(row=0, column=5, padx=5, pady=5)
        self.bars_combo['values'] = ["100", "200", "500", "1000"]
        self.bars_combo.current(1)  # Default to 200 bars
        
        ttk.Button(controls_frame, text="Load Chart", command=self.load_chart).grid(row=0, column=6, padx=5, pady=5)
        
        # Chart frame
        self.chart_frame = ttk.Frame(self.charts_frame)
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initial message
        ttk.Label(self.chart_frame, text="Click 'Load Chart' to display price data").pack(pady=50)
    
    def create_positions_tab(self):
        """Create the positions tab"""
        # Positions table
        columns = ("Ticket", "Symbol", "Type", "Volume", "Open Price", "Current Price", "SL", "TP", "Profit", "Open Time")
        self.positions_tree = ttk.Treeview(self.positions_frame, columns=columns, show="headings")
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            width = 80 if col not in ["Open Time", "Symbol"] else 120
            self.positions_tree.column(col, width=width)
        
        self.positions_tree.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add scrollbar
        positions_scrollbar = ttk.Scrollbar(self.positions_frame, orient="vertical", command=self.positions_tree.yview)
        positions_scrollbar.place(relx=1, rely=0, relheight=1, anchor="ne")
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        # Button frame
        button_frame = ttk.Frame(self.positions_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Refresh", command=self.refresh_positions).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close Selected", command=self.close_selected_position).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Close All", command=self.close_all_positions).pack(side="left", padx=5)
    
    def create_logs_tab(self):
        """Create the logs tab"""
        # Log text area
        self.log_text = scrolledtext.ScrolledText(self.logs_frame)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.log_text.config(state="disabled")
        
        # Button frame
        button_frame = ttk.Frame(self.logs_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(button_frame, text="Clear Logs", command=self.clear_logs).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Logs", command=self.save_logs).pack(side="left", padx=5)
        
        # Create a custom handler to redirect logs to the GUI
        self.setup_log_handler()
    
    def create_settings_tab(self):
        """Create the settings tab"""
        # Main settings frame
        settings_frame = ttk.Frame(self.settings_frame)
        settings_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a notebook for categorized settings
        settings_notebook = ttk.Notebook(settings_frame)
        
        # Account settings
        account_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(account_frame, text="Account")
        
        ttk.Label(account_frame, text="MT5 Path:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.mt5_path_entry = ttk.Entry(account_frame, width=50)
        self.mt5_path_entry.insert(0, self.bot.config["account"]["path"])
        self.mt5_path_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(account_frame, text="Login:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.login_entry = ttk.Entry(account_frame)
        self.login_entry.insert(0, str(self.bot.config["account"]["login"]))
        self.login_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(account_frame, text="Password:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.password_entry = ttk.Entry(account_frame, show="*")
        self.password_entry.insert(0, self.bot.config["account"]["password"])
        self.password_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(account_frame, text="Server:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.server_entry = ttk.Entry(account_frame)
        self.server_entry.insert(0, self.bot.config["account"]["server"])
        self.server_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Trading settings
        trading_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(trading_frame, text="Trading")
        
        ttk.Label(trading_frame, text="Symbols (comma-separated):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.symbols_entry = ttk.Entry(trading_frame, width=50)
        self.symbols_entry.insert(0, ",".join(self.bot.config["trading"]["symbols"]))
        self.symbols_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Timeframes (comma-separated):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.timeframes_entry = ttk.Entry(trading_frame)
        self.timeframes_entry.insert(0, ",".join(self.bot.config["trading"]["timeframes"]))
        self.timeframes_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Risk %:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.risk_entry = ttk.Entry(trading_frame)
        self.risk_entry.insert(0, str(self.bot.config["trading"]["risk_percent"]))
        self.risk_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Max Risk %:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.max_risk_entry = ttk.Entry(trading_frame)
        self.max_risk_entry.insert(0, str(self.bot.config["trading"]["max_risk_percent"]))
        self.max_risk_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Min Risk:Reward:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.min_rr_entry = ttk.Entry(trading_frame)
        self.min_rr_entry.insert(0, str(self.bot.config["trading"]["min_risk_reward"]))
        self.min_rr_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Max Open Trades:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.max_trades_entry = ttk.Entry(trading_frame)
        self.max_trades_entry.insert(0, str(self.bot.config["trading"]["max_open_trades"]))
        self.max_trades_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        # Trailing stop settings
        self.use_trailing = tk.BooleanVar(value=self.bot.config["trading"]["use_trailing_stop"])
        ttk.Checkbutton(
            trading_frame, text="Use Trailing Stop", variable=self.use_trailing
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Trailing Activation (pips):").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.trailing_activation_entry = ttk.Entry(trading_frame)
        self.trailing_activation_entry.insert(0, str(self.bot.config["trading"]["trailing_stop_activation"]))
        self.trailing_activation_entry.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Trailing Distance (pips):").grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.trailing_distance_entry = ttk.Entry(trading_frame)
        self.trailing_distance_entry.insert(0, str(self.bot.config["trading"]["trailing_stop_distance"]))
        self.trailing_distance_entry.grid(row=8, column=1, sticky="w", padx=5, pady=5)
        
        # Strategy settings
        strategy_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(strategy_frame, text="Strategy")
        
        # Price action settings
        price_action_frame = ttk.LabelFrame(strategy_frame, text="Price Action")
        price_action_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.use_price_action = tk.BooleanVar(value=self.bot.config["strategy"]["price_action"]["enabled"])
        ttk.Checkbutton(
            price_action_frame, text="Enable Price Action", variable=self.use_price_action
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(price_action_frame, text="Min Candle Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.min_candle_entry = ttk.Entry(price_action_frame)
        self.min_candle_entry.insert(0, str(self.bot.config["strategy"]["price_action"]["min_candle_size"]))
        self.min_candle_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(price_action_frame, text="Rejection Level:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.rejection_entry = ttk.Entry(price_action_frame)
        self.rejection_entry.insert(0, str(self.bot.config["strategy"]["price_action"]["rejection_level"]))
        self.rejection_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(price_action_frame, text="Confirmation Candles:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.confirm_candles_entry = ttk.Entry(price_action_frame)
        self.confirm_candles_entry.insert(0, str(self.bot.config["strategy"]["price_action"]["confirmation_candles"]))
        self.confirm_candles_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Volume settings
        volume_frame = ttk.LabelFrame(strategy_frame, text="Volume Analysis")
        volume_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.use_volume = tk.BooleanVar(value=self.bot.config["strategy"]["volume"]["enabled"])
        ttk.Checkbutton(
            volume_frame, text="Enable Volume Analysis", variable=self.use_volume
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(volume_frame, text="Volume Threshold:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.volume_threshold_entry = ttk.Entry(volume_frame)
        self.volume_threshold_entry.insert(0, str(self.bot.config["strategy"]["volume"]["threshold"]))
        self.volume_threshold_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Indicators frame
        indicators_frame = ttk.LabelFrame(strategy_frame, text="Indicators")
        indicators_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        
        # RSI settings
        self.use_rsi = tk.BooleanVar(value=self.bot.config["strategy"]["indicators"]["rsi"]["enabled"])
        ttk.Checkbutton(
            indicators_frame, text="RSI", variable=self.use_rsi
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Period:").grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.rsi_period_entry = ttk.Entry(indicators_frame, width=8)
        self.rsi_period_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["rsi"]["period"]))
        self.rsi_period_entry.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Overbought:").grid(row=0, column=3, sticky="w", padx=5, pady=5)
        self.rsi_ob_entry = ttk.Entry(indicators_frame, width=8)
        self.rsi_ob_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["rsi"]["overbought"]))
        self.rsi_ob_entry.grid(row=0, column=4, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Oversold:").grid(row=0, column=5, sticky="w", padx=5, pady=5)
        self.rsi_os_entry = ttk.Entry(indicators_frame, width=8)
        self.rsi_os_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["rsi"]["oversold"]))
        self.rsi_os_entry.grid(row=0, column=6, sticky="w", padx=5, pady=5)
        
        # MACD settings
        self.use_macd = tk.BooleanVar(value=self.bot.config["strategy"]["indicators"]["macd"]["enabled"])
        ttk.Checkbutton(
            indicators_frame, text="MACD", variable=self.use_macd
        ).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Fast:").grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.macd_fast_entry = ttk.Entry(indicators_frame, width=8)
        self.macd_fast_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["macd"]["fast"]))
        self.macd_fast_entry.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Slow:").grid(row=1, column=3, sticky="w", padx=5, pady=5)
        self.macd_slow_entry = ttk.Entry(indicators_frame, width=8)
        self.macd_slow_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["macd"]["slow"]))
        self.macd_slow_entry.grid(row=1, column=4, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Signal:").grid(row=1, column=5, sticky="w", padx=5, pady=5)
        self.macd_signal_entry = ttk.Entry(indicators_frame, width=8)
        self.macd_signal_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["macd"]["signal"]))
        self.macd_signal_entry.grid(row=1, column=6, sticky="w", padx=5, pady=5)
        
        # Support/Resistance settings
        self.use_sr = tk.BooleanVar(value=self.bot.config["strategy"]["indicators"]["support_resistance"]["enabled"])
        ttk.Checkbutton(
            indicators_frame, text="Support/Resistance", variable=self.use_sr
        ).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Lookback:").grid(row=2, column=1, sticky="w", padx=5, pady=5)
        self.sr_lookback_entry = ttk.Entry(indicators_frame, width=8)
        self.sr_lookback_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["support_resistance"]["lookback"]))
        self.sr_lookback_entry.grid(row=2, column=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(indicators_frame, text="Threshold:").grid(row=2, column=3, sticky="w", padx=5, pady=5)
        self.sr_threshold_entry = ttk.Entry(indicators_frame, width=8)
        self.sr_threshold_entry.insert(0, str(self.bot.config["strategy"]["indicators"]["support_resistance"]["threshold"]))
        self.sr_threshold_entry.grid(row=2, column=4, sticky="w", padx=5, pady=5)
        
        # Save and cancel buttons
        button_frame = ttk.Frame(settings_frame)
        button_frame.pack(fill="x", side="bottom", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.load_settings).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_settings).pack(side="left", padx=5)
        
        # Add notebook to frame
        settings_notebook.pack(fill="both", expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create status bar at the bottom of the window"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side="bottom", fill="x")
        
        # Connection status
        self.connection_label = ttk.Label(self.status_bar, text="Disconnected", foreground="red")
        self.connection_label.pack(side="left", padx=10)
        
        # Bot status
        self.bot_status_label = ttk.Label(self.status_bar, text="Bot: Stopped", foreground="red")
        self.bot_status_label.pack(side="left", padx=10)
        
        # Last update time
        self.update_time_label = ttk.Label(self.status_bar, text="Last Update: Never")
        self.update_time_label.pack(side="right", padx=10)
        
        # Balance
        self.balance_label = ttk.Label(self.status_bar, text="Balance: 0.00")
        self.balance_label.pack(side="right", padx=10)
    
    def connect_to_mt5(self):
        """Connect to MetaTrader 5 platform"""
        if not self.account_connected.get():
            result = self.bot.connect()
            if result:
                self.account_connected.set(True)
                self.connection_label.config(text="Connected", foreground="green")
                messagebox.showinfo("Connection", "Successfully connected to MetaTrader 5")
                self.refresh_account_info()
                self.refresh_positions()
                self.connect_button.config(text="Disconnect from MT5")
            else:
                messagebox.showerror("Connection Error", "Failed to connect to MetaTrader 5")
        else:
            self.bot.disconnect()
            self.account_connected.set(False)
            self.connection_label.config(text="Disconnected", foreground="red")
            self.connect_button.config(text="Connect to MT5")
    
    def disconnect_from_mt5(self):
        """Disconnect from MetaTrader 5 platform"""
        if self.account_connected.get():
            self.bot.disconnect()
            self.account_connected.set(False)
            self.connection_label.config(text="Disconnected", foreground="red")
            messagebox.showinfo("Disconnection", "Disconnected from MetaTrader 5")
            self.connect_button.config(text="Connect to MT5")
    
    def start_bot(self):
        """Start the trading bot"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Please connect to MetaTrader 5 first")
            return
            
        if not self.bot_running.get():
            result = self.bot.start()
            if result:
                self.bot_running.set(True)
                self.bot_status_label.config(text="Bot: Running", foreground="green")
                self.bot_button.config(text="Stop Bot")
                messagebox.showinfo("Bot Started", "Trading bot has been started")
            else:
                messagebox.showerror("Error", "Failed to start trading bot")
    
    def stop_bot(self):
        """Stop the trading bot"""
        if self.bot_running.get():
            result = self.bot.stop()
            if result:
                self.bot_running.set(False)
                self.bot_status_label.config(text="Bot: Stopped", foreground="red")
                self.bot_button.config(text="Start Bot")
                messagebox.showinfo("Bot Stopped", "Trading bot has been stopped")
            else:
                messagebox.showerror("Error", "Failed to stop trading bot")
    
    def toggle_bot(self):
        """Toggle bot running state"""
        if self.bot_running.get():
            self.stop_bot()
        else:
            self.start_bot()
    
    def update_auto_trading(self):
        """Update auto trading setting"""
        self.bot.config["auto_trading"] = self.auto_trading.get()
        self.bot.save_config()
    
    def update_risk(self, *args):
        """Update risk percentage setting and label"""
        value = round(float(self.risk_percent.get()), 1)
        self.risk_value_label.config(text=f"{value}%")
        self.bot.config["trading"]["risk_percent"] = value
        self.bot.save_config()
    
    def refresh_account_info(self):
        """Refresh account information"""
        if not self.account_connected.get():
            return
            
        account_info = self.bot.get_account_summary()
        if account_info:
            # Update account labels
            self.account_labels["server"].config(text=self.bot.config["account"]["server"])
            self.account_labels["login"].config(text=str(self.bot.config["account"]["login"]))
            self.account_labels["balance"].config(text=f"{account_info['balance']:.2f}")
            self.account_labels["equity"].config(text=f"{account_info['equity']:.2f}")
            self.account_labels["profit"].config(
                text=f"{account_info['profit']:.2f}",
                foreground="green" if account_info['profit'] >= 0 else "red"
            )
            self.account_labels["margin"].config(text=f"{account_info['margin']:.2f}")
            self.account_labels["free_margin"].config(text=f"{account_info['free_margin']:.2f}")
            self.account_labels["margin_level"].config(text=f"{account_info['margin_level']:.2f}%" if account_info['margin_level'] else "0.00%")
            
            # Update status bar
            self.balance_label.config(text=f"Balance: {account_info['balance']:.2f} {account_info['currency']}")
    
    def refresh_positions(self):
        """Refresh open positions"""
        if not self.account_connected.get():
            return
            
        # Clear existing positions
        for item in self.positions_tree.get_children():
            self.positions_tree.delete(item)
            
        positions = self.bot.get_open_positions()
        for pos in positions:
            self.positions_tree.insert(
                "", "end",
                values=(
                    pos["ticket"],
                    pos["symbol"],
                    pos["type"].upper(),
                    pos["volume"],
                    f"{pos['open_price']:.5f}",
                    f"{pos['current_price']:.5f}",
                    f"{pos['sl']:.5f}" if pos['sl'] else "-",
                    f"{pos['tp']:.5f}" if pos['tp'] else "-",
                    f"{pos['profit']:.2f}",
                    pos["open_time"]
                )
            )
    
    def close_selected_position(self):
        """Close selected position"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
            
        selected = self.positions_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "No position selected")
            return
            
        item = self.positions_tree.item(selected[0])
        ticket = item["values"][0]
        symbol = item["values"][1]
        
        if messagebox.askyesno("Close Position", f"Close position #{ticket} for {symbol}?"):
            result = self.bot.close_position(ticket)
            if result:
                messagebox.showinfo("Success", f"Position #{ticket} closed successfully")
                self.refresh_positions()
            else:
                messagebox.showerror("Error", f"Failed to close position #{ticket}")
    
    def close_all_positions(self):
        """Close all open positions"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
            
        if messagebox.askyesno("Close All Positions", "Close all open positions?"):
            positions = self.bot.get_open_positions()
            closed = 0
            failed = 0
            
            for pos in positions:
                if self.bot.close_position(pos["ticket"]):
                    closed += 1
                else:
                    failed += 1
            
            if failed == 0:
                messagebox.showinfo("Success", f"All {closed} positions closed successfully")
            else:
                messagebox.showwarning("Warning", f"Closed {closed} positions, {failed} failed")
                
            self.refresh_positions()
    
    def analyze_symbol(self):
        """Analyze selected symbol"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
            
        symbol = self.selected_symbol.get()
        timeframe = self.selected_timeframe.get()
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, f"Analyzing {symbol} on {timeframe} timeframe...\n\n")
        
        signals = self.bot.analyze_symbol(symbol)
        
        if not signals:
            self.analysis_text.insert(tk.END, "No tradable signals found.")
            return
        
        # Find signal for our timeframe if it exists
        tf_signal = None
        for signal in signals:
            if signal["timeframe"] == timeframe:
                tf_signal = signal
                break
        
        # Display the analysis
        if tf_signal:
            self.analysis_text.insert(tk.END, f"Signal: {tf_signal['action'].upper()}\n")
            self.analysis_text.insert(tk.END, f"Strength: {tf_signal['strength']}/10\n")
            self.analysis_text.insert(tk.END, f"Patterns: {', '.join(tf_signal['patterns'])}\n\n")
            
            self.analysis_text.insert(tk.END, f"Entry price: {tf_signal['entry']:.5f}\n")
            self.analysis_text.insert(tk.END, f"Stop loss: {tf_signal['stop_loss']:.5f}\n")
            self.analysis_text.insert(tk.END, f"Take profit: {tf_signal['take_profit']:.5f}\n")
            self.analysis_text.insert(tk.END, f"Position size: {tf_signal['lot_size']:.2f} lots\n\n")
            
            # Calculate risk metrics
            if tf_signal["entry"] and tf_signal["stop_loss"]:
                risk_pips = abs(tf_signal["entry"] - tf_signal["stop_loss"]) * 10000
                reward_pips = abs(tf_signal["entry"] - tf_signal["take_profit"]) * 10000
                rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                
                self.analysis_text.insert(tk.END, f"Risk: {risk_pips:.1f} pips\n")
                self.analysis_text.insert(tk.END, f"Reward: {reward_pips:.1f} pips\n")
                self.analysis_text.insert(tk.END, f"Risk/Reward ratio: 1:{rr_ratio:.2f}\n")
            
            # Pre-fill the manual trading fields
            self.action_combo.set(tf_signal['action'].capitalize())
            self.lot_entry.delete(0, tk.END)
            self.lot_entry.insert(0, f"{tf_signal['lot_size']:.2f}")
            self.sl_entry.delete(0, tk.END)
            self.sl_entry.insert(0, f"{tf_signal['stop_loss']:.5f}")
            self.tp_entry.delete(0, tk.END)
            self.tp_entry.insert(0, f"{tf_signal['take_profit']:.5f}")
        else:
            # Just show the strongest signal
            strongest = signals[0]
            self.analysis_text.insert(tk.END, f"No signal for {timeframe} timeframe.\n")
            self.analysis_text.insert(tk.END, f"Strongest signal ({strongest['timeframe']}):\n\n")
            self.analysis_text.insert(tk.END, f"Action: {strongest['action'].upper()}\n")
            self.analysis_text.insert(tk.END, f"Strength: {strongest['strength']}/10\n")
            self.analysis_text.insert(tk.END, f"Patterns: {', '.join(strongest['patterns'])}\n")
    
    def execute_manual_trade(self):
        """Execute a manual trade based on user input"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
            
        symbol = self.selected_symbol.get()
        action = self.action_combo.get().lower()
        
        try:
            lot_size = float(self.lot_entry.get())
            sl = float(self.sl_entry.get()) if self.sl_entry.get() else 0
            tp = float(self.tp_entry.get()) if self.tp_entry.get() else 0
            
            if lot_size <= 0:
                messagebox.showerror("Error", "Lot size must be greater than zero")
                return
                
            if messagebox.askyesno("Confirm Trade", f"Execute {action} order for {lot_size} lots of {symbol}?"):
                result = self.bot.execute_trade(symbol, action, lot_size, 0, sl, tp)
                
                if result is not None:
                    messagebox.showinfo("Success", f"Order executed successfully. Ticket: {result.order}")
                    self.refresh_positions()
                else:
                    messagebox.showerror("Error", "Failed to execute trade")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please check lot size, SL, and TP values")
    
    def load_chart(self):
        """Load and display a chart"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
            
        symbol = self.chart_symbol_combo.get()
        timeframe = self.chart_tf_combo.get()
        bars = int(self.bars_combo.get())
        
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Get market data
        df = self.bot.get_market_data(symbol, timeframe, bars)
        
        if df is None or len(df) == 0:
            ttk.Label(self.chart_frame, text="No data available for the selected symbol/timeframe").pack(pady=50)
            return
        
        # Add indicators
        df = self.bot.calculate_indicators(df)
        
        # Set datetime as index
        df.set_index("time", inplace=True)
        
        # Create figure
        fig = plt.figure(figsize=(10, 6), dpi=100)
        
        # Style for the chart based on theme
        if self.theme.get() == "dark":
            plt.style.use("dark_background")
            candle_up = "#26a69a"
            candle_down = "#ef5350"
        else:
            plt.style.use("default")
            candle_up = "#26a69a"
            candle_down = "#ef5350"
        
        # Create candlestick chart
        mpf_style = mpf.make_mpf_style(
            base_mpf_style="yahoo" if self.theme.get() == "light" else "charles",
            gridstyle=":",
            gridaxis="both",
            y_on_right=False,
            facecolor="white" if self.theme.get() == "light" else "#121212",
            edgecolor="black" if self.theme.get() == "light" else "#333333",
            figcolor="white" if self.theme.get() == "light" else "#121212",
            gridcolor="lightgrey" if self.theme.get() == "light" else "#333333",
            rc={'axes.labelcolor': 'black' if self.theme.get() == "light" else "white"}
        )
        
        # Create subplots
        if 'rsi' in df.columns:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})
            fig.subplots_adjust(hspace=0.1)
            
            # Plot candlesticks
            mpf.plot(
                df, type='candle', style=mpf_style,
                title=f"{symbol} - {timeframe}",
                ylabel='Price',
                ax=ax1,
                volume=False  # Disable default volume
            )
            
            # Plot RSI
            ax2.plot(df.index, df['rsi'], color='blue', linewidth=0.8)
            ax2.axhline(y=30, color='green', linestyle='-', alpha=0.3)
            ax2.axhline(y=70, color='red', linestyle='-', alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color='green', alpha=0.3)
            ax2.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color='red', alpha=0.3)
            ax2.set_ylabel('RSI')
            ax2.grid(True, alpha=0.3)
            
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot candlesticks
            mpf.plot(
                df, type='candle', style=mpf_style,
                title=f"{symbol} - {timeframe}",
                ylabel='Price',
                ax=ax1,
                volume=True  # Enable volume in this case
            )
        
        # Create Tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def clear_logs(self):
        """Clear log display"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
    
    def save_logs(self):
        """Save logs to file"""
        try:
            filename = f"trading_bot_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as file:
                file.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Logs saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {e}")
    
    def setup_log_handler(self):
        """Set up handler to redirect logs to GUI"""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
            
            def emit(self, record):
                msg = self.format(record)
                
                def append():
                    self.text_widget.config(state="normal")
                    self.text_widget.insert(tk.END, msg + "\n")
                    self.text_widget.see(tk.END)
                    self.text_widget.config(state="disabled")
                
                # Schedule to main thread since logging might occur in a different thread
                self.text_widget.after(0, append)
        
        # Add the text handler to the logger
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)
    
    def edit_settings(self):
        """Open settings tab"""
        self.notebook.select(self.settings_frame)
    
    def save_settings(self):
        """Save settings to config file"""
        try:
            # Account settings
            self.bot.config["account"]["path"] = self.mt5_path_entry.get()
            self.bot.config["account"]["login"] = int(self.login_entry.get())
            self.bot.config["account"]["password"] = self.password_entry.get()
            self.bot.config["account"]["server"] = self.server_entry.get()
            
            # Trading settings
            self.bot.config["trading"]["symbols"] = [s.strip() for s in self.symbols_entry.get().split(',')]
            self.bot.config["trading"]["timeframes"] = [tf.strip() for tf in self.timeframes_entry.get().split(',')]
            self.bot.config["trading"]["risk_percent"] = float(self.risk_entry.get())
            self.bot.config["trading"]["max_risk_percent"] = float(self.max_risk_entry.get())
            self.bot.config["trading"]["min_risk_reward"] = float(self.min_rr_entry.get())
            self.bot.config["trading"]["max_open_trades"] = int(self.max_trades_entry.get())
            self.bot.config["trading"]["use_trailing_stop"] = self.use_trailing.get()
            self.bot.config["trading"]["trailing_stop_activation"] = float(self.trailing_activation_entry.get())
            self.bot.config["trading"]["trailing_stop_distance"] = float(self.trailing_distance_entry.get())
            
            # Strategy settings
            self.bot.config["strategy"]["price_action"]["enabled"] = self.use_price_action.get()
            self.bot.config["strategy"]["price_action"]["min_candle_size"] = float(self.min_candle_entry.get())
            self.bot.config["strategy"]["price_action"]["rejection_level"] = float(self.rejection_entry.get())
            self.bot.config["strategy"]["price_action"]["confirmation_candles"] = int(self.confirm_candles_entry.get())
            
            self.bot.config["strategy"]["volume"]["enabled"] = self.use_volume.get()
            self.bot.config["strategy"]["volume"]["threshold"] = float(self.volume_threshold_entry.get())
            
            self.bot.config["strategy"]["indicators"]["rsi"]["enabled"] = self.use_rsi.get()
            self.bot.config["strategy"]["indicators"]["rsi"]["period"] = int(self.rsi_period_entry.get())
            self.bot.config["strategy"]["indicators"]["rsi"]["overbought"] = int(self.rsi_ob_entry.get())
            self.bot.config["strategy"]["indicators"]["rsi"]["oversold"] = int(self.rsi_os_entry.get())
            
            self.bot.config["strategy"]["indicators"]["macd"]["enabled"] = self.use_macd.get()
            self.bot.config["strategy"]["indicators"]["macd"]["fast"] = int(self.macd_fast_entry.get())
            self.bot.config["strategy"]["indicators"]["macd"]["slow"] = int(self.macd_slow_entry.get())
            self.bot.config["strategy"]["indicators"]["macd"]["signal"] = int(self.macd_signal_entry.get())
            
            self.bot.config["strategy"]["indicators"]["support_resistance"]["enabled"] = self.use_sr.get()
            self.bot.config["strategy"]["indicators"]["support_resistance"]["lookback"] = int(self.sr_lookback_entry.get())
            self.bot.config["strategy"]["indicators"]["support_resistance"]["threshold"] = int(self.sr_threshold_entry.get())
            
            # Save to file
            success = self.bot.save_config()
            if success:
                messagebox.showinfo("Success", "Settings saved successfully")
            else:
                messagebox.showerror("Error", "Failed to save settings")
                
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your inputs: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def load_settings(self):
        """Load settings from config file"""
        self.bot.load_config()
        
        # Refresh the settings fields
        self.edit_settings()
    
    def reset_settings(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to default values?"):
            self.bot.config = self.bot.get_default_config()
            self.bot.save_config()
            self.load_settings()
            messagebox.showinfo("Reset Complete", "Settings have been reset to default values")
    
    def show_about(self):
        """Show about dialog"""
        about_text = "MetaTrader 5 Trading Bot\nVersion 1.0\n\n"
        about_text += "A customizable automated trading bot for MetaTrader 5.\n"
        about_text += "© 2025 - All rights reserved\n\n"
        about_text += "This bot uses price action, technical indicators, and volume analysis\n"
        about_text += "to find trading opportunities in Forex and other financial markets."
        
        messagebox.showinfo("About", about_text)
    
    def open_documentation(self):
        """Open documentation"""
        messagebox.showinfo("Documentation", "Documentation is not yet available.")
    
    def load_theme(self):
        """Load and apply the theme"""
        try:
            with open(self.bot.config_path, 'r') as file:
                config = json.load(file)
                if "gui" in config and "theme" in config["gui"]:
                    self.theme.set(config["gui"]["theme"])
        except:
            # Default to dark theme if config can't be loaded
            self.theme.set("dark")
            
        self.apply_theme()
    
    def apply_theme(self):
        """Apply the selected theme"""
        theme = self.theme.get()
        
        if theme == "dark":
            self.root.configure(bg="#121212")
            style = ttk.Style()
            style.theme_use("clam")
            style.configure(".", background="#121212", foreground="white", fieldbackground="#1E1E1E")
            style.configure("TLabelframe", background="#121212")
            style.configure("TLabelframe.Label", background="#121212", foreground="white")
            style.configure("TNotebook", background="#121212", tabmargins=[2, 5, 2, 0])
            style.configure("TNotebook.Tab", background="#2D2D2D", foreground="white", padding=[10, 2])
            style.map("TNotebook.Tab", background=[("selected", "#1E90FF")], foreground=[("selected", "white")])
            style.configure("Treeview", background="#1E1E1E", foreground="white", fieldbackground="#1E1E1E")
            style.map("Treeview", background=[("selected", "#1E90FF")])
            
            # Save theme preference
            if "gui" not in self.bot.config:
                self.bot.config["gui"] = {}
            self.bot.config["gui"]["theme"] = "dark"
            self.bot.save_config()
            
        else:  # Light theme
            self.root.configure(bg="white")
            style = ttk.Style()
            style.theme_use("clam")
            style.configure(".", background="#F0F0F0", foreground="black", fieldbackground="white")
            style.configure("TLabelframe", background="#F0F0F0")
            style.configure("TLabelframe.Label", background="#F0F0F0", foreground="black")
            style.configure("TNotebook", background="#F0F0F0", tabmargins=[2, 5, 2, 0])
            style.configure("TNotebook.Tab", background="#E1E1E1", foreground="black", padding=[10, 2])
            style.map("TNotebook.Tab", background=[("selected", "#4080BF")], foreground=[("selected", "white")])
            style.configure("Treeview", background="white", foreground="black", fieldbackground="white")
            style.map("Treeview", background=[("selected", "#4080BF")])
            
            # Save theme preference
            if "gui" not in self.bot.config:
                self.bot.config["gui"] = {}
            self.bot.config["gui"]["theme"] = "light"
            self.bot.save_config()
    
    def update_gui_data(self):
        """Update GUI data in a separate thread"""
        while self.gui_update_running:
            if self.account_connected.get():
                # Update account info
                self.refresh_account_info()
                
                # Update positions
                self.refresh_positions()
                
                # Update signals
                self.update_signals()
                
                # Update last update time
                self.update_time_label.config(text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
            
            time.sleep(2)  # Update every 2 seconds
    
    def update_signals(self):
        """Update signals display"""
        if not self.account_connected.get():
            return
            
        # Clear existing signals
        for item in self.signals_tree.get_children():
            self.signals_tree.delete(item)
            
        # Get recent signals
        signals = self.bot.run_trading_cycle()
        if signals:
            for signal in signals:
                self.signals_tree.insert(
                    "", "end",
                    values=(
                        signal["timestamp"],
                        signal["symbol"],
                        signal["action"].upper(),
                        signal["timeframe"],
                        f"{signal['strength']}/10",
                        f"{signal['entry']:.5f}",
                        f"{signal['stop_loss']:.5f}",
                        f"{signal['take_profit']:.5f}"
                    )
                )
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.gui_update_running = False
            
            # Stop the bot if running
            if self.bot_running.get():
                self.bot.stop()
                
            # Disconnect from MT5
            if self.account_connected.get():
                self.bot.disconnect()
                
            self.root.destroy()


# Main execution
if __name__ == "__main__":
    # Create the root window and initialize the GUI
    root = tk.Tk()
    app = TradingBotGUI(root)
    
    # Start the main loop
    root.mainloop()