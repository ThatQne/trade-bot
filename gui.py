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
from datetime import datetime, timedelta  # Add timedelta import here
import os
import sys
import logging
import matplotlib.dates as mdates

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
        self.risk_percent = tk.DoubleVar(value=self.bot.config["trading"]["risk_percent"])
        
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

        self.available_symbols = []
    
        # Create GUI variables
        self.account_connected = tk.BooleanVar(value=False)
        self.bot_running = tk.BooleanVar(value=False)
        
        # Default symbols until we get the actual list from the broker
        available_symbols = self.bot.config["trading"]["symbols"]
        self.selected_symbol = tk.StringVar(value=available_symbols[0] if available_symbols else "EURUSD")
    
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
        
        # Configure column headings with sort functionality
        self.sort_column = "Strength"  # Default sort column
        self.sort_reverse = True       # Default sort direction (descending)
        
        for col in columns:
            self.signals_tree.heading(col, text=col, 
                                    command=lambda c=col: self.sort_signal_column(c))
            self.signals_tree.column(col, width=80)
        
        # Show sort indicator on the default column
        self.signals_tree.heading(self.sort_column, text=f"{self.sort_column} ↓")
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
        risk_slider.bind("<ButtonRelease-1>", lambda event: self.update_risk("end"))
        self.risk_value_label = ttk.Label(controls_frame, text=f"{self.risk_percent.get()}%")
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

        # Add symbol refresh button
        refresh_button = ttk.Button(controls_frame, text="↻", width=3, command=self.refresh_symbols)
        refresh_button.grid(row=0, column=7, padx=5, pady=5)
        
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
        """Create the settings tab with all configuration options"""
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
        
        ttk.Label(trading_frame, text="Timeframes (comma-separated):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.timeframes_entry = ttk.Entry(trading_frame)
        self.timeframes_entry.insert(0, ",".join(self.bot.config["trading"]["timeframes"]))
        self.timeframes_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Risk %:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.risk_entry = ttk.Entry(trading_frame)
        self.risk_entry.insert(0, str(self.bot.config["trading"]["risk_percent"]))
        self.risk_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Max Risk %:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.max_risk_entry = ttk.Entry(trading_frame)
        self.max_risk_entry.insert(0, str(self.bot.config["trading"]["max_risk_percent"]))
        self.max_risk_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Min Risk:Reward:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.min_rr_entry = ttk.Entry(trading_frame)
        self.min_rr_entry.insert(0, str(self.bot.config["trading"]["min_risk_reward"]))
        self.min_rr_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Max Open Trades:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.max_trades_entry = ttk.Entry(trading_frame)
        self.max_trades_entry.insert(0, str(self.bot.config["trading"]["max_open_trades"]))
        self.max_trades_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # Trailing stop settings
        self.use_trailing = tk.BooleanVar(value=self.bot.config["trading"]["use_trailing_stop"])
        ttk.Checkbutton(
            trading_frame, text="Use Trailing Stop", variable=self.use_trailing
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Trailing Activation (pips):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.trailing_activation_entry = ttk.Entry(trading_frame)
        self.trailing_activation_entry.insert(0, str(self.bot.config["trading"]["trailing_stop_activation"]))
        self.trailing_activation_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(trading_frame, text="Trailing Distance (pips):").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.trailing_distance_entry = ttk.Entry(trading_frame)
        self.trailing_distance_entry.insert(0, str(self.bot.config["trading"]["trailing_stop_distance"]))
        self.trailing_distance_entry.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        # Analysis settings (new tab)
        analysis_frame = ttk.Frame(settings_notebook)
        settings_notebook.add(analysis_frame, text="Analysis")
        
        ttk.Label(analysis_frame, text="Minimum Signal Strength:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.min_signal_strength_entry = ttk.Entry(analysis_frame)
        self.min_signal_strength_entry.insert(0, str(self.bot.config.get("analysis", {}).get("min_signal_strength", 4.0)))
        self.min_signal_strength_entry.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Maximum Warnings:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.max_warnings_entry = ttk.Entry(analysis_frame)
        self.max_warnings_entry.insert(0, str(self.bot.config.get("analysis", {}).get("max_warnings", 3)))
        self.max_warnings_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Signal Expiry (minutes):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.signal_expiry_entry = ttk.Entry(analysis_frame)
        self.signal_expiry_entry.insert(0, str(self.bot.config.get("analysis", {}).get("signal_expiry_minutes", 5)))
        self.signal_expiry_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Symbols to Analyze:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.symbols_to_analyze_entry = ttk.Entry(analysis_frame)
        self.symbols_to_analyze_entry.insert(0, str(self.bot.config.get("analysis", {}).get("symbols_to_analyze", 30)))
        self.symbols_to_analyze_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Symbol Refresh Interval (s):").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.symbol_refresh_entry = ttk.Entry(analysis_frame)
        self.symbol_refresh_entry.insert(0, str(self.bot.config.get("analysis", {}).get("symbol_refresh_interval", 600)))
        self.symbol_refresh_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Analysis Interval (s):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.analysis_interval_entry = ttk.Entry(analysis_frame)
        self.analysis_interval_entry.insert(0, str(self.bot.config.get("analysis", {}).get("analysis_interval", 60)))
        self.analysis_interval_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(analysis_frame, text="Repeat Analysis Interval (s):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.repeat_analysis_entry = ttk.Entry(analysis_frame)
        self.repeat_analysis_entry.insert(0, str(self.bot.config.get("analysis", {}).get("repeat_analysis_interval", 600)))
        self.repeat_analysis_entry.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        
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
        

        self.live_update = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            button_frame, text="Apply changes immediately", 
            variable=self.live_update
        ).pack(side="left", padx=5)
        
        # Bind change events to all settings fields
        for entry_widget in [self.risk_entry, self.max_risk_entry, self.min_rr_entry, 
                        self.max_trades_entry, self.trailing_activation_entry,
                        self.trailing_distance_entry]:
            entry_widget.bind("<FocusOut>", self.live_update_settings)
        
        # Add validation and binding to checkbuttons
        self.use_trailing.trace_add("write", self.live_update_settings)
        self.use_price_action.trace_add("write", self.live_update_settings)
        self.use_volume.trace_add("write", self.live_update_settings)
        self.use_rsi.trace_add("write", self.live_update_settings)
        self.use_macd.trace_add("write", self.live_update_settings)
        self.use_sr.trace_add("write", self.live_update_settings)
    
        # Add notebook to frame
        settings_notebook.pack(fill="both", expand=True, padx=5, pady=5)

    def live_update_settings(self, *args):
        """Update settings in real-time if enabled"""
        if not self.live_update.get():
            return
        
        try:
            # Update trading settings in memory
            if "trading" in self.bot.config:
                self.bot.config["trading"]["risk_percent"] = float(self.risk_entry.get())
                self.bot.config["trading"]["max_risk_percent"] = float(self.max_risk_entry.get())
                self.bot.config["trading"]["min_risk_reward"] = float(self.min_rr_entry.get())
                self.bot.config["trading"]["max_open_trades"] = int(self.max_trades_entry.get())
                self.bot.config["trading"]["use_trailing_stop"] = self.use_trailing.get()
                self.bot.config["trading"]["trailing_stop_activation"] = float(self.trailing_activation_entry.get())
                self.bot.config["trading"]["trailing_stop_distance"] = float(self.trailing_distance_entry.get())
            
            # Update strategy settings in memory
            if "strategy" in self.bot.config:
                self.bot.config["strategy"]["price_action"]["enabled"] = self.use_price_action.get()
                self.bot.config["strategy"]["volume"]["enabled"] = self.use_volume.get()
                self.bot.config["strategy"]["indicators"]["rsi"]["enabled"] = self.use_rsi.get()
                self.bot.config["strategy"]["indicators"]["macd"]["enabled"] = self.use_macd.get()
                self.bot.config["strategy"]["indicators"]["support_resistance"]["enabled"] = self.use_sr.get()
            
            # Don't save to file yet - that happens when Save is clicked
            logger.debug("Settings updated in memory (live update)")
        except Exception as e:
            logger.error(f"Error in live settings update: {e}")
    
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

    def refresh_symbols(self):
        """Refresh available symbols from the broker"""
        if not self.account_connected.get():
            return
            
        try:
            symbols = self.bot.get_available_symbols()
            if symbols:
                self.available_symbols = symbols
                
                # Update the dropdowns
                self.symbol_combo['values'] = symbols
                self.chart_symbol_combo['values'] = symbols
                
                # Update config with new symbols if there are enough
                if len(symbols) >= 5:  # Make sure we have enough symbols
                    self.bot.config["trading"]["symbols"] = symbols[:10]  # Take first 10 as default
                    self.bot.save_config()
                
                logger.info(f"Updated symbol lists with {len(symbols)} symbols from broker")
        except Exception as e:
            logger.error(f"Error refreshing symbols: {e}")
            messagebox.showerror("Symbol Error", f"Failed to get symbols: {str(e)}")
    
    def connect_to_mt5(self):
        """Connect to MetaTrader 5 platform and load available symbols"""
        if not self.account_connected.get():
            try:
                result = self.bot.connect()
                if result:
                    self.account_connected.set(True)
                    self.connection_label.config(text="Connected", foreground="green")
                    
                    # Update the symbol mapping if method exists
                    if hasattr(self.bot, 'update_symbol_mapping'):
                        self.bot.update_symbol_mapping()
                    
                    # Get available symbols from the broker if method exists
                    if hasattr(self.bot, 'get_available_symbols'):
                        self.refresh_symbols()
                    
                    messagebox.showinfo("Connection", "Successfully connected to MetaTrader 5")
                    self.refresh_account_info()
                    self.refresh_positions()
                    self.connect_button.config(text="Disconnect from MT5")
                else:
                    messagebox.showerror("Connection Error", "Failed to connect to MetaTrader 5")
            except Exception as e:
                messagebox.showerror("Connection Error", f"Error connecting to MT5: {str(e)}")
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
            # Use a thread to start the bot without freezing the UI
            def start_bot_thread():
                try:
                    result = self.bot.start()
                    if result:
                        # Update UI from main thread
                        self.root.after(0, lambda: self._update_bot_started())
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to start trading bot"))
                except Exception as e:
                    error_msg = f"Error starting bot: {str(e)}"
                    logger.error(error_msg)
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            
            # Show a "Starting..." message
            self.bot_status_label.config(text="Bot: Starting...", foreground="orange")
            start_thread = threading.Thread(target=start_bot_thread, daemon=True)
            start_thread.start()
        
    def _update_bot_started(self):
        """Update UI after bot successfully starts"""
        self.bot_running.set(True)
        self.bot_status_label.config(text="Bot: Running", foreground="green")
        self.bot_button.config(text="Stop Bot")
        messagebox.showinfo("Bot Started", "Trading bot has been started")

    def stop_bot(self):
        """Stop the trading bot"""
        if self.bot_running.get():
            # Use a thread to stop the bot without freezing the UI
            def stop_bot_thread():
                try:
                    result = self.bot.stop()
                    if result:
                        # Update UI from main thread
                        self.root.after(0, lambda: self._update_bot_stopped())
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", "Failed to stop trading bot"))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Error stopping bot: {str(e)}"))
            
            # Show a "Stopping..." message
            self.bot_status_label.config(text="Bot: Stopping...", foreground="orange")
            threading.Thread(target=stop_bot_thread, daemon=True).start()

    def _update_bot_stopped(self):
        """Update UI after bot successfully stops"""
        self.bot_running.set(False)
        self.bot_status_label.config(text="Bot: Stopped", foreground="red")
        self.bot_button.config(text="Start Bot")
        messagebox.showinfo("Bot Stopped", "Trading bot has been stopped")
    
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
        """Update risk percentage setting and label without excessive saves"""
        try:
            # Round to one decimal place
            value = round(float(self.risk_percent.get()), 1)
            self.risk_value_label.config(text=f"{value}%")
            
            # Only update the config value in memory, don't save to file during dragging
            if "trading" in self.bot.config:
                self.bot.config["trading"]["risk_percent"] = value
                
                # We'll save only when the slider is released (binding the slider release event)
                if hasattr(args, '__len__') and len(args) > 0 and args[0] == "end":
                    # Only save when slider is released
                    self.bot.save_config()
                    logger.info(f"Risk percentage updated to {value}%")
        except Exception as e:
            # Add error handling to prevent crashes
            logger.error(f"Error updating risk: {e}")
    
    def refresh_account_info(self):
        """Refresh account information"""
        if not self.account_connected.get():
            return
            
        try:
            if hasattr(self.bot, 'get_account_summary'):
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
        except Exception as e:
            logger.error(f"Error refreshing account info: {e}")
    
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
        """Analyze selected symbol with improved error handling and async processing"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
                
        symbol = self.symbol_combo.get()
        timeframe = self.timeframe_combo.get()
        
        # Disable analyze button during analysis
        analyze_button = None
        for child in self.symbol_combo.master.winfo_children():
            if isinstance(child, ttk.Button) and "Analyze" in child["text"]:
                analyze_button = child
                child.config(state="disabled", text="Analyzing...")
                break
        
        # Clear previous analysis
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, f"Analyzing {symbol} on {timeframe} timeframe...\n\n")
        self.root.update_idletasks()  # Force UI update
        
        # Create a thread for analysis to prevent UI freezing
        def run_analysis():
            try:
                # Get market data
                df = self.bot.get_market_data(symbol, timeframe)
                if df is None or len(df) == 0:
                    self.root.after(0, lambda: self._update_analysis_text(
                        "No data available for this symbol/timeframe or symbol not found.\n"
                        "Check if the symbol is available from your broker."))
                    return
                
                # Calculate indicators
                df = self.bot.calculate_indicators(df)
                if df is None or len(df) == 0:
                    self.root.after(0, lambda: self._update_analysis_text(
                        "Error calculating indicators. Insufficient data points."))
                    return
                
                # Analyze price action
                try:
                    pa_signals, levels = self.bot.analyze_price_action(df)
                    if pa_signals is None:
                        self.root.after(0, lambda: self._update_analysis_text(
                            "No analyzable patterns found."))
                        return
                except Exception as e:
                    logger.error(f"Error in price action analysis: {e}")
                    self.root.after(0, lambda: self._update_analysis_text(
                        f"Error analyzing price action: {str(e)}"))
                    return
                
                # Update the UI from the main thread
                self.root.after(0, lambda: self._display_analysis_results(
                    symbol, timeframe, pa_signals, levels, df))
                
            except Exception as e:
                logger.error(f"Error analyzing symbol {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.root.after(0, lambda: self._update_analysis_text(
                    f"Error during analysis: {str(e)}\n\n"
                    f"This might be due to data issues or the symbol not being supported by your broker."))
            finally:
                # Re-enable analyze button
                if analyze_button:
                    self.root.after(0, lambda: analyze_button.config(state="normal", text="Analyze"))
        
        # Start analysis in a separate thread
        threading.Thread(target=run_analysis, daemon=True).start()

    def _update_analysis_text(self, message):
        """Update the analysis text with error or status messages"""
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(tk.END, message)

    def _display_analysis_results(self, symbol, timeframe, pa_signals, levels, df):
        """Display analysis results in the text widget"""
        # Clear previous text
        self.analysis_text.delete(1.0, tk.END)
        
        try:
            # Get the last row data
            last_row = df.iloc[-1].copy() if len(df) > 0 else None
            
            # Display results
            if pa_signals["buy"] or pa_signals["sell"]:
                action = "BUY" if pa_signals["buy"] else "SELL"
                self.analysis_text.insert(tk.END, f"Signal: {action}\n")
                self.analysis_text.insert(tk.END, f"Strength: {pa_signals['strength']}/10\n")
                
                patterns = pa_signals.get("patterns", [])
                if patterns:
                    self.analysis_text.insert(tk.END, f"Patterns: {', '.join(patterns)}\n")
                    
                confirmations = pa_signals.get("confirmations", [])
                if confirmations:
                    self.analysis_text.insert(tk.END, f"Confirmations: {', '.join(confirmations)}\n")
                
                warnings = pa_signals.get("warnings", [])
                if warnings:
                    self.analysis_text.insert(tk.END, f"Warnings: {', '.join(warnings)}\n")
                
                self.analysis_text.insert(tk.END, "\nTrade parameters:\n")
                
                # Handle potential None values by checking explicitly
                if levels.get('entry') is not None:
                    self.analysis_text.insert(tk.END, f"Entry price: {levels['entry']:.5f}\n")
                if levels.get('stop_loss') is not None:
                    self.analysis_text.insert(tk.END, f"Stop loss: {levels['stop_loss']:.5f}\n")
                if levels.get('take_profit') is not None:
                    self.analysis_text.insert(tk.END, f"Take profit: {levels['take_profit']:.5f}\n")
                    
                # Calculate risk metrics if we have the needed values
                if levels.get("entry") is not None and levels.get("stop_loss") is not None:
                    try:
                        # Calculate lot size
                        lot_size = self.bot.calculate_position_size(symbol, levels["entry"], levels["stop_loss"])
                        
                        risk_pips = abs(levels["entry"] - levels["stop_loss"]) * 10000
                        
                        # Make sure take_profit is available before calculating reward
                        if levels.get("take_profit") is not None:
                            reward_pips = abs(levels["entry"] - levels["take_profit"]) * 10000
                            rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                        else:
                            reward_pips = 0
                            rr_ratio = 0
                        
                        self.analysis_text.insert(tk.END, f"Position size: {lot_size:.2f} lots\n")
                        self.analysis_text.insert(tk.END, f"Risk: {risk_pips:.1f} pips\n")
                        
                        if reward_pips > 0:
                            self.analysis_text.insert(tk.END, f"Reward: {reward_pips:.1f} pips\n")
                            self.analysis_text.insert(tk.END, f"Risk/Reward ratio: 1:{rr_ratio:.2f}\n")
                        
                        # Pre-fill the manual trading fields
                        self.action_combo.set("Buy" if pa_signals['buy'] else "Sell")
                        self.lot_entry.delete(0, tk.END)
                        self.lot_entry.insert(0, f"{lot_size:.2f}")
                        
                        if levels.get('stop_loss') is not None:
                            self.sl_entry.delete(0, tk.END)
                            self.sl_entry.insert(0, f"{levels['stop_loss']:.5f}")
                        
                        if levels.get('take_profit') is not None:
                            self.tp_entry.delete(0, tk.END)
                            self.tp_entry.insert(0, f"{levels['take_profit']:.5f}")
                    except Exception as e:
                        logger.error(f"Error calculating position metrics: {e}")
                        self.analysis_text.insert(tk.END, f"\nError calculating position metrics: {str(e)}\n")
            else:
                self.analysis_text.insert(tk.END, "No tradable signals found at this time.\n")
                self.analysis_text.insert(tk.END, "\nCurrent market conditions:\n")
                
                # Show price relative to key MAs if available
                if last_row is not None:
                    try:
                        if 'ma50' in last_row and 'ma200' in last_row:
                            # Use explicit float conversion to avoid Series truth value issues
                            price_ma50 = "above" if float(last_row['close']) > float(last_row['ma50']) else "below"
                            price_ma200 = "above" if float(last_row['close']) > float(last_row['ma200']) else "below"
                            ma_trend = "bullish" if float(last_row['ma50']) > float(last_row['ma200']) else "bearish"
                            
                            self.analysis_text.insert(tk.END, f"Price is {price_ma50} 50 MA and {price_ma200} 200 MA\n")
                            self.analysis_text.insert(tk.END, f"MA trend is {ma_trend} (50 vs 200)\n")
                        
                        if 'rsi' in last_row:
                            rsi_val = float(last_row['rsi'])
                            rsi_state = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral"
                            self.analysis_text.insert(tk.END, f"RSI: {rsi_val:.1f} ({rsi_state})\n")
                    except Exception as e:
                        logger.error(f"Error showing market conditions: {e}")
        except Exception as e:
            logger.error(f"Error displaying analysis results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.analysis_text.insert(tk.END, f"Error displaying analysis: {str(e)}")
    
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
        """Load chart with improved performance and robustness"""
        if not self.account_connected.get():
            messagebox.showwarning("Warning", "Not connected to MetaTrader 5")
            return
            
        symbol = self.chart_symbol_combo.get()
        timeframe = self.chart_tf_combo.get()
        bars = int(self.bars_combo.get())
        
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # Show loading message
        loading_label = ttk.Label(self.chart_frame, text=f"Loading chart data for {symbol}...")
        loading_label.pack(pady=50)
        self.chart_frame.update_idletasks()  # Force update to show loading message
        
        # Load data in a background thread to prevent UI freezing
        def load_chart_data():
            try:
                # Get market data
                df = self.bot.get_market_data(symbol, timeframe, bars)
                
                if df is None or len(df) == 0:
                    return None
                    
                # Calculate indicators
                df = self.bot.calculate_indicators(df)
                
                # Set datetime as index
                df.set_index("time", inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching chart data: {e}")
                return None
        
        # Create and start loading thread
        load_thread = threading.Thread(target=lambda: self.after_chart_data_loaded(load_chart_data(), symbol, timeframe))
        load_thread.daemon = True
        load_thread.start()

    def after_chart_data_loaded(self, df, symbol, timeframe):
        """Handle chart plotting after data is loaded"""
        # Remove loading message
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        if df is None:
            ttk.Label(self.chart_frame, text=f"No data available for {symbol} on {timeframe} timeframe").pack(pady=50)
            return
        
        try:
            # Close any existing figures
            plt.close('all')
            
            # Create a figure and axes
            fig = plt.figure(figsize=(12, 8), dpi=100)
            
            # Create price and indicator subplots
            if 'rsi' in df.columns:
                # Price chart (top)
                ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=3, fig=fig)
                # Volume (middle)
                ax2 = plt.subplot2grid((6, 1), (3, 0), rowspan=1, fig=fig, sharex=ax1)
                # RSI (bottom)
                ax3 = plt.subplot2grid((6, 1), (4, 0), rowspan=2, fig=fig, sharex=ax1)
                
                # Add title
                fig.suptitle(f"{symbol} - {timeframe}", fontsize=14)
                
                # Plot OHLC data
                mpf.plot(df, type='candle', style='yahoo', ax=ax1, volume=False)
                
                # Plot volume
                ax2.bar(df.index, df['tick_volume'], color='blue', alpha=0.5)
                ax2.set_ylabel('Volume')
                
                # Plot RSI
                ax3.plot(df.index, df['rsi'], color='purple')
                ax3.axhline(70, linestyle='--', color='r', alpha=0.5)
                ax3.axhline(30, linestyle='--', color='g', alpha=0.5)
                ax3.fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color='green', alpha=0.3)
                ax3.fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color='red', alpha=0.3)
                ax3.set_ylim(0, 100)
                ax3.set_ylabel('RSI')
                
                # Remove x-axis tick labels from upper plots
                ax1.tick_params(axis='x', labelsize=7)
                ax1.set_xticklabels([])
                ax2.tick_params(axis='x', labelsize=7)
                ax2.set_xticklabels([])
            else:
                # Simplified chart without RSI
                ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=4, fig=fig)
                ax2 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, fig=fig, sharex=ax1)
                
                fig.suptitle(f"{symbol} - {timeframe}", fontsize=14)
                mpf.plot(df, type='candle', style='yahoo', ax=ax1, volume=False)
                
                ax2.bar(df.index, df['tick_volume'], color='blue', alpha=0.5)
                ax2.set_ylabel('Volume')
                
                ax1.tick_params(axis='x', labelsize=7)
                ax1.set_xticklabels([])
            
            # Format the x-axis to display dates nicely
            ax3 = plt.gca()
            date_format = mdates.DateFormatter('%m-%d %H:%M')
            ax3.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Create canvas to display the chart
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Add toolbar for interactive features
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, self.chart_frame)
            toolbar.update()
            
        except Exception as e:
            logger.error(f"Error plotting chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Show error message
            ttk.Label(self.chart_frame, text=f"Error creating chart: {str(e)}").pack(pady=50)
    
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
            
            # Analysis settings
            if "analysis" not in self.bot.config:
                self.bot.config["analysis"] = {}
            
            self.bot.config["analysis"]["min_signal_strength"] = float(self.min_signal_strength_entry.get())
            self.bot.config["analysis"]["max_warnings"] = int(self.max_warnings_entry.get())
            self.bot.config["analysis"]["signal_expiry_minutes"] = int(self.signal_expiry_entry.get())
            self.bot.config["analysis"]["symbols_to_analyze"] = int(self.symbols_to_analyze_entry.get())
            self.bot.config["analysis"]["symbol_refresh_interval"] = int(self.symbol_refresh_entry.get())
            self.bot.config["analysis"]["analysis_interval"] = int(self.analysis_interval_entry.get())
            self.bot.config["analysis"]["repeat_analysis_interval"] = int(self.repeat_analysis_entry.get())
            
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

    def sort_signal_column(self, column):
        """Sort signals by clicking on column headers"""
        # If already sorting by this column, reverse direction
        if self.sort_column == column:
            self.sort_reverse = not self.sort_reverse
        else:
            # Save new sort column and reset to descending
            self.sort_column = column
            self.sort_reverse = True
        
        # Update column headings to show sort direction
        for col in self.signals_tree["columns"]:
            # Remove any existing sort indicator
            text = col.replace(" ↑", "").replace(" ↓", "")
            if col == self.sort_column:
                text += " ↓" if self.sort_reverse else " ↑"
            self.signals_tree.heading(col, text=text)
        
        # Perform the sort
        if hasattr(self, 'stored_signals'):
            self.update_signals(resort=True)

    def update_signals(self, resort=False):
        """Update signals display with sorting capability"""
        if not self.account_connected.get():
            return
                
        try:
            # Get recent signals if we're not just resorting
            if not resort:
                current_signals = self.bot.run_trading_cycle()
                
                # Get current time for expiration check
                current_time = datetime.now()
                
                # Get signal expiry time from config
                signal_expiry_minutes = self.bot.config.get("analysis", {}).get("signal_expiry_minutes", 5)
                
                # Store signals with timestamp if not already present
                if not hasattr(self, 'stored_signals'):
                    self.stored_signals = []
                
                # Process new signals
                if current_signals:
                    for signal in current_signals:
                        # Add expiry time to new signals
                        signal['expiry_time'] = current_time + timedelta(minutes=signal_expiry_minutes)
                        
                        # Check if signal already exists
                        existing_signal = next((s for s in self.stored_signals 
                                            if s['symbol'] == signal['symbol'] 
                                            and s['timeframe'] == signal['timeframe']
                                            and s['action'] == signal['action']), None)
                        
                        if existing_signal:
                            # Update existing signal with new data
                            existing_signal.update(signal)
                        else:
                            # Add new signal to stored signals
                            self.stored_signals.append(signal)
                            logger.info(f"New signal added: {signal['symbol']} {signal['timeframe']} {signal['action']}")
                
                # Get minimum strength threshold from config
                min_strength = self.bot.config.get("analysis", {}).get("min_signal_strength", 4.0)
                
                # Remove expired signals or signals with strength below threshold
                self.stored_signals = [s for s in self.stored_signals 
                                    if s['expiry_time'] > current_time and s['strength'] >= min_strength]
            
            # Clear existing signals in treeview
            for item in self.signals_tree.get_children():
                self.signals_tree.delete(item)
            
            # Add all stored signals to treeview with sorting
            if hasattr(self, 'stored_signals') and self.stored_signals:
                logger.debug(f"Displaying {len(self.stored_signals)} signals in dashboard")
                
                # Sort signals based on current sort column and direction
                if self.sort_column == "Time":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: x.get('timestamp', ''),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "Symbol":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: x.get('symbol', ''),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "Action":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: x.get('action', ''),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "Timeframe":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: x.get('timeframe', ''),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "Strength":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: float(x.get('strength', 0)),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "Entry":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: float(x.get('entry', 0) or 0),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "SL":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: float(x.get('stop_loss', 0) or 0),
                                        reverse=self.sort_reverse)
                elif self.sort_column == "TP":
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: float(x.get('take_profit', 0) or 0),
                                        reverse=self.sort_reverse)
                else:
                    # Default sort by strength
                    sorted_signals = sorted(self.stored_signals, 
                                        key=lambda x: float(x.get('strength', 0)),
                                        reverse=True)
                
                # Display the sorted signals
                for signal in sorted_signals:
                    try:
                        # Format values with proper defaults for missing fields
                        timestamp = signal.get("timestamp", "")
                        symbol = signal.get("symbol", "")
                        action = signal.get("action", "").upper()
                        timeframe = signal.get("timeframe", "")
                        
                        # Format the strength with trade type
                        strength_display = f"{signal.get('strength', 0):.1f}/10"
                        if 'trade_type' in signal:
                            strength_display += f" ({signal['trade_type']})"
                        
                        # Format prices with proper precision based on symbol
                        entry = f"{signal.get('entry', 0):.5f}" if signal.get('entry') else "-"
                        sl = f"{signal.get('stop_loss', 0):.5f}" if signal.get('stop_loss') else "-"
                        tp = f"{signal.get('take_profit', 0):.5f}" if signal.get('take_profit') else "-"
                        
                        self.signals_tree.insert(
                            "", "end",
                            values=(timestamp, symbol, action, timeframe, strength_display, entry, sl, tp)
                        )
                        
                    except Exception as e:
                        logger.error(f"Error adding signal to tree: {e}")
                
                # Force a refresh of the treeview
                self.signals_tree.yview_moveto(0)
                    
        except Exception as e:
            logger.error(f"Error updating signals display: {e}")
    
    def update_gui_data(self):
        """Update GUI data in a separate thread with improved performance"""
        while self.gui_update_running:
            try:
                if self.account_connected.get():
                    # Define tasks that will be executed
                    tasks = []
                    
                    # Only update what's necessary based on the visible tab
                    current_tab = self.notebook.index(self.notebook.select())
                    
                    # Always update account info (lightweight)
                    tasks.append(('account', self.refresh_account_info))
                    
                    # Update positions if positions tab is visible
                    if current_tab == 3:  # Positions tab index
                        tasks.append(('positions', self.refresh_positions))
                    
                    # Update signals if dashboard tab is visible
                    if current_tab == 0:  # Dashboard tab index
                        tasks.append(('signals', self.update_signals))
                    
                    # Execute tasks sequentially to avoid overwhelming the UI
                    for task_name, task_func in tasks:
                        try:
                            task_func()
                        except Exception as e:
                            logger.error(f"Error updating {task_name}: {e}")
                    
                    # Update timestamp
                    self.update_time_label.config(text=f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
                
                # Sleep between updates - adaptive timing based on system load
                import psutil
                cpu_load = psutil.cpu_percent()
                if cpu_load > 70:  # High system load
                    time.sleep(3)  # Update less frequently
                else:
                    time.sleep(1.5)  # Normal update frequency
                    
            except Exception as e:
                logger.error(f"Error in GUI update thread: {e}")
                time.sleep(2)  # Recover from error
    
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