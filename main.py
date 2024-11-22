import pandas as pd
import numpy as np
import os
from data_fetcher import DataFetcher
from signal_calculator import SignalCalculator
from trade_manager import TradeManager
from utils import (
    calculate_zigzag_daily,
    process_signals,
    calculate_sl_tp,
    extract_instrument_from_filename,
    analyze_results
)
from entry import Entry

def main():
    # Base path where your data files are stored
    #base_path = r"C:\Users\grave\OneDrive\Coding\PAC\fxdata"
    # For MacOS/Linux, uncomment the line below
    base_path = r"/Users/koengraveland/PAC/fxdata"
    file_pairs = [('EUR_USD_D.xlsx', 'EUR_USD_H1.xlsx')]

    for daily_file, hourly_file in file_pairs:
        daily_path = os.path.join(base_path, daily_file)
        hourly_path = os.path.join(base_path, hourly_file)

        instrument = extract_instrument_from_filename(daily_file)

        # Fetch data
        data_fetcher = DataFetcher(daily_path, hourly_path)
        data_fetcher.fetch_data()
        df_daily = data_fetcher.df_daily
        df_hourly = data_fetcher.df_hourly

        # Calculate signals and ATR
        signal_calculator = SignalCalculator(df_hourly, df_daily)
        df_daily = signal_calculator.calculate_signal()
        df_daily = signal_calculator.calculate_atr()

        # Load precomputed zigzag data
        #zigzag_file_path = r"C:\Users\grave\OneDrive\Coding\PAC\zigzag.xlsx"
        # For MacOS/Linux, uncomment the line below
        zigzag_file_path = r"/Users/koengraveland/PAC/zigzag.xlsx"
        zigzag_df = pd.read_excel(zigzag_file_path)
        zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])
        print(f"Loaded zigzag file: {zigzag_file_path}")

        # Process signals and calculate entries
        entries = process_signals(df_daily, df_hourly, zigzag_df, instrument)

        # Calculate stop loss and take profit
        daily_zigzag = calculate_zigzag_daily(df_daily, depth=3)
        calculate_sl_tp(entries, df_daily, df_hourly, zigzag_df, daily_zigzag, instrument)

        # Initialize lists to manage orders and trades
        open_orders = entries.copy()  # Orders that are pending execution
        open_trades = []              # Trades that are currently open
        closed_trades = []            # Trades that have been closed

        # Ensure hourly data is sorted by time
        df_hourly = df_hourly.sort_values('time').reset_index(drop=True)

        # Iterate over each hour in the hourly data
        for current_time in df_hourly['time']:
            # Update open orders
            for entry in open_orders[:]:  # Iterate over a copy to allow removal
                # Check if the order should be considered at this time
                if current_time >= entry.order_time:
                    trade_manager = TradeManager(df_hourly, entry, zigzag_df)
                    if trade_manager.check_order_execution():
                        # Order filled, add to open trades
                        open_trades.append(entry)
                        open_orders.remove(entry)
                    elif entry.order_status == "CANCELLED":
                        # Order not filled within time frame, remove it
                        open_orders.remove(entry)

            # Manage open trades
            for entry in open_trades[:]:
                # Only manage trades at or after the filled time
                if current_time >= entry.filled_time:
                    trade_manager = TradeManager(df_hourly, entry, zigzag_df)
                    exit_info = trade_manager.manage_trade()
                    if exit_info:
                        # Trade exited, record exit info
                        entry.exit_time = exit_info['exit_time']
                        entry.exit_price = exit_info['exit_price']
                        closed_trades.append(entry)
                        open_trades.remove(entry)

        # After iterating through all times, process any remaining open trades
        for entry in open_trades:
            # Handle trades that are still open at the end of the data
            last_row = df_hourly.iloc[-1]
            entry.exit_time = last_row['time']
            entry.exit_price = last_row['c']  # Closing price of the last bar
            closed_trades.append(entry)

        # Analyze results
        analyze_results(closed_trades)

if __name__ == "__main__":
    main()
