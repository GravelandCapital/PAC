# main.py

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
    analyze_results,
    calculate_zigzag
)
from entry import Entry
import logging
import matplotlib.pyplot as plt

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all levels of log messages
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_log.log"),  # Log messages to a file
            logging.StreamHandler()                 # Also log messages to the console
        ]
    )
    
    # Base path where your data files are stored
    base_path = r"C:\Users\grave\OneDrive\Coding\fxdata"
    results_path = r"C:\Users\grave\OneDrive\Coding\PAC\results"
    output_dir = r"C:\Users\grave\OneDrive\Coding\fxdata"
    # For MacOS/Linux, uncomment the line below
    # base_path = r"/Users/koengraveland/PAC/fxdata"
    
    # List of file_pairs: [(daily_file1, hourly_file1), (daily_file2, hourly_file2), ...]
    file_pairs = [
        ('EUR_USD_D.xlsx', 'EUR_USD_H1.xlsx'),
        ('GBP_USD_D.xlsx', 'GBP_USD_H1.xlsx'),
        ('USD_JPY_D.xlsx', 'USD_JPY_H1.xlsx'),
        ('AUD_USD_D.xlsx', 'AUD_USD_H1.xlsx'),
        ('USD_CAD_D.xlsx', 'USD_CAD_H1.xlsx'),
        ('USD_CHF_D.xlsx', 'USD_CHF_H1.xlsx'),
        ('EUR_GBP_D.xlsx', 'EUR_GBP_H1.xlsx'),
        ('EUR_JPY_D.xlsx', 'EUR_JPY_H1.xlsx'),
        ('GBP_JPY_D.xlsx', 'GBP_JPY_H1.xlsx'),
        ('CAD_JPY_D.xlsx', 'CAD_JPY_H1.xlsx'),
        ('AUD_JPY_D.xlsx', 'AUD_JPY_H1.xlsx'),
        ('EUR_NZD_D.xlsx', 'EUR_NZD_H1.xlsx'),
        ('XAG_USD_D.xlsx', 'XAG_USD_H1.xlsx'),
        ('XAU_USD_D.xlsx', 'XAU_USD_H1.xlsx'),
        # Add additional file pairs here
    ]
    
    # Combined trades across all instruments
    combined_closed_trades = []
    start_date = '2015-01-01'

    for daily_file, hourly_file in file_pairs:
        # Extract instrument name from daily_file (assuming naming convention like 'EUR_USD_D.xlsx')
        instrument = extract_instrument_from_filename(daily_file)
        print(f"\nProcessing Instrument: {instrument}")
        logging.info(f"Processing Instrument: {instrument}")

        # Setup file paths
        daily_path = os.path.join(base_path, daily_file)
        hourly_path = os.path.join(base_path, hourly_file)
        zigzag_file_path = r"C:\Users\grave\OneDrive\Coding\PAC\zigzag.xlsx"
        # For MacOS/Linux, uncomment the line below
        # zigzag_file_path = r"/Users/koengraveland/PAC/zigzag.xlsx"

        try:
            # Fetch data
            data_fetcher = DataFetcher(daily_path, hourly_path)
            data_fetcher.fetch_data()
            df_daily = data_fetcher.df_daily
            df_hourly = data_fetcher.df_hourly
            logging.info(f"Data fetched for {instrument}.")
            print(f"Data fetched for {instrument}.")
        except Exception as e:
            logging.error(f"Error fetching data for {instrument}: {e}")
            print(f"Error fetching data for {instrument}: {e}")
            continue  # Skip to the next file pair

        try:
            # Calculate signals and ATR
            signal_calculator = SignalCalculator(df_hourly, df_daily)
            df_daily = signal_calculator.calculate_signal()
            df_daily = signal_calculator.calculate_atr()
        except Exception as e:
            logging.error(f"Error calculating signals and ATR for {instrument}: {e}")
            print(f"Error calculating signals and ATR for {instrument}: {e}")
            continue  # Skip to the next file pair

        try:
            # Load precomputed zigzag data
            zigzag_df = calculate_zigzag(df_hourly, depth=4, output_dir=output_dir, instrument=instrument)
            zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])
        except Exception as e:
            logging.error(f"Error loading/calculating zigzag data for {instrument}: {e}")
            print(f"Error loading/calculating zigzag data for {instrument}: {e}")
            continue  # Skip to the next file pair

        try:
            # Process signals and calculate entries
            entries = process_signals(df_daily, df_hourly, zigzag_df, instrument, start_date)
        except Exception as e:
            logging.error(f"Error processing signals for {instrument}: {e}")
            print(f"Error processing signals for {instrument}: {e}")
            continue  # Skip to the next file pair

        try:
            # Calculate stop loss and take profit
            daily_zigzag = calculate_zigzag_daily(df_daily, depth=3, output_dir=output_dir, instrument=instrument)
            calculate_sl_tp(entries, df_daily, df_hourly, zigzag_df, daily_zigzag, instrument)
        except Exception as e:
            logging.error(f"Error calculating SL/TP for {instrument}: {e}")
            print(f"Error calculating SL/TP for {instrument}: {e}")
            continue  # Skip to the next file pair

        # Initialize lists to manage orders and trades
        open_orders = entries.copy()  # Orders that are pending execution
        open_trades = []              # Trades that are currently open
        closed_trades = []            # Trades that have been closed

        try:
            # Ensure hourly data is sorted by time
            df_hourly = df_hourly.sort_values('time').reset_index(drop=True)
        except Exception as e:
            logging.error(f"Error sorting hourly data for {instrument}: {e}")
            print(f"Error sorting hourly data for {instrument}: {e}")
            continue  # Skip to the next file pair

        try:
            # Iterate over each hour in the hourly data
            for current_time in df_hourly['time']:
                # Update open orders
                for entry in open_orders[:]:  # Iterate over a copy to allow removal
                    # Check if the order should be considered at this time
                    if current_time >= entry.order_time:
                        trade_manager = TradeManager(df_hourly, entry, zigzag_df, depth=4)
                        try:
                            if trade_manager.check_order_execution():
                                # Order filled, add to open trades
                                open_trades.append(entry)
                                open_orders.remove(entry)
                            elif entry.order_status == "CANCELLED":
                                # Order not filled within time frame, remove it
                                open_orders.remove(entry)
                        except Exception as e:
                            logging.error(f"Error managing open orders for {instrument}: {e}")
                            print(f"Error managing open orders for {instrument}: {e}")
                            continue  # Skip to the next order

                # Manage open trades
                for entry in open_trades[:]:
                    # Only manage trades at or after the filled time
                    if current_time >= entry.filled_time:
                        trade_manager = TradeManager(df_hourly, entry, zigzag_df, depth=4)
                        try:
                            exit_info = trade_manager.manage_trade()
                            if exit_info:
                                # Trade exited, record exit info
                                entry.exit_time = exit_info['exit_time']
                                entry.exit_price = exit_info['exit_price']
                                closed_trades.append(entry)
                                open_trades.remove(entry)
                        except Exception as e:
                            logging.error(f"Error managing open trades for {instrument}: {e}")
                            print(f"Error managing open trades for {instrument}: {e}")
                            continue  # Skip to the next trade
        except Exception as e:
            logging.error(f"Error during trade management loop for {instrument}: {e}")
            print(f"Error during trade management loop for {instrument}: {e}")
            continue  # Skip to the next file pair

        try:
            # After iterating through all times, process any remaining open trades
            for entry in open_trades:
                try:
                    # Handle trades that are still open at the end of the data
                    last_row = df_hourly.iloc[-1]
                    entry.exit_time = last_row['time']
                    entry.exit_price = last_row['c']  # Closing price of the last bar
                    closed_trades.append(entry)
                except Exception as e:
                    logging.error(f"Error forcefully closing trade for {instrument} at row {entry.row_index}: {e}")
                    print(f"Error forcefully closing trade for {instrument} at row {entry.row_index}: {e}")
                    continue  # Skip to the next trade
        except Exception as e:
            print(f"Error during force-closing trades for {instrument}: {e}")
            # Not skipping here as it's the end of trade processing

        try:
            # Analyze results
            analyze_results(closed_trades, name=instrument)
            logging.info(f"Completed analysis for {instrument}.")
            print(f"Completed analysis for {instrument}.")
        except Exception as e:
            logging.error(f"Error analyzing results for {instrument}: {e}")
            print(f"Error analyzing results for {instrument}: {e}")
            # Not skipping here as it's the end of processing for this pair

        # Append to combined trades
        combined_closed_trades.extend(closed_trades)

    analyze_results(combined_closed_trades, name="Combined", output_path = r"C:\Users\grave\OneDrive\Coding\PAC\results")

    plt.show()


if __name__ == "__main__":
    main()