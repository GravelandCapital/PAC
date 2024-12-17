# utils.py

import pandas as pd
from collections import defaultdict
from entry import Entry
from engulfing_handler import EngulfingHandler
from hammer_shooting_star_handler import HammerShootingStarHandler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_log.log"),
        logging.StreamHandler()
    ]
)

def get_group_key(entry):
    if entry.signal in ['bull_eng', 'bear_eng']:
        return (entry.signal, entry.row_index)
    elif entry.signal in ['hammer', 'shooting_star']:
        return (entry.signal, entry.row_index, entry.entry_type)
    else:
        return None  # For signals that don't require grouping

def calculate_zigzag_daily(df_daily, depth=3, output_dir = None, instrument = None):
    """Calculates ZigZag pivots for the provided depth in the daily data."""

    file_path = os.path.join(output_dir, f"{instrument}, {depth}, _Daily_ZigZag.xlsx")

    if os.path.exists(file_path):
        return pd.read_excel(file_path)

    pivots = []
    total_rows = len(df_daily)

    for row_index in range(depth, total_rows - depth):
        pivot_high = True
        pivot_low = True

        current_time = df_daily.iloc[row_index]['time']
        current_low = df_daily.iloc[row_index]['l']
        current_high = df_daily.iloc[row_index]['h']

        for i in range(row_index - depth, row_index + depth + 1):
            if i < 0 or i >= total_rows:
                continue

            if df_daily.iloc[row_index]['l'] > df_daily.iloc[i]['l']:
                pivot_low = False
            if df_daily.iloc[row_index]['h'] < df_daily.iloc[i]['h']:
                pivot_high = False

        if pivot_high:
            pivots.append([row_index, current_time, current_high, 'h'])

        if pivot_low:
            pivots.append([row_index, current_time, current_low, 'l'])

    daily_zigzag = pd.DataFrame(pivots, columns=['index', 'time', 'price', 'type'])
    daily_zigzag['time'] = pd.to_datetime(daily_zigzag['time'])

    daily_zigzag.to_excel(file_path, index=False)
    print(f"ZigZag pivots calculated and saved to {file_path}.")

    return daily_zigzag

def calculate_zigzag(df_hourly, depth=4, output_dir = None, instrument = None):
        
        file_path = os.path.join(output_dir, f"{instrument}, {depth}, _Hourly_ZigZag.xlsx")

        if os.path.exists(file_path):
            return pd.read_excel(file_path)
                                 
        pivots = []

        # Ensure row_index is within valid bounds
        for row_index in range(depth, len(df_hourly) - depth):
            pivot_high = True  # Assume it's a pivot high
            pivot_low = True   # Assume it's a pivot low
            
            # Get current candle details (from row_index)
            current_time = df_hourly.iloc[row_index]['time']
            current_low = df_hourly.iloc[row_index]['l']
            current_high = df_hourly.iloc[row_index]['h']

            # Loop over neighboring candles (from row_index - depth to row_index + depth)
            for i in range(row_index - depth, row_index + depth + 1):
                if df_hourly.iloc[row_index]['l'] > df_hourly.iloc[i]['l']:
                    pivot_low = False  # Disqualify as a pivot low
                if df_hourly.iloc[row_index]['h'] < df_hourly.iloc[i]['h']:
                    pivot_high = False  # Disqualify as a pivot high

            if pivot_high:
                pivots.append([row_index, current_time, current_high, 'h'])

            if pivot_low:
                pivots.append([row_index, current_time, current_low, 'l'])

        zigzag_df = pd.DataFrame(pivots, columns=['index', 'time', 'price', 'type'])
        zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])

        zigzag_df.to_excel(file_path, index=False)
        print(f"ZigZag pivots calculated and saved to {file_path}.")

        return zigzag_df

def process_signals(df_daily, df_hourly, zigzag_df, instrument, start_date = None):
    """
    Processes trading signals and generates Entry objects.
    """
    entries = []

    if start_date:
        df_daily = df_daily[df_daily['time'] >= start_date]
        df_hourly = df_hourly[df_hourly['time'] >= start_date]

    for row_index in df_daily.index:
        signal = df_daily.loc[row_index, 'signal']
        if signal in ['bull_eng', 'bear_eng']:
            handler = EngulfingHandler(df_daily, df_hourly, row_index, zigzag_df, instrument)
        elif signal in ['hammer', 'shooting_star']:
            handler = HammerShootingStarHandler(df_daily, df_hourly, row_index, zigzag_df, instrument)
        else:
            continue  # No signal, skip

        # Calculate entries
        signal_entries = handler.calculate_entries()
        entries.extend(signal_entries)
    return entries

def calculate_sl_tp(entries, df_daily, df_hourly, zigzag_df, daily_zigzag, instrument):
    """
    Calculates Stop Loss and Take Profit for each entry.
    Groups entries appropriately to prevent duplicates and selects the best trade per group.
    """

    processed_entries = []
    for entry in entries:
        if entry.signal in ['bull_eng', 'bear_eng']:
            handler = EngulfingHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        elif entry.signal in ['hammer', 'shooting_star']:
            handler = HammerShootingStarHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        else:
            continue

        # Calculate stop loss and take profit
        try:
            stop_loss_list = handler.calculate_stop_loss(entry)
            entry.take_profit = handler.calculate_take_profit(entry, daily_zigzag)
        except Exception as e:
            continue

        if entry.take_profit is None or not stop_loss_list:
            continue    

        valid_combos = generate_valid_combinations(stop_loss_list, entry)

        if not valid_combos:
            continue  # Skip to the next entry if no valid combinations

        best_trade = select_best_trade(valid_combos, entry)
        if best_trade:
            # Update the Entry object with the best trade details
            entry.price = best_trade['entry_price']
            entry.stop_loss = best_trade['stop_loss']
            entry.original_stop_loss = best_trade['stop_loss']
            entry.take_profit = best_trade['take_profit']
            processed_entries.append(entry)
  
    # Now, group the processed entries
    grouped_entries = defaultdict(list)
    for entry in processed_entries:
        key = get_group_key(entry)
        if key:
            grouped_entries[key].append(entry)

    final_entries = []
    for key, group in grouped_entries.items():
        is_bullish = group[0].signal in ['bull_eng', 'hammer']

        if is_bullish:
            # Select the entry with the highest entry_price
            best_entry = max(group, key=lambda x: x.price)
        else:
            # Select the entry with the lowest entry_price
            best_entry = min(group, key=lambda x: x.price)

        final_entries.append(best_entry)

    # Replace entries with final list that meets R/R condition
    entries[:] = final_entries

def generate_valid_combinations(stop_loss_list, entry):
    """
    Generates valid combinations of stop loss and take profit based on R:R ratio.
    """
    valid_combos = []
    for stop_loss in stop_loss_list:
        entry_price = entry.price
        take_profit = entry.take_profit

        if entry.signal in ['bull_eng', 'hammer']:
            rr_ratio = (take_profit - entry_price) / (entry_price - stop_loss)
        elif entry.signal in ['bear_eng', 'shooting_star']:
            rr_ratio = (entry_price - take_profit) / (stop_loss - entry_price)
        else:
            rr_ratio = 0  # Undefined signal type


        if rr_ratio >= 1.5:
            valid_combos.append({'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit})


    return valid_combos

def select_best_trade(valid_combos, entry):
    """
    Selects the best trade from valid combinations based on signal type.
    """
    is_bullish = entry.signal in ['bull_eng', 'hammer']

    if is_bullish:
        # Select the combo with the highest entry_price
        max_entry_price = max(comb['entry_price'] for comb in valid_combos)
        max_entry_combinations = [comb for comb in valid_combos if comb['entry_price'] == max_entry_price]
        best_trade = max(max_entry_combinations, key=lambda x: x['stop_loss'])
    else:
        # Select the combo with the lowest entry_price
        min_entry_price = min(comb['entry_price'] for comb in valid_combos)
        min_entry_combinations = [comb for comb in valid_combos if comb['entry_price'] == min_entry_price]
        best_trade = min(min_entry_combinations, key=lambda x: x['stop_loss'])

    return best_trade

def extract_instrument_from_filename(filename):
    """Extracts the instrument name from the filename."""
    return filename.split('_')[0] + '_' + filename.split('_')[1]

def analyze_results(trade_results, name="Combined", results_path=r"C:\Users\grave\OneDrive\Coding\Coding\results"):
    """
    Analyzes trade results by creating a DataFrame, calculating basic statistics,
    and plotting a simple equity curve.

    Parameters:
    - trade_results: List of Entry objects representing closed trades.
    - name: Identifier for the analysis (e.g., instrument name or "Combined").
    - results_path: Directory where output files (CSV and plots) will be stored.
    """
    if not trade_results:
        print(f"No closed trades to analyze for {name}.")
        return

    # 1. Create DataFrame from trade results
    data_list = []
    for trade in trade_results:
        trade_data = {
            'Date': trade.exit_time,
            'Instrument': trade.instrument,
            'Signal': trade.signal,
            'Type': trade.entry_type,
            'Entry Price': trade.price,
            'Stop Loss': trade.original_stop_loss,
            'Take Profit': trade.take_profit,
            'Filled Time': trade.filled_time,
            'Exit Time': trade.exit_time,
            'Exit Price': trade.exit_price
        }
        data_list.append(trade_data)
    df = pd.DataFrame(data_list)
     
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # 2. Calculate Basic Statistics
    df['Risk'] = df.apply(
        lambda row: row['Entry Price'] - row['Stop Loss'] if row['Signal'] in ['bull_eng', 'hammer']
        else row['Stop Loss'] - row['Entry Price'], axis=1
    )
    df['Reward'] = df.apply(
        lambda row: row['Exit Price'] - row['Entry Price'] if row['Signal'] in ['bull_eng', 'hammer']
        else row['Entry Price'] - row['Exit Price'], axis=1
    )
    df['R_Ratio'] = df.apply(
        lambda row: row['Reward'] / row['Risk'] if row['Risk'] != 0 else 0, axis=1
    )

    total_trades = len(df)
    total_r = df['R_Ratio'].sum()
    average_r = df['R_Ratio'].mean()

    stats = {
        'Total Trades': total_trades,
        'Total R': round(total_r, 2),
        'Average R per Trade': round(average_r, 2)
    }

    # 3. Print Basic Statistics
    print(f"\n--- Basic Trade Statistics for {name} ---")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(df)

    # 4. Plot Equity Curve
    df_sorted = df.sort_values('Exit Time').reset_index(drop=True)
    df_sorted['Cumulative R'] = df_sorted['R_Ratio'].cumsum()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sorted, x='Exit Time', y='Cumulative R', marker='o')
    plt.title(f'Equity Curve for {name}')
    plt.xlabel('Exit Time')
    plt.ylabel('Cumulative R')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

"""
    # Define output directories
    csv_output_dir = os.path.join(results_path, 'csv', name)
    plots_output_dir = os.path.join(results_path, 'plots', name)

    # Ensure directories exist
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)

    # Save Equity Curve Plot
    equity_curve_filename = f'equity_curve_{name}.png'
    equity_curve_path = os.path.join(plots_output_dir, equity_curve_filename)
    plt.savefig(equity_curve_path)  # Save the plot as an image file
    plt.close()  # Close the plot to free memory
    logging.info(f"Equity curve saved as {equity_curve_path}.")

    # Save Trade Results CSV
    trade_results_filename = f'trade_results_with_metrics_{name}.csv'
    trade_results_path = os.path.join(csv_output_dir, trade_results_filename)
    df.to_csv(trade_results_path, index=False)
    logging.info(f"Trade analysis completed and results saved to '{trade_results_path}'.")
"""

