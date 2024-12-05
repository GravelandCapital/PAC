import pandas as pd 
from entry import Entry
from engulfing_handler import EngulfingHandler
from hammer_shooting_star_handler import HammerShootingStarHandler

def calculate_zigzag_daily(df_daily, depth=3):
    """Calculates ZigZag pivots for the provided depth in the daily data."""
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
    return daily_zigzag

def process_signals(df_daily, df_hourly, zigzag_df, instrument):
    entries = []
    for row_index in df_daily.index:
        signal = df_daily.loc[row_index, 'signal']
        if signal == 'bull_eng' or signal == 'bear_eng':
            handler = EngulfingHandler(df_daily, df_hourly, row_index, zigzag_df, instrument)
        elif signal == 'hammer' or signal == 'shooting_star':
            handler = HammerShootingStarHandler(df_daily, df_hourly, row_index, zigzag_df, instrument)
        else:
            continue  # No signal, skip

        # Calculate entries
        signal_entries = handler.calculate_entries()
        entries.extend(signal_entries)
    return entries

def calculate_sl_tp(entries, df_daily, df_hourly, zigzag_df, daily_zigzag, instrument):
    entry_rr_list = []

    for entry in entries:
        if entry.signal in ['bull_eng', 'bear_eng']:
            handler = EngulfingHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        elif entry.signal in ['hammer', 'shooting_star']:
            handler = HammerShootingStarHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        else:
            continue

        # Calculate stop loss and take profit
        entry.stop_loss, entry.original_stop_loss = handler.calculate_stop_loss(entry)
        entry.take_profit = handler.calculate_take_profit(entry, daily_zigzag)

        # Ensure stop loss and take profit are set
        if entry.stop_loss and entry.take_profit:
            # Calculate risk/reward ratio
            risk = abs(entry.price - entry.stop_loss)
            reward = abs(entry.take_profit - entry.price)
            rr_ratio = reward / risk if risk > 0 else 0

            # Append to final list only if R/R ratio is at least 1.5
            if rr_ratio >= 1.5:
                entry_rr_list.append((entry, rr_ratio))
            
    if not entry_rr_list: 
        entries[:] = []  # Clear entries if no valid R/R ratios
        return

    # Select entries with highest R/R ratios
    best_entry, best_rr = max(entry_rr_list, key=lambda x: x[1])

    # Replace entries with the best entry
    entries[:] = [best_entry]
    
    print(f"Selected entry with type {best_entry.entry_type}, price {best_entry.price}, RR {best_rr:.2f}")

def extract_instrument_from_filename(filename):
    """Extracts the instrument name from the filename."""
    return filename.split('_')[0] + '_' + filename.split('_')[1]

def analyze_results(trade_results):
    total_r = 0  # Initialize total R
    print("\nTrade Analysis:\n")
    
    for trade in trade_results:
        # Determine risk and reward based on signal type
        if trade.signal in ['bull_eng', 'hammer']:
            risk = trade.price - trade.original_stop_loss
            reward = trade.exit_price - trade.price
        elif trade.signal in ['bear_eng', 'shooting_star']:
            risk = trade.original_stop_loss - trade.price
            reward = trade.price - trade.exit_price
        else:
            risk = 0
            reward = 0

        # Calculate R with safeguard against divide-by-zero
        if risk > 0:
            r = reward / risk
        else:
            r = 0  # Set R to 0 if risk is zero

        # Accumulate total R
        total_r += r

        # Print trade details
        print(f"Trade Signal: {trade.signal}, filled: {trade.filled_time}")
        print(f"entry price: {trade.price}, stop: {trade.original_stop_loss}, take profit: {trade.take_profit}")
        print(f"exit price: {trade.exit_price}, exit time: {trade.exit_time}")
        print(f"Risk: {risk:.2f}, Reward: {reward:.2f}, R: {r:.2f}\n")

    print(f"Total R for all trades: {total_r:.2f}")
    print(f"Number of trades: {len(trade_results)}")
    return total_r