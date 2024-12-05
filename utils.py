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
    final_entries = []
    daily_zigzag = calculate_zigzag_daily(df_daily, depth=3)
    for row_index in df_daily.index:
        signal = df_daily.loc[row_index, 'signal']
        if signal == 'bull_eng' or signal == 'bear_eng':
            handler = EngulfingHandler(df_daily, df_hourly, row_index, zigzag_df, instrument, daily_zigzag)
        elif signal == 'hammer' or signal == 'shooting_star':
            handler = HammerShootingStarHandler(df_daily, df_hourly, row_index, zigzag_df, instrument, daily_zigzag)
        else:
            continue  # No signal, skip

        # Calculate entries
        entries = handler.calculate_entries()
        if not entries:
            continue

        valid_combinations = handler.generate_valid_combinations(entries)
        if not valid_combinations:
            continue

        best_trade = handler.select_best_trade(valid_combinations)
        if best_trade:
            # Update the entry with selected stop loss and take profit
            entry = best_trade['entry']
            entry.stop_loss = best_trade['stop_loss']
            entry.take_profit = best_trade['take_profit']
            entry.rr_ratio = best_trade['rr_ratio']
            final_entries.append(entry)

    return final_entries


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