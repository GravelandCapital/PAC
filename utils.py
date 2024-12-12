# utils.py

import pandas as pd
from collections import defaultdict
from entry import Entry
from engulfing_handler import EngulfingHandler
from hammer_shooting_star_handler import HammerShootingStarHandler
import logging

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
    """
    Processes trading signals and generates Entry objects.
    """
    entries = []
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
    logging.info("Starting calculate_sl_tp")

    processed_entries = []
    for entry in entries:
        if entry.signal in ['bull_eng', 'bear_eng']:
            handler = EngulfingHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        elif entry.signal in ['hammer', 'shooting_star']:
            handler = HammerShootingStarHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        else:
            logging.warning(f"Unknown signal {entry.signal} at row {entry.row_index}, skipping")
            continue

        # Calculate stop loss and take profit
        try:
            stop_loss_list = handler.calculate_stop_loss(entry)
            entry.take_profit = handler.calculate_take_profit(entry, daily_zigzag)
        except Exception as e:
            logging.error(f"Error calculating SL/TP for Entry at row {entry.row_index}: {e}")
            continue

        if entry.take_profit is None or not stop_loss_list:
            logging.warning(f"Invalid TP or empty SL list for Entry at row {entry.row_index}, skipping")
            continue    

        valid_combos = generate_valid_combinations(stop_loss_list, entry)
        logging.info(f"Valid combos for Entry at row {entry.row_index}: {valid_combos}")

        if not valid_combos:
            logging.warning(f"No valid combos for Entry at row {entry.row_index}, skipping")
            continue  # Skip to the next entry if no valid combinations

        best_trade = select_best_trade(valid_combos, entry)
        if best_trade:
            # Update the Entry object with the best trade details
            entry.price = best_trade['entry_price']
            entry.stop_loss = best_trade['stop_loss']
            entry.original_stop_loss = best_trade['stop_loss']
            entry.take_profit = best_trade['take_profit']
            processed_entries.append(entry)
            logging.info(f"Processed Entry at row {entry.row_index}: {entry}")
        else:
            logging.warning(f"No best trade found for Entry at row {entry.row_index}, skipping")

    # Now, group the processed entries
    grouped_entries = defaultdict(list)
    for entry in processed_entries:
        key = get_group_key(entry)
        if key:
            grouped_entries[key].append(entry)
            logging.info(f"Grouped Entry at row {entry.row_index} under key {key}")

    final_entries = []
    for key, group in grouped_entries.items():
        logging.info(f"Processing group {key} with {len(group)} entries")
        is_bullish = group[0].signal in ['bull_eng', 'hammer']

        if is_bullish:
            # Select the entry with the highest entry_price
            best_entry = max(group, key=lambda x: x.price)
            logging.info(f"Selected best bullish Entry from group {key}: {best_entry}")
        else:
            # Select the entry with the lowest entry_price
            best_entry = min(group, key=lambda x: x.price)
            logging.info(f"Selected best bearish Entry from group {key}: {best_entry}")

        final_entries.append(best_entry)

    # Replace entries with final list that meets R/R condition
    entries[:] = final_entries
    logging.info(f"Final entries count after SL and TP calculations: {len(final_entries)}")

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

        logging.debug(f"Entry at row {entry.row_index}: SL={stop_loss}, TP={take_profit}, R:R={rr_ratio}")

        if rr_ratio >= 1.5:
            valid_combos.append({'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit})
            logging.info(f"Valid combo added for Entry at row {entry.row_index}: {valid_combos[-1]}")
        else: 
            logging.info(f"Combo with SL={stop_loss} and TP={take_profit} discarded due to low R:R ratio")

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
        logging.info(f"Best bullish trade for Entry at row {entry.row_index}: {best_trade}")
    else:
        # Select the combo with the lowest entry_price
        min_entry_price = min(comb['entry_price'] for comb in valid_combos)
        min_entry_combinations = [comb for comb in valid_combos if comb['entry_price'] == min_entry_price]
        best_trade = min(min_entry_combinations, key=lambda x: x['stop_loss'])
        logging.info(f"Best bearish trade for Entry at row {entry.row_index}: {best_trade}")

    return best_trade

def extract_instrument_from_filename(filename):
    """Extracts the instrument name from the filename."""
    return filename.split('_')[0] + '_' + filename.split('_')[1]

def analyze_results(trade_results):
    total_r = 0  # Initialize total R
    logging.info("\nTrade Analysis:\n")
    
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

        # Log trade details
        logging.info(f"Trade Signal: {trade.signal}, filled: {trade.filled_time}")
        logging.info(f"Entry price: {trade.price}, Stop: {trade.original_stop_loss}, Take Profit: {trade.take_profit}")
        logging.info(f"Exit price: {trade.exit_price}, Exit time: {trade.exit_time}")
        logging.info(f"Risk: {risk:.2f}, Reward: {reward:.2f}, R: {r:.2f}\n")

    logging.info(f"Total R for all trades: {total_r:.2f}")
    logging.info(f"Number of trades: {len(trade_results)}")
    return total_r
