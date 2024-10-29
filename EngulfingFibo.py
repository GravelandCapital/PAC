import pandas as pd
import numpy as np
import os
from dataclasses import dataclass

# Define the instruments (if needed for future extensions)
instruments = ['EUR_USD']

@dataclass
class Entry:
    instrument: str
    signal: str
    entry_type: str
    price: float
    time: pd.Timestamp
    row_index: int  # Index in the daily dataframe
    stop_loss: float = None
    take_profit: float = None

class DataFetcher:
    """Class to fetch daily and hourly data from Excel files."""
    def __init__(self, daily_path, hourly_path):
        self.daily_path = daily_path
        self.hourly_path = hourly_path
        self.df_daily = None
        self.df_hourly = None

    def fetch_data(self):
        """Reads data from Excel files and stores them in DataFrame attributes."""
        self.df_daily = pd.read_excel(self.daily_path)
        self.df_hourly = pd.read_excel(self.hourly_path)

        # Convert 'time' columns to datetime format
        self.df_daily['time'] = pd.to_datetime(self.df_daily['time']) + pd.Timedelta(hours=24)
        self.df_hourly['time'] = pd.to_datetime(self.df_hourly['time'])

        print(f"Data fetched for {self.daily_path} and {self.hourly_path}")

class SignalCalculator:
    """Class to calculate trading signals and indicators."""
    def __init__(self, df_hourly, df_daily):
        self.df_hourly = df_hourly
        self.df_daily = df_daily

    def calculate_signal(self):
        """Calculates signals like bull_eng, bear_eng, hammer, and shooting star."""
        self.df_daily['signal'] = None

        for i in range(1, len(self.df_daily) - 1):
            prev_row = self.df_daily.iloc[i - 1]
            curr_row = self.df_daily.iloc[i]
            high = curr_row["h"]
            low = curr_row["l"]
            close = curr_row["c"]
            open_price = curr_row["o"]
            prev_high = prev_row["h"]
            prev_low = prev_row["l"]
            candle_range = high - low
            lower_wick = min(close, open_price) - low
            upper_wick = high - max(close, open_price)

            if prev_low > low and close > prev_high:
                self.df_daily.loc[i, 'signal'] = "bull_eng"
            elif prev_high < high and close < prev_low:
                self.df_daily.loc[i, 'signal'] = "bear_eng"
            elif (lower_wick >= 0.67 * candle_range and low < prev_low and
                  close < prev_high and close > prev_low and high < prev_high):
                self.df_daily.loc[i, 'signal'] = "hammer"
            elif (upper_wick >= 0.67 * candle_range and high > prev_high and
                  close > prev_low and close < prev_high and low > prev_low):
                self.df_daily.loc[i, 'signal'] = "shooting_star"
            else:
                self.df_daily.loc[i, 'signal'] = None

        return self.df_daily

    def calculate_atr(self, n=14):
        """Calculates the Average True Range (ATR) indicator."""
        self.df_daily['tr0'] = abs(self.df_daily['h'] - self.df_daily['l'])
        self.df_daily['tr1'] = abs(self.df_daily['h'] - self.df_daily['c'].shift(1))
        self.df_daily['tr2'] = abs(self.df_daily['l'] - self.df_daily['c'].shift(1))
        self.df_daily['tr'] = self.df_daily[['tr0', 'tr1', 'tr2']].max(axis=1)
        self.df_daily['atr'] = self.df_daily['tr'].ewm(alpha=1 / n, adjust=False).mean()
        return self.df_daily

class EngulfingHandler:
    """Handler for Engulfing signals."""
    def __init__(self, df_daily, df_hourly, row_index, zigzag_df, instrument):
        self.df_daily = df_daily
        self.df_hourly = df_hourly
        self.row_index = row_index
        self.signal = df_daily.loc[row_index, 'signal']
        self.zigzag_df = zigzag_df
        self.instrument = instrument

    def calculate_entries(self):
        entries = []
        fib_entry = self.calculate_fib_entry()
        if fib_entry:
            entries.append(fib_entry)
        lhpb_entry = self.calculate_lhpb_entry()
        if lhpb_entry:
            entries.append(lhpb_entry)
        
        # Select max (for bull_Eng) and min (for bear_Eng) price levels
        if entries: 
            if self.signal == "bull_eng":
                best_entry = max(entries, key=lambda x: x.price)
            elif self.signal == "bear_eng":
                best_entry = min(entries, key=lambda x: x.price)
            else:
                best_entry = None
            return [best_entry] if best_entry else []
        else: 
            return []

    def calculate_fib_entry(self):
        fibo_level, half_level = self.calculate_fibo()
        if fibo_level is None:
            return None

        # Get naked levels
        hourly_data = self.get_hourly_data()
        naked_levels = self.calculate_naked_level(hourly_data)

        # Calculate ATR range
        atr_range = self.df_daily.loc[self.row_index, 'atr'] * 0.1

        # Define comparison function based on signal
        if self.signal == "bull_eng":
            compare = lambda level: level > half_level
        elif self.signal == "bear_eng":
            compare = lambda level: level < half_level
        else:
            return None

        # Find matching naked level
        for naked_level in naked_levels:
            difference = abs(naked_level - fibo_level)
            if difference <= atr_range and compare(naked_level):
                # Entry found
                entry_time = self.find_entry_time(hourly_data, naked_level)
                return Entry(
                    instrument = self.instrument,
                    signal=self.signal,
                    entry_type='fib',
                    price=naked_level,
                    time=entry_time,
                    row_index=self.row_index
                )
        return None

    def calculate_lhpb_entry(self):
        entries = []  # Collect all valid entries
        fibo_level, half_level = self.calculate_fibo()

        if fibo_level is None:
            return None

        # Get filtered pivots
        filtered_pivots = self.find_filtered_pivots()

        hourly_data = self.get_hourly_data()

        if self.signal == "bull_eng":
            # Iterate through pivots for a bull engulfing signal
            for _, pivot in filtered_pivots.iterrows():
                pivot_price = pivot['price']
                
                for idx in range(len(hourly_data)):
                    row = hourly_data.iloc[idx]
                    if row['c'] > pivot_price:
                        if idx > 0:
                            last_high_pre_break = hourly_data.iloc[idx - 1]['h']

                        else:
                            break
                        future_candles = hourly_data.iloc[idx + 1:]
                        if (future_candles['l'] > last_high_pre_break).all() and last_high_pre_break < pivot_price and last_high_pre_break > half_level:
                            entry_time = self.find_entry_time(hourly_data, last_high_pre_break)
                            entries.append(Entry(
                                instrument=self.instrument,
                                signal=self.signal,
                                entry_type='LHPB',
                                price=last_high_pre_break,
                                time=entry_time,
                                row_index=self.row_index
                            ))
                        else: 
                            break # Stop searching if the condition is not met

        elif self.signal == "bear_eng":
            # Iterate through pivots for a bear engulfing signal
            for _, pivot in filtered_pivots.iterrows():
                pivot_price = pivot['price']

                for idx in range(len(hourly_data)):
                    row = hourly_data.iloc[idx]
                    if row['c'] < pivot_price:
                        if idx > 0:
                            last_low_pre_break = hourly_data.iloc[idx - 1]['l']
                        else:
                            break
                        future_candles = hourly_data.iloc[idx + 1:]
                        if (future_candles['h'] < last_low_pre_break).all() and last_low_pre_break > pivot_price and last_low_pre_break < half_level:
                            entry_time = self.find_entry_time(hourly_data, last_low_pre_break)
                            entries.append(Entry(
                                instrument=self.instrument,
                                signal=self.signal,
                                entry_type='LLPB',
                                price=last_low_pre_break,
                                time=entry_time,
                                row_index=self.row_index
                            ))
                        else: 
                            break # Stop searching if the condition is not met

        # After collecting all entries, return the max for LHPB and min for LLPB
        if entries:
            if self.signal == "bull_eng":
                best_entry = max(entries, key=lambda x: x.price)
            elif self.signal == "bear_eng":
                best_entry = min(entries, key=lambda x: x.price)
            return best_entry

        return None


    def calculate_fibo(self):
        signal = self.signal
        high = self.df_daily.loc[self.row_index, 'h']
        low = self.df_daily.loc[self.row_index, 'l']
        diff = high - low
        if signal == "bull_eng":
            fibo_level = high - diff * 0.382
            half_level = high - diff * 0.5
        elif signal == "bear_eng":
            fibo_level = low + diff * 0.382
            half_level = low + diff * 0.5
        else:
            fibo_level = None
            half_level = None
        return fibo_level, half_level

    def calculate_naked_level(self, hourly_data):
        naked_levels = []
        signal = self.signal

        for i in range(1, len(hourly_data)):
            prev_row = hourly_data.iloc[i - 1]
            curr_row = hourly_data.iloc[i]

            if signal == "bull_eng":
                previous_high = prev_row["h"]
                current_close = curr_row["c"]

                if current_close > previous_high:
                    future_candles = hourly_data.iloc[i + 1:]
                    if (future_candles['l'] >= previous_high).all():
                        naked_levels.append(previous_high)

            elif signal == "bear_eng":
                previous_low = prev_row["l"]
                current_close = curr_row["c"]

                if current_close < previous_low:
                    future_candles = hourly_data.iloc[i + 1:]
                    if (future_candles['h'] <= previous_low).all():
                        naked_levels.append(previous_low)
        return naked_levels

    def find_filtered_pivots(self):
        signal = self.signal
        signal_date = self.df_daily.loc[self.row_index, 'time'].date()
        signal_day = self.df_daily.loc[self.row_index, 'time']
        daily_close = self.df_daily.loc[self.row_index, 'c']
        daily_high = self.df_daily.loc[self.row_index, 'h']
        daily_low = self.df_daily.loc[self.row_index, 'l']

        if signal == "bull_eng":
            relevant_pivots = self.zigzag_df[
                (self.zigzag_df['price'] < daily_close) &
                (self.zigzag_df['price'] > daily_low) &
                (self.zigzag_df['type'] == 'h')
            ].sort_values('time', ascending=False)

            # Segment pivots into those on the signal day and before
            relevant_pivots_including = relevant_pivots[relevant_pivots['time'].dt.date == signal_date]
            relevant_pivots_excluding = relevant_pivots[relevant_pivots['time'].dt.date < signal_date]

            # Initialize the filtered pivots list
            filtered_pivots = []

            # Process pivots on the signal day
            for _, pivot in relevant_pivots_including.iterrows():
                filtered_pivots.append(pivot)

            # Initialize the highest pivot variable
            highest_pivot = None

            # Process pivots before the signal day
            for _, pivot in relevant_pivots_excluding.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']

                # First pivot is always appended
                if highest_pivot is None:
                    filtered_pivots.append(pivot)
                    highest_pivot = pivot_price
                elif pivot_price > highest_pivot:
                    future_candles = self.df_hourly[
                        (self.df_hourly['time'] > pivot_time) &
                        (self.df_hourly['time'] < signal_day - pd.Timedelta(hours=24))
                    ]
                    future_highs = future_candles['h'].max() if not future_candles.empty else None

                    if future_highs is None or pivot_price > future_highs:
                        filtered_pivots.append(pivot)
                        highest_pivot = pivot_price

                    if future_highs is not None and future_highs > daily_close:
                        break

        elif signal == "bear_eng":
            relevant_pivots = self.zigzag_df[
                (self.zigzag_df['price'] > daily_close) &
                (self.zigzag_df['price'] < daily_high) &
                (self.zigzag_df['type'] == 'l')
            ].sort_values('time', ascending=False)

            # Segment pivots into those on the signal day and before
            relevant_pivots_including = relevant_pivots[relevant_pivots['time'].dt.date == signal_date]
            relevant_pivots_excluding = relevant_pivots[relevant_pivots['time'].dt.date < signal_date]

            # Initialize the filtered pivots list
            filtered_pivots = []

            # Process pivots on the signal day
            for _, pivot in relevant_pivots_including.iterrows():
                filtered_pivots.append(pivot)

            # Initialize the lowest pivot variable
            lowest_pivot = None

            # Process pivots before the signal day
            for _, pivot in relevant_pivots_excluding.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']

                # First pivot is always appended
                if lowest_pivot is None:
                    filtered_pivots.append(pivot)
                    lowest_pivot = pivot_price
                elif pivot_price < lowest_pivot:
                    future_candles = self.df_hourly[
                        (self.df_hourly['time'] > pivot_time) &
                        (self.df_hourly['time'] < signal_day - pd.Timedelta(hours=24))
                    ]
                    future_lows = future_candles['l'].min() if not future_candles.empty else None

                    if future_lows is None or pivot_price < future_lows:
                        filtered_pivots.append(pivot)
                        lowest_pivot = pivot_price

                    if future_lows is not None and future_lows < daily_close:
                        break
        else:
            filtered_pivots = []

        filtered_pivots = pd.DataFrame(filtered_pivots)
        return filtered_pivots

    def get_hourly_data(self):
        start = self.df_daily.loc[self.row_index, 'time'] - pd.Timedelta(hours=24)
        end = self.df_daily.loc[self.row_index, 'time'] - pd.Timedelta(hours=1)
        hourly_data = self.df_hourly[(self.df_hourly['time'] >= start) & (self.df_hourly['time'] <= end)]
        return hourly_data

    def find_entry_time(self, hourly_data, price_level):
        # Find the time when the price level was reached
        if self.signal == "bull_eng":
            entry_candle = hourly_data[hourly_data['h'] == price_level]
        elif self.signal == "bear_eng":
            entry_candle = hourly_data[hourly_data['l'] == price_level]
        if not entry_candle.empty:
            return entry_candle.iloc[0]['time']
        else:
            return None

    def get_pip_value(self, instrument):
        """Returns the pip value for a given currency pair."""
        # If the pair involves JPY, pip is 0.01, else it's 0.0001
        if "JPY" in instrument:
            return 0.01
        else:
            return 0.0001

    def calculate_stop_loss(self, entry):
        sl_pivots = self.sl_pivots(entry.price)
        entry_candle_time = entry.time
        pip_value = self.get_pip_value(self.instrument)
        min_atr = self.df_daily.loc[self.row_index, 'atr'] * 0.4
        entry_price = entry.price

        # Define the end time as the last hourly candle of the signal (i.e., the end of the daily candle)
        daily_signal_time = self.df_daily.loc[self.row_index, 'time']
        end_time = entry_candle_time

        # Initialize list to store all potential stop losses
        stop_loss_list = []

        if self.signal == "bull_eng":
            failure_point = self.df_daily.loc[self.row_index, 'l']
            signal_high = self.df_daily.loc[self.row_index, 'h']
            stop_loss = failure_point - pip_value

            for _, pivot in sl_pivots.iterrows():
                original_pivot_price = pivot['price']  # Store the original pivot price
                pivot_price = original_pivot_price  # Initialize pivot price as original pivot

                # Filter hourly data between the pivot time and the end time (last hourly candle of the daily signal)
                hourly_data = self.df_hourly[
                    (self.df_hourly['time'] >= pivot['time']) & (self.df_hourly['time'] <= end_time)
                ]

                for idx in range(len(hourly_data)):
                    row = hourly_data.iloc[idx]
                    
                    # If the candle high exceeds the signal high, stop the search, no more valid pivots
                    if row['h'] > signal_high:
                        break

                    # Update the pivot price if the candle wicks but does not close above the pivot
                    if row['h'] > pivot_price and row['c'] <= pivot_price:
                        pivot_price = row['h']

                    # Check if current close is above the pivot price
                    if row['c'] > pivot_price:

                        # If the close is above, store future candles up to the end time
                        future_candles = hourly_data.iloc[idx + 1:]
                        # Find the minimum low of all future candles
                        future_min_low = future_candles['l'].min()


                        # Validate the stop loss based on future candles
                        if future_min_low > row['l']:
                            temp_stop_loss = row['l']
                            stop_loss_value = entry_price - stop_loss
                            # Final check to ensure stop loss is not less than the original pivot price
                            if temp_stop_loss < original_pivot_price and stop_loss_value >= min_atr:
                                temp_stop_loss = row['l'] - pip_value
                                stop_loss_list.append(temp_stop_loss)
                                continue  # Move to the next pivot once a valid stop loss is stored
                            else: 
                                continue

            # Return the max stop loss for a bullish signal
            if stop_loss_list:
                max_stop_loss = max(stop_loss_list)
                return max_stop_loss
            else:
                return stop_loss

        elif self.signal == "bear_eng":
            failure_point = self.df_daily.loc[self.row_index, 'h']
            signal_low = self.df_daily.loc[self.row_index, 'l']
            stop_loss = failure_point + pip_value

            for _, pivot in sl_pivots.iterrows():
                original_pivot_price = pivot['price']  # Store the original pivot price
                pivot_price = original_pivot_price  # Initialize pivot price as original pivot

                # Filter hourly data between the pivot time and the end time (last hourly candle of the daily signal)
                hourly_data = self.df_hourly[
                    (self.df_hourly['time'] >= pivot['time']) & (self.df_hourly['time'] <= end_time)
                ]

                for idx in range(len(hourly_data)):
                    row = hourly_data.iloc[idx]
                    
                    # If the candle low is below the signal low, stop the search, no more valid pivots
                    if row['l'] < signal_low:
                        break

                    # Update the pivot price if the candle wicks but does not close below the pivot
                    if row['l'] < pivot_price and row['c'] >= pivot_price:
                        pivot_price = row['l']

                    # Check if current close is below the pivot price
                    if row['c'] < pivot_price:

                        # If the close is below, store future candles up to the end time
                        future_candles = hourly_data.iloc[idx + 1:]
                        # Find the maximum high of all future candles
                        future_max_high = future_candles['h'].max()


                        # Validate the stop loss based on future candles
                        if future_max_high < row['h']:
                            temp_stop_loss = row['h']
                            stop_loss_value = stop_loss - entry_price
                            # Final check to ensure stop loss is not more than the original pivot price
                            if temp_stop_loss > original_pivot_price and stop_loss_value >= min_atr:
                                temp_stop_loss = row['h'] + pip_value
                                stop_loss_list.append(temp_stop_loss)
                                continue  # Move to the next pivot once a valid stop loss is stored
                            else:
                                continue
                    
            # Return the min stop loss for a bearish signal
            if stop_loss_list:
                min_stop_loss = min(stop_loss_list)
                return min_stop_loss
            else:
                return stop_loss


    def sl_pivots(self, entry_price):
        entry_time = self.df_daily.loc[self.row_index, 'time']
        threshold_time = entry_time - pd.Timedelta(days=365)
        confirmation_time = entry_time - pd.Timedelta(hours = 4)
        daily_high = self.df_daily.loc[self.row_index, 'h']
        daily_low = self.df_daily.loc[self.row_index, 'l']

        if self.signal == "bull_eng":
            relevant_pivots = self.zigzag_df[
                (self.zigzag_df['time'] < confirmation_time) & 
                (self.zigzag_df['price'] < entry_price) &
                (self.zigzag_df['time'] >= threshold_time) &
                (self.zigzag_df['type'] == 'h')
            ].sort_values('time', ascending=False)


            sl_pivots = []
            highest_pivot = None

            for _, pivot in relevant_pivots.iterrows():
                pivot_price = pivot['price']

                if pivot_price < daily_high:
                    if highest_pivot is None or (pivot_price > highest_pivot):
                        sl_pivots.append(pivot)
                        highest_pivot = pivot_price
                else:
                    break

        elif self.signal == "bear_eng":
            relevant_pivots = self.zigzag_df[
                (self.zigzag_df['time'] < confirmation_time) & 
                (self.zigzag_df['price'] > entry_price) &
                (self.zigzag_df['time'] >= threshold_time) &
                (self.zigzag_df['type'] == 'l')
            ].sort_values('time', ascending=False)


            sl_pivots = []
            lowest_pivot = None

            for _, pivot in relevant_pivots.iterrows():
                pivot_price = pivot['price']

                if pivot_price > daily_low:
                    if lowest_pivot is None or (pivot_price < lowest_pivot):
                        sl_pivots.append(pivot)
                        lowest_pivot = pivot_price
                else:
                    break

        # Check if sl_pivots has any entries; if not, return an empty DataFrame
        if not sl_pivots:
            return pd.DataFrame()  # Return an empty DataFrame if no pivots

        sl_pivots_df = pd.DataFrame(sl_pivots)
        return sl_pivots_df




    def calculate_take_profit(self, entry, daily_zigzag):
        entry_price = entry.price
        entry_time = self.df_daily.loc[self.row_index, 'time']
        depth = 3
        confirmation_index = self.row_index - depth
        if confirmation_index < 0:
            return None

        adjusted_entry_time = self.df_daily.loc[confirmation_index, 'time']

        # Get all pivots before the entry time
        valid_pivots = daily_zigzag[daily_zigzag['time'] < adjusted_entry_time]

        if self.signal == 'bull_eng':
            pivot_highs = valid_pivots[
                (valid_pivots['type'] == 'h') &
                (valid_pivots['price'] > entry_price)
            ]
            if not pivot_highs.empty:
                for idx, pivot in pivot_highs.sort_values('time', ascending=False).iterrows():
                    pivot_time = pivot['time']
                    pivot_price = pivot['price']
                    intermediate_highs = self.df_daily[
                        (self.df_daily['time'] > pivot_time) &
                        (self.df_daily['time'] <= entry_time)
                    ]['h']
                    if (intermediate_highs < pivot_price).all():
                        tp_level = pivot_price
                        return tp_level
                    else:
                        continue
        elif self.signal == 'bear_eng':
            pivot_lows = valid_pivots[
                (valid_pivots['type'] == 'l') &
                (valid_pivots['price'] < entry_price)
            ]
            if not pivot_lows.empty:
                for idx, pivot in pivot_lows.sort_values('time', ascending=False).iterrows():
                    pivot_time = pivot['time']
                    pivot_price = pivot['price']
                    intermediate_lows = self.df_daily[
                        (self.df_daily['time'] > pivot_time) &
                        (self.df_daily['time'] <= entry_time)
                    ]['l']
                    if (intermediate_lows > pivot_price).all():
                        tp_level = pivot_price
                        return tp_level
                    else:
                        continue
        return None

class HammerShootingStarHandler:
    """Handler for Hammer and Shooting Star signals."""
    def __init__(self, df_daily, df_hourly, row_index, zigzag_df, instrument):
        self.df_daily = df_daily
        self.df_hourly = df_hourly
        self.row_index = row_index
        self.signal = df_daily.loc[row_index, 'signal']
        self.zigzag_df = zigzag_df
        self.instrument = instrument

    def calculate_entries(self):
        entries = []
        pdh_pdl_entries = self.calculate_pdh_pdl_entry()
        entries.extend(pdh_pdl_entries)
        gowith_entries = self.calculate_gowith_entry()
        entries.extend(gowith_entries)

        # For PDL entries, select the max entry price
        # For PDH entries, select the min entry price
        if entries:
            pdh_pdl_entries = [e for e in entries if e.entry_type in ['PDH', 'PDL']]
            other_entries = [e for e in entries if e.entry_type not in ['PDH', 'PDL']]

            # Process PDH/PDL entries
            if pdh_pdl_entries:
                if any(e.entry_type == 'PDL' for e in pdh_pdl_entries):
                    # For PDL, select max price
                    pdl_entries = [e for e in pdh_pdl_entries if e.entry_type == 'PDL']
                    best_pdl_entry = max(pdl_entries, key=lambda x: x.price)
                    processed_entries = [best_pdl_entry]
                elif any(e.entry_type == 'PDH' for e in pdh_pdl_entries):
                    # For PDH, select min price
                    pdh_entries = [e for e in pdh_pdl_entries if e.entry_type == 'PDH']
                    best_pdh_entry = min(pdh_entries, key=lambda x: x.price)
                    processed_entries = [best_pdh_entry]
                else:
                    processed_entries = []
            else:
                processed_entries = []

            # Add other entries (GWSS, GWHMR) without modification
            processed_entries.extend(other_entries)
            return processed_entries
        else:
            return []

    def calculate_pdh_pdl_entry(self):
        signal = self.signal
        start = self.df_daily.loc[self.row_index, 'time'] - pd.Timedelta(hours=24)
        end = self.df_daily.loc[self.row_index, 'time'] - pd.Timedelta(hours=1)
        hourly_data = self.df_hourly[(self.df_hourly['time'] >= start) & (self.df_hourly['time'] <= end)]
        entries = []
        if self.row_index > 0:
            pdl = self.df_daily.loc[self.row_index - 1, 'l']
            pdh = self.df_daily.loc[self.row_index - 1, 'h']

        # Previous day's low and high entry for hammer and shooting star
        if signal == "hammer":
            for i in range(len(hourly_data) - 1):
                row = hourly_data.iloc[i]
                high = row['h']
                next_candle_close = hourly_data.iloc[i + 1]['c']
                future_candles = hourly_data.iloc[i + 2:]

                if high < pdl and next_candle_close > high and (future_candles['l'] > high).all():
                    entry_time = row['time']
                    entry_price = high
                    entry = Entry(
                        instrument = self.instrument,
                        signal=signal,
                        entry_type='PDL',
                        price=entry_price,
                        time=entry_time,
                        row_index=self.row_index
                    )
                    entries.append(entry)
        elif signal == "shooting_star":
            for i in range(len(hourly_data) - 1):
                row = hourly_data.iloc[i]
                low = row['l']
                next_candle_close = hourly_data.iloc[i + 1]['c']
                future_candles = hourly_data.iloc[i + 2:]

                if low > pdh and next_candle_close < low and (future_candles['h'] < low).all():
                    entry_time = row['time']
                    entry_price = low
                    entry = Entry(
                        instrument=self.instrument,
                        signal=signal,
                        entry_type='PDH',
                        price=entry_price,
                        time=entry_time,
                        row_index=self.row_index
                    )
                    entries.append(entry)
        return entries

    def calculate_gowith_entry(self):
        signal = self.signal
        signal_time = self.df_daily.loc[self.row_index, 'time']
        entries = []
        relevant_hourly = self.df_hourly[self.df_hourly['time'] >= signal_time].reset_index(drop=True)

        if signal == "hammer":
            hammer_high = self.df_daily.loc[self.row_index, 'h']
            breakout_level = hammer_high
            failure_point = self.df_daily.loc[self.row_index, 'l']
            highs_between = [hammer_high]
            failed_breakout_count = 0
            i = 0
            while i < len(relevant_hourly) - 1:
                row = relevant_hourly.iloc[i]
                close = row['c']
                high = row['h']
                low = row['l']

                highs_between.append(high)

                if close > breakout_level:
                    # Remove the breakout candle's high from highs_between temporarily
                    breakout_candle_high = highs_between.pop()

                    if i + 1 < len(relevant_hourly):
                        next_row = relevant_hourly.iloc[i + 1]
                        next_low = next_row['l']
                        next_high = next_row['h']
                    else:
                        break  # No more data

                    max_high = max(highs_between) if highs_between else breakout_level

                    if next_low > max_high:
                        # Valid breakout
                        entry_time = row['time']
                        entry_price = max_high
                        entry = Entry(
                            instrument=self.instrument,
                            signal=signal,
                            entry_type='GWHMR',
                            price=entry_price,
                            time=entry_time,
                            row_index=self.row_index
                        )
                        entries.append(entry)
                        break
                    else:
                        # Failed breakout
                        failed_breakout_count += 1

                        # Include the breakout candle's high and the next candle's high into highs_between
                        highs_between.append(breakout_candle_high)  # Re-add breakout candle's high
                        highs_between.append(next_high)  # Include next candle's high

                        # Update breakout_level to new max_high
                        max_high = max(highs_between)
                        breakout_level = max_high

                        if failed_breakout_count == 2:
                            break
                        else:
                            # Skip to the candle after the next one
                            i += 2
                            continue
                elif low < failure_point:
                    break
                else:
                    i += 1  # Move to the next candle

        elif signal == "shooting_star":
            shooting_star_low = self.df_daily.loc[self.row_index, 'l']
            breakout_level = shooting_star_low
            failure_point = self.df_daily.loc[self.row_index, 'h']
            lows_between = [shooting_star_low]
            failed_breakout_count = 0
            i = 0
            while i < len(relevant_hourly) - 1:
                row = relevant_hourly.iloc[i]
                close = row['c']
                high = row['h']
                low = row['l']

                lows_between.append(low)

                if close < breakout_level:
                    # Remove the breakout candle's low from lows_between temporarily
                    breakout_candle_low = lows_between.pop()

                    if i + 1 < len(relevant_hourly):
                        next_row = relevant_hourly.iloc[i + 1]
                        next_high = next_row['h']
                        next_low = next_row['l']
                    else:
                        break  # No more data

                    min_low = min(lows_between) if lows_between else breakout_level

                    if next_high < min_low:
                        # Valid breakout
                        entry_time = row['time']
                        entry_price = min_low
                        entry = Entry(
                            instrument=self.instrument,
                            signal=signal,
                            entry_type='GWSS',
                            price=entry_price,
                            time=entry_time,
                            row_index=self.row_index
                        )
                        entries.append(entry)
                        break
                    else:
                        # Failed breakout
                        failed_breakout_count += 1

                        # Include the breakout candle's low and the next candle's low into lows_between
                        lows_between.append(breakout_candle_low)  # Re-add breakout candle's low
                        lows_between.append(next_low)  # Include next candle's low

                        # Update breakout_level to new min_low
                        min_low = min(lows_between)
                        breakout_level = min_low

                        if failed_breakout_count == 2:
                            break
                        else:
                            # Skip to the candle after the next one
                            i += 2
                            continue
                elif high > failure_point:
                    break
                else:
                    i += 1  # Move to the next candle

        return entries
    
    def get_pip_value(self, instrument): 
        if "JPY" in instrument:
            return 0.01
        else:
            return 0.0001

    def calculate_stop_loss(self, entry):
        pip_value = self.get_pip_value(self.instrument)
        min_atr = self.df_daily.loc[self.row_index, 'atr'] * 0.4
        entry_price = entry.price

        if entry.entry_type == "PDH":
            stop_loss = self.df_daily.loc[entry.row_index, 'h'] + pip_value
            return stop_loss

        elif entry.entry_type == "PDL":
            stop_loss = self.df_daily.loc[entry.row_index, 'l'] - pip_value
            return stop_loss

        elif entry.entry_type in ["GWHMR", "GWSS"]:
            if entry.signal == "hammer":
                entry_time = entry.time
                failure_point = self.df_daily.loc[self.row_index, 'l']
                stop_loss = failure_point - pip_value

                relevant_pivots = self.zigzag_df[
                    (self.zigzag_df['time'] < entry_time) &
                    (self.zigzag_df['price'] < entry.price) &
                    (self.zigzag_df['price'] > failure_point) &
                    (self.zigzag_df['type'] == 'l')
                ].sort_values('time', ascending=False)

                if not relevant_pivots.empty:
                    for _, pivot in relevant_pivots.iterrows():
                        pivot_time = pivot['time']
                        pivot_price = pivot['price']

                        future_candles = self.df_hourly[
                            (self.df_hourly['time'] > pivot_time) & 
                            (self.df_hourly['time'] < entry_time)
                        ]
                        future_lows = future_candles['l'].min() if not future_candles.empty else None

                        if future_lows is None or (future_lows > pivot_price):
                            temp_stop_loss = pivot_price - pip_value
                            stop_loss_value = entry_price - temp_stop_loss
                            if stop_loss_value >= min_atr:
                                stop_loss = temp_stop_loss
                                break
                            else:
                                continue
                        else: 
                            continue

                return stop_loss

            elif entry.signal == "shooting_star":
                entry_time = entry.time
                failure_point = self.df_daily.loc[self.row_index, 'h']
                stop_loss = failure_point + pip_value

                relevant_pivots = self.zigzag_df[
                    (self.zigzag_df['time'] < entry_time) &
                    (self.zigzag_df['price'] > entry.price) &
                    (self.zigzag_df['price'] < failure_point) &
                    (self.zigzag_df['type'] == 'h')
                ].sort_values('time', ascending=False)

                if not relevant_pivots.empty:
                    for _, pivot in relevant_pivots.iterrows():
                        pivot_time = pivot['time']
                        pivot_price = pivot['price']

                        future_candles = self.df_hourly[
                            (self.df_hourly['time'] > pivot_time) & 
                            (self.df_hourly['time'] < entry_time)
                        ]
                        future_highs = future_candles['h'].max() if not future_candles.empty else None

                        if future_highs is None or (future_highs < pivot_price):
                            temp_stop_loss = pivot_price + pip_value
                            stop_loss_value = temp_stop_loss - entry_price
                            if stop_loss_value >= min_atr:
                                stop_loss = temp_stop_loss
                                break
                            else:
                                continue
                        else: 
                            continue

                return stop_loss



    def calculate_take_profit(self, entry, daily_zigzag):
        entry_price = entry.price
        entry_time = self.df_daily.loc[self.row_index, 'time']
        depth = 3
        confirmation_index = self.row_index - depth
        if confirmation_index < 0:
            return None

        adjusted_entry_time = self.df_daily.loc[confirmation_index, 'time']

        # Get all pivots before the entry time
        valid_pivots = daily_zigzag[daily_zigzag['time'] < adjusted_entry_time]

        if entry.signal == 'hammer':
            pivot_highs = valid_pivots[
                (valid_pivots['type'] == 'h') &
                (valid_pivots['price'] > entry_price)
            ]
            if not pivot_highs.empty:
                for idx, pivot in pivot_highs.sort_values('time', ascending=False).iterrows():
                    pivot_time = pivot['time']
                    pivot_price = pivot['price']
                    intermediate_highs = self.df_daily[
                        (self.df_daily['time'] > pivot_time) &
                        (self.df_daily['time'] <= entry_time)
                    ]['h']
                    if (intermediate_highs < pivot_price).all():
                        tp_level = pivot_price
                        return tp_level
                    else:
                        continue
        elif entry.signal == 'shooting_star':
            pivot_lows = valid_pivots[
                (valid_pivots['type'] == 'l') &
                (valid_pivots['price'] < entry_price)
            ]
            if not pivot_lows.empty:
                for idx, pivot in pivot_lows.sort_values('time', ascending=False).iterrows():
                    pivot_time = pivot['time']
                    pivot_price = pivot['price']
                    intermediate_lows = self.df_daily[
                        (self.df_daily['time'] > pivot_time) &
                        (self.df_daily['time'] <= entry_time)
                    ]['l']
                    if (intermediate_lows > pivot_price).all():
                        tp_level = pivot_price
                        return tp_level
                    else:
                        continue
        return None
    
    

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
    final_entries = []
    for entry in entries:
        if entry.signal in ['bull_eng', 'bear_eng']:
            handler = EngulfingHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        elif entry.signal in ['hammer', 'shooting_star']:
            handler = HammerShootingStarHandler(df_daily, df_hourly, entry.row_index, zigzag_df, instrument)
        else:
            continue

        # Calculate stop loss and take profit
        entry.stop_loss = handler.calculate_stop_loss(entry)
        entry.take_profit = handler.calculate_take_profit(entry, daily_zigzag)

        # Ensure stop loss and take profit are set
        if entry.stop_loss and entry.take_profit:
            # Calculate risk/reward ratio
            risk = abs(entry.price - entry.stop_loss)
            reward = abs(entry.take_profit - entry.price)
            rr_ratio = reward / risk if risk > 0 else 0

            print(f"Entry: {entry.entry_type}, date {entry.time} | Risk: {risk} | Reward: {reward} | R/R Ratio: {rr_ratio}")

            # Append to final list only if R/R ratio is at least 1.5
            if rr_ratio >= 1.5:
                final_entries.append(entry)
    
    # Replace entries with final list that meets R/R condition
    entries[:] = final_entries

def extract_instrument_from_filename(filename):
    """Extracts the instrument name from the filename."""
    return filename.split('_')[0] + '_' + filename.split('_')[1]

def main():
    # Base path where your data files are stored
    base_path = r"C:\Users\grave\OneDrive\Coding\PAC\fxdata"
    #base_path = r"/Users/koengraveland/PAC/fxdata"
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
        zigzag_file_path = r"C:\Users\grave\OneDrive\Coding\PAC\zigzag.xlsx"
        #zigzag_file_path = r"/Users/koengraveland/PAC/zigzag.xlsx"
        zigzag_df = pd.read_excel(zigzag_file_path)
        zigzag_df['time'] = pd.to_datetime(zigzag_df['time'])
        print(f"Loaded zigzag file: {zigzag_file_path}")

        # Process signals and calculate entries
        entries = process_signals(df_daily, df_hourly, zigzag_df, instrument)

        # Calculate stop loss and take profit
        daily_zigzag = calculate_zigzag_daily(df_daily, depth=3)
        calculate_sl_tp(entries, df_daily, df_hourly, zigzag_df, daily_zigzag, instrument)

        # Print entries
        for entry in entries:
            print(f"{entry.instrument}, {entry.signal}, {entry.entry_type}, {entry.time}, Price: {entry.price}, SL: {entry.stop_loss}, TP: {entry.take_profit}")

if __name__ == "__main__":
    main()
