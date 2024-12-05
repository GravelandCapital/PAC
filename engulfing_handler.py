from entry import Entry
import pandas as pd

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
            best_entry = self.select_best_entry(fib_entry)
            return best_entry
        lhpb_entry = self.calculate_lhpb_entry()
        if lhpb_entry:
            best_entry = self.select_best_entry(lhpb_entry)
        
    
    def select_best_entry(self, entry, entries):
        if entry.signal == "bull_eng":
            best_entry = max(entries, key=lambda entry: entry.price)
        elif entry.signal == "bear_eng":
            best_entry = min(entries, key=lambda entry: entry.price)
        return best_entry

    def calculate_fib_entry(self):
        fibo_level, half_level = self.calculate_fibo()
        if fibo_level is None:
            return None

        # Get naked levels
        hourly_data = self.get_hourly_data()
        naked_levels = self.calculate_naked_level(hourly_data)

        # Calculate ATR range
        atr_range = self.df_daily.loc[self.row_index, 'atr'] * 0.1

        # Define comparison function and difference calculation based on signal
        if self.signal == "bull_eng":
            compare = lambda level: level > half_level
        elif self.signal == "bear_eng":
            compare = lambda level: level < half_level
        else:
            return None

        fib_entries = []

        # Find matching naked levels
        for naked_level_info in naked_levels:
            naked_level = naked_level_info['price']
            level_time = naked_level_info['time']
            if fibo_level > naked_level: 
                difference = fibo_level - naked_level 
            elif fibo_level < naked_level: 
                difference = naked_level - fibo_level

            if 0 <= difference <= atr_range and compare(naked_level):
                order_time = self.df_daily.loc[self.row_index, 'time']
                entry = Entry(
                    instrument=self.instrument,
                    signal=self.signal,
                    entry_type='FIB',
                    price=naked_level,
                    order_time=order_time,
                    entry_candle_time=level_time,
                    row_index=self.row_index,
                    order_status="PENDING"
                )
                fib_entries.append(entry)
        
        if fib_entries:
            return fib_entries
        else: 
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
                            entry_candle_time = hourly_data.iloc[idx - 1]['time']
                        else:
                            break
                        future_candles = hourly_data.iloc[idx + 1:]
                        if (future_candles['l'] > last_high_pre_break).all() and last_high_pre_break < pivot_price and last_high_pre_break > half_level:
                            order_time = self.df_daily.loc[self.row_index, 'time']
                            entry = Entry(
                                instrument=self.instrument,
                                signal=self.signal,
                                entry_type='LHPB',
                                price=last_high_pre_break,
                                order_time= order_time,
                                entry_candle_time=entry_candle_time,
                                row_index=self.row_index,
                                order_status="PENDING"
                            )
                            entries.append(entry) 
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
                            entry_candle_time = hourly_data.iloc[idx - 1]['time']
                        else:
                            break
                        future_candles = hourly_data.iloc[idx + 1:]
                        if (future_candles['h'] < last_low_pre_break).all() and last_low_pre_break > pivot_price and last_low_pre_break < half_level:
                            order_time = self.df_daily.loc[self.row_index, 'time']
                            entry = Entry(
                                instrument=self.instrument,
                                signal=self.signal,
                                entry_type='LLPB',
                                price=last_low_pre_break,
                                order_time=order_time,
                                entry_candle_time=entry_candle_time,
                                row_index=self.row_index,
                                order_status="PENDING"
                            )
                            entries.append(entry)
                        else: 
                            break # Stop searching if the condition is not met

        # After collecting all entries, return the max for LHPB and min for LLPB
        if entries:
            return entries 

        else: 
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
                        level_time = prev_row["time"]
                        naked_levels.append({'price': previous_high, 'time': level_time})

            elif signal == "bear_eng":
                previous_low = prev_row["l"]
                current_close = curr_row["c"]

                if current_close < previous_low:
                    future_candles = hourly_data.iloc[i + 1:]
                    if (future_candles['h'] <= previous_low).all():
                        level_time = prev_row["time"]
                        naked_levels.append({'price': previous_low, 'time': level_time})
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

    def get_pip_value(self, instrument):
        """Returns the pip value for a given currency pair."""
        # If the pair involves JPY, pip is 0.01, else it's 0.0001
        if "JPY" in instrument:
            return 0.01
        else:
            return 0.0001

    def calculate_stop_loss(self, entry):
        sl_pivots = self.sl_pivots(entry.price)
        entry_candle_time = entry.order_time
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
            original_stop_loss = stop_loss

            for _, pivot in sl_pivots.iterrows():
                original_pivot_price = pivot['price']
                pivot_price = original_pivot_price

                hourly_data = self.df_hourly[
                    (self.df_hourly['time'] >= pivot['time']) & 
                    (self.df_hourly['time'] <= end_time) &
                    (self.df_hourly['time'] != entry_candle_time) 
                ]

                for idx in range(len(hourly_data)):
                    row = hourly_data.iloc[idx]

                    if row['h'] > signal_high:
                        break

                    if row['h'] > pivot_price and row['c'] <= pivot_price:
                        pivot_price = row['h']

                    if row['c'] > pivot_price:
                        breakout_low = None
                        breakout_low = row['l']
                        future_candles = hourly_data.iloc[idx + 1:]
                        future_min_low = future_candles['l'].min()

                        if future_min_low > breakout_low:
                            temp_stop_loss = breakout_low
                            stop_loss_value = entry_price - temp_stop_loss

                            if temp_stop_loss < original_pivot_price and stop_loss_value >= min_atr:
                                temp_stop_loss = breakout_low - pip_value
                                stop_loss_list.append(temp_stop_loss)
                                break
                            else:
                                break

            if stop_loss_list:
                max_stop_loss = max(stop_loss_list)
                return max_stop_loss, max_stop_loss
            else:
                return stop_loss, stop_loss

        elif self.signal == "bear_eng":
            failure_point = self.df_daily.loc[self.row_index, 'h']
            signal_low = self.df_daily.loc[self.row_index, 'l']
            stop_loss = failure_point
            original_stop_loss = stop_loss

            for _, pivot in sl_pivots.iterrows():
                original_pivot_price = pivot['price']
                pivot_price = original_pivot_price

                hourly_data = self.df_hourly[
                    (self.df_hourly['time'] >= pivot['time']) & (self.df_hourly['time'] <= end_time)
                ]

                for idx in range(len(hourly_data)):
                    row = hourly_data.iloc[idx]

                    if row['l'] < signal_low:
                        break

                    if row['l'] < pivot_price and row['c'] >= pivot_price:
                        pivot_price = row['l']

                    if row['c'] < pivot_price:
                        breakout_high = None 
                        breakout_high = row['h']
                        future_candles = hourly_data.iloc[idx + 1:]
                        future_max_high = future_candles['h'].max()

                        if future_max_high < breakout_high:
                            temp_stop_loss = breakout_high
                            stop_loss_value = temp_stop_loss - entry_price

                            if temp_stop_loss > original_pivot_price and stop_loss_value >= min_atr:
                                temp_stop_loss = breakout_high + pip_value
                                stop_loss_list.append(temp_stop_loss)
                                break
                            else:
                                break

            if stop_loss_list:
                min_stop_loss = min(stop_loss_list)
                return min_stop_loss, min_stop_loss
            else:
                stop_loss = stop_loss + pip_value
                return stop_loss, stop_loss


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
