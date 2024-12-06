import pandas as pd 
from entry import Entry

class HammerShootingStarHandler:
    """Handler for Hammer and Shooting Star signals."""
    def __init__(self, df_daily, df_hourly, row_index, zigzag_df, instrument, daily_zigzag):
        self.df_daily = df_daily
        self.df_hourly = df_hourly
        self.row_index = row_index
        self.signal = df_daily.loc[row_index, 'signal']
        self.zigzag_df = zigzag_df
        self.instrument = instrument
        self.daily_zigzag = daily_zigzag

    def calculate_entries(self):
        entries = []
        pdh_pdl_entries = self.calculate_pdh_pdl_entry()
        entries.extend(pdh_pdl_entries)
        gowith_entries = self.calculate_gowith_entry()
        entries.extend(gowith_entries)

        return entries 

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
                entry_candle_time = row['time']
                next_candle_close = hourly_data.iloc[i + 1]['c']
                future_candles = hourly_data.iloc[i + 2:]

                if high < pdl and next_candle_close > high and (future_candles['l'] > high).all():
                    order_time = self.df_daily.loc[self.row_index, 'time']
                    entry_price = high
                    entry = Entry(
                        instrument = self.instrument,
                        signal=signal,
                        entry_type='PDL',
                        price=entry_price,
                        entry_candle_time=entry_candle_time,
                        order_time=order_time,
                        row_index=self.row_index,
                        order_status="PENDING"
                    )
                    entries.append(entry)
        elif signal == "shooting_star":
            for i in range(len(hourly_data) - 1):
                row = hourly_data.iloc[i]
                low = row['l']
                entry_candle_time = row['time']
                next_candle_close = hourly_data.iloc[i + 1]['c']
                future_candles = hourly_data.iloc[i + 2:]

                if low > pdh and next_candle_close < low and (future_candles['h'] < low).all():
                    entry_time = self.df_daily.loc[self.row_index, 'time']
                    entry_price = low
                    entry = Entry(
                        instrument=self.instrument,
                        signal=signal,
                        entry_type='PDH',
                        price=entry_price,
                        order_time=entry_time,
                        entry_candle_time=entry_candle_time,
                        row_index=self.row_index,
                        order_status="PENDING"
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
                        order_candle = relevant_hourly.iloc[i + 2]
                    else:
                        break  # No more data

                    max_high = max(highs_between) if highs_between else breakout_level

                    if next_low > max_high:
                        # Valid breakout
                        entry_time = order_candle['time']
                        entry_price = max_high
                        entry = Entry(
                            instrument=self.instrument,
                            signal=signal,
                            entry_type='GWHMR',
                            price=entry_price,
                            entry_candle_time = entry_time,
                            order_time=entry_time,
                            row_index=self.row_index,
                            order_status="PENDING"
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
                        order_candle = relevant_hourly.iloc[i + 2]
                    else:
                        break  # No more data

                    min_low = min(lows_between) if lows_between else breakout_level

                    if next_high < min_low:
                        # Valid breakout
                        entry_time = order_candle['time']
                        entry_price = min_low
                        entry = Entry(
                            instrument=self.instrument,
                            signal=signal,
                            entry_type='GWSS',
                            price=entry_price,
                            entry_candle_time=entry_time,
                            order_time=entry_time,
                            row_index=self.row_index,
                            order_status="PENDING"
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
        entry_candle_time = entry.entry_candle_time
        

        stop_loss_list = []

        if entry.entry_type == "PDH":
            stop_loss = self.df_daily.loc[entry.row_index, 'h'] + pip_value
            stop_loss_time = self.df_daily.loc[entry.row_index, 'time']
            origignal_stop_loss = stop_loss
            entry.original_stop_loss = origignal_stop_loss
            stop_loss_list.append(stop_loss)
            return stop_loss_list

        elif entry.entry_type == "PDL":
            stop_loss = self.df_daily.loc[entry.row_index, 'l'] - pip_value
            stop_loss_time = self.df_daily.loc[entry.row_index, 'time']
            origignal_stop_loss = stop_loss
            entry.original_stop_loss = origignal_stop_loss
            stop_loss_list.append(stop_loss)
            return stop_loss_list

        elif entry.entry_type in ["GWHMR", "GWSS"]:
            if entry.signal == "hammer":
                entry_time = entry.order_time
                failure_point = self.df_daily.loc[self.row_index, 'l']
                stop_loss = failure_point - pip_value
                original_stop_loss = stop_loss
                entry.original_stop_loss = original_stop_loss
                stop_loss_list.append(original_stop_loss)

                if self.row_index > 0:
                    previous_day_time = self.df_daily.loc[self.row_index - 1, 'time']
                else: 
                    previous_day_time = None

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
                            stop_loss_time = previous_day_time
                            if stop_loss_value >= min_atr:
                                stop_loss_list.append(temp_stop_loss)
                            else:
                                continue
                        else: 
                            continue

                return stop_loss_list

            elif entry.signal == "shooting_star":
                entry_time = entry.order_time
                failure_point = self.df_daily.loc[self.row_index, 'h']
                stop_loss = failure_point + pip_value
                original_stop_loss = stop_loss
                entry.original_stop_loss = original_stop_loss
                stop_loss_list.append(original_stop_loss)

                if self.row_index > 0:
                    previous_day_time = self.df_daily.loc[self.row_index - 1, 'time']
                else:
                    previous_day_time = None

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
                            stop_loss_time = previous_day_time
                            if stop_loss_value >= min_atr:
                                stop_loss_list.append(temp_stop_loss)
                            else:
                                continue
                        else: 
                            continue

                return stop_loss_list

    def calculate_take_profit(self, entry):
        entry_price = entry.price
        entry_time = self.df_daily.loc[self.row_index, 'time']
        depth = 3
        confirmation_index = self.row_index - depth
        if confirmation_index < 0:
            return None

        adjusted_entry_time = self.df_daily.loc[confirmation_index, 'time']

        # Get all pivots before the entry time
        valid_pivots = self.daily_zigzag[self.daily_zigzag['time'] < adjusted_entry_time]

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
            atr = self.df_daily.loc[self.row_index, 'atr']
            tp_level = entry_price + (atr * 2)
            return tp_level
            

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
        
            atr = self.df_daily.loc[self.row_index, 'atr']
            tp_level = entry_price - (atr * 2)
            return tp_level


    def generate_valid_combinations(self, entries):
        valid_combinations = []
        take_profit = self.calculate_take_profit(entries[0])

        for entry in entries:
            stop_losses = self.calculate_stop_loss(entry)
            for sl in stop_losses:
                risk = abs(entry.price - sl)
                reward = abs(take_profit - entry.price)
                rr_ratio = reward / risk if risk > 0 else 0
                if rr_ratio >= 1.5:
                    valid_combinations.append({
                        'entry': entry,
                        'stop_loss': sl,
                        'take_profit': take_profit,
                        'rr_ratio': rr_ratio
                    })
        return valid_combinations
    
    def select_best_trade(self, valid_combinations):
        if not valid_combinations:
            return None
        best_trade = max(valid_combinations, key=lambda x: x['rr_ratio'])
        return best_trade