import pandas as pd 
from entry import Entry 

class TradeManager:
    def __init__(self, df_hourly, entry, zigzag_df, depth):
        self.df_hourly = df_hourly
        self.entry = entry
        self.zigzag_df = zigzag_df
        self.depth = depth
        self.pip_value = self.get_pip_value(entry.instrument)


    def get_pip_value(self, instrument):
        """Returns the pip value for a given instrument."""
        instrument = instrument.upper().strip() # Ensure instrument is always in uppercase
        if "JPY" in instrument or "XAG" in instrument:
            pip_value = 0.01  # Pip value for JPY pairs and XAG/USD
        elif "XAU" in instrument:
            pip_value = 0.5  # Corrected pip value for gold (XAU/USD)
        else:
            pip_value = 0.0001  # Default pip value for most instruments
        return pip_value


    def check_order_execution(self):
        start_time = self.entry.order_time 
        end_time = start_time + pd.Timedelta(hours=24)
        hourly_data = self.df_hourly[(self.df_hourly['time'] > start_time) & (self.df_hourly['time'] <= end_time)]

        if self.entry.signal in ['bull_eng', 'hammer']:
            condition = hourly_data['l'] <= self.entry.price
        elif self.entry.signal in ['bear_eng', 'shooting_star']:
            condition = hourly_data['h'] >= self.entry.price
        else: 
            return False 

        if condition.any():
            fill_time = hourly_data[condition].iloc[0]['time']
            self.entry.filled_time = fill_time
            self.entry.order_status = "FILLED"
            return True
        else: 
            self.entry.order_status = "CANCELLED"
            return False

    def manage_trade(self):
        start_time = self.entry.filled_time
        if not start_time:
            return None  # Trade not filled yet

        # Subset hourly data after trade is filled
        hourly_data = self.df_hourly[self.df_hourly['time'] >= start_time].reset_index(drop=True)

        stop_loss = self.entry.stop_loss

        direction = 'bullish' if self.entry.signal in ['bull_eng', 'hammer'] else 'bearish'


        pivot = self.pre_fill_pivot(start_time, direction)

        if pivot is not None:
            pivot_price = pivot['price']
            pivot_time = pivot['time']

        first_pivot = self.find_first_pivot_after_entry(start_time, direction)

        if first_pivot is not None: 
            exit_pivot_price = first_pivot['price']
            exit_pivot_time = first_pivot['time']
        
        else: 
            exit_pivot_price = None
        
        # Iterate over each hourly bar
        for idx in range(len(hourly_data)):
            current_row = hourly_data.iloc[idx]
            current_time = current_row['time']
            current_open = current_row['o']
            current_close = current_row['c']
            current_high = current_row['h']
            current_low = current_row['l']
            self.current_open = current_open  # Set current_open for use in get_latest_pivot


            #  Logic for exit on close of the exit pivot 
            if exit_pivot_price is not None:
                confirmation_time = exit_pivot_time + pd.Timedelta(hours=4)
                if direction == 'bullish':
                    if current_low < exit_pivot_price and current_close > exit_pivot_price and current_time >= confirmation_time:
                        exit_pivot_price = current_low
                    elif current_close < exit_pivot_price and current_time >= confirmation_time:
                        exit_time = current_time 
                        exit_price = current_close 
                        return {'exit_time': current_time, 'exit_price': current_close}
                elif direction == 'bearish':
                    if current_high > exit_pivot_price and current_close < exit_pivot_price and current_time >= confirmation_time:
                        exit_pivot_price = current_high
                    elif current_close > exit_pivot_price and current_time >= confirmation_time:
                        exit_time = current_time
                        exit_price = current_close
                        return {'exit_time': current_time, 'exit_price': current_close}

                if direction == 'bullish': 
                    if pivot_price is not None: 
                        if current_high > pivot_price and current_close < pivot_price:
                            pivot_price = current_high
                elif direction == 'bearish':
                    if pivot_price is not None: 
                        if current_low < pivot_price and current_close > pivot_price:
                            pivot_price = current_low
                        

                # Logic for if price closes through a pivot 
                pivot_crossed = False 
                if pivot_price is not None: 
                    if direction == 'bullish' and current_close > pivot_price:
                        pivot_crossed = True
                    elif direction == 'bearish' and current_close < pivot_price:
                        pivot_crossed = True
                
                if pivot_crossed:
                    # Adjust stop_loss 
                    if direction == 'bullish': 
                        stop_loss = current_low - self.pip_value
                    elif direction == 'bearish':
                        stop_loss = current_high + self.pip_value
                    self.entry.stop_loss = stop_loss

                    # Call get latest pivot because pivot was crossed 
                    pivot = self.get_latest_pivot(current_time, direction)
                    if pivot is not None:
                        pivot_price = pivot['price']
                        pivot_time = pivot['time']
                    else:
                        pivot_price = None

            # Detect if a new pivot has formed closer to the current entry price
            new_pivot = self.check_for_new_pivot(pivot_price, start_time, current_time, direction)
            if new_pivot is not None and new_pivot['price'] != pivot_price:
                pivot = new_pivot
                pivot_price = pivot['price']
                pivot_time = pivot['time']

            # Check for exit condition
            if self.check_exit_condition(current_row, stop_loss):
                exit_time = current_time
                exit_price = stop_loss
                return {'exit_time': exit_time, 'exit_price': exit_price}
                
    def get_latest_pivot(self, current_time, direction):
        time_interval = pd.Timedelta(hours=1)  # Adjust if your data has a different time interval
        depth_timedelta = self.depth * time_interval

        # Calculate confirmation time for each pivot
        confirmed_pivots = self.zigzag_df.copy()
        confirmed_pivots['confirmation_time'] = confirmed_pivots['time'] + depth_timedelta

        # Only consider pivots confirmed before or at the current time
        confirmed_pivots = confirmed_pivots[confirmed_pivots['confirmation_time'] <= current_time]

        if direction == 'bullish':
            # Get pivot highs
            pivot_highs = confirmed_pivots[confirmed_pivots['type'] == 'h']
            # Sort by confirmation time descending to get the most recent confirmed pivot high
            pivot_highs = pivot_highs.sort_values('confirmation_time', ascending=False)

            for _, pivot in pivot_highs.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']

                # Get hourly data 
                hourly_data = self.df_hourly[(self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] <= current_time)]

                # Check if all closes and highs remain below pivot price 
                if (hourly_data['c'] < pivot_price).all() and hourly_data['h'].max() < pivot_price:
                    return pivot
                
                # If a high is above pivot price but close stays below, update the pivot price
                elif (hourly_data['c'] < pivot_price).all():
                    wicked_bars = hourly_data[(hourly_data['h'] > pivot_price) & (hourly_data['c'] < pivot_price)]
                    for _, wicked_bars in wicked_bars.iterrows():
                        if wicked_bars['h'] > pivot_price:
                            pivot_price = wicked_bars['h']
                    pivot['price'] = pivot_price
                    return pivot

        elif direction == 'bearish':
            # Get pivot lows
            pivot_lows = confirmed_pivots[confirmed_pivots['type'] == 'l']
            # Sort by confirmation time descending to get the most recent confirmed pivot low
            pivot_lows = pivot_lows.sort_values('confirmation_time', ascending=False)

            for _, pivot in pivot_lows.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']
                
                # Get hourly data 
                hourly_data = self.df_hourly[(self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] <= current_time)]

                # Check if all closes and lows remain above pivot price 
                if (hourly_data['c'] > pivot_price).all() and hourly_data['l'].min() > pivot_price:
                    return pivot
            
                # If a low is below pivot price but close stays above, update the pivot price
                elif (hourly_data['c'] > pivot_price).all(): 
                    wicked_bars = hourly_data[(hourly_data['l'] < pivot_price) & (hourly_data['c'] > pivot_price)]
                    for _, wicked_bars in wicked_bars.iterrows():
                        if wicked_bars['l'] < pivot_price:
                            pivot_price = wicked_bars['l']
                    pivot['price'] = pivot_price
                    return pivot
                
    def pre_fill_pivot(self, pre_fill_time, direction): 
        time_interval = pd.Timedelta(hours=1)
        depth_timedelta = 3 * time_interval

        pre_fill_time = pre_fill_time - pd.Timedelta(hours=1)

        confirmed_pivots = self.zigzag_df.copy()

        confirmed_pivots['confirmation_time'] = confirmed_pivots['time'] + depth_timedelta

        confirmed_pivots = confirmed_pivots[confirmed_pivots['confirmation_time'] <= pre_fill_time]

        if direction == 'bullish':
            pivot_highs = confirmed_pivots[confirmed_pivots['type'] == 'h']
            pivot_highs = pivot_highs.sort_values('confirmation_time', ascending=False)

            for _, pivot in pivot_highs.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']

                hourly_data = self.df_hourly[(self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] <= pre_fill_time)]

                if (hourly_data['c'] < pivot_price).all() and hourly_data['h'].max() < pivot_price:
                    return pivot
                elif (hourly_data['c'] < pivot_price).all():
                    wicked_bars = hourly_data[(hourly_data['h'] > pivot_price) & (hourly_data['c'] < pivot_price)]
                    for _, wicked_bars in wicked_bars.iterrows():
                        if wicked_bars['h'] > pivot_price:
                            pivot_price = wicked_bars['h']
                    pivot['price'] = pivot_price
                    return pivot
                
        elif direction == 'bearish':
            pivot_lows = confirmed_pivots[confirmed_pivots['type'] == 'l']
            pivot_lows = pivot_lows.sort_values('confirmation_time', ascending=False)

            for _, pivot in pivot_lows.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']
                
                hourly_data = self.df_hourly[(self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] <= pre_fill_time)]

                if (hourly_data['c'] > pivot_price).all() and hourly_data['l'].min() > pivot_price:
                    return pivot
            
                elif (hourly_data['c'] > pivot_price).all(): 
                    wicked_bars = hourly_data[(hourly_data['l'] < pivot_price) & (hourly_data['c'] > pivot_price)]
                    for _, wicked_bars in wicked_bars.iterrows():
                        if wicked_bars['l'] < pivot_price:
                            pivot_price = wicked_bars['l']
                    pivot['price'] = pivot_price
                    return pivot

    def check_for_new_pivot(self, last_pivot_price, filled_time, current_time, direction): 
        depth = 4
        depth_timedelta = depth * pd.Timedelta(hours=1)
        confirmed_pivots = self.zigzag_df.copy()
        confirmed_pivots['confirmation_time'] = confirmed_pivots['time'] + depth_timedelta
        filled_time = filled_time - pd.Timedelta(hours=4)

        confirmed_pivots = confirmed_pivots [(confirmed_pivots['confirmation_time'] <= current_time) & (confirmed_pivots['time'] > filled_time)]
       
        if direction == 'bullish':
            new_pivots = confirmed_pivots[(confirmed_pivots['type'] == 'h') & 
            (confirmed_pivots['time'] > filled_time) & 
            (confirmed_pivots['price'] < last_pivot_price)]

            if new_pivots.empty:
                return None

            newest_pivot = new_pivots.sort_values('price', ascending=False).iloc[0]

            hourly_data = self.df_hourly[(self.df_hourly['time'] > newest_pivot['time']) & (self.df_hourly['time'] <= current_time)]

            max_high = hourly_data['h'].max()

            if max_high < newest_pivot['price']:
                return newest_pivot
            else: 
                return None
        
        elif direction == 'bearish':
            new_pivots = confirmed_pivots[(confirmed_pivots['type'] == 'l') & 
            (confirmed_pivots['time'] > filled_time) & 
            (confirmed_pivots['price'] > last_pivot_price)]

            if new_pivots.empty:
                return None

            newest_pivot = new_pivots.sort_values('price', ascending=True).iloc[0]

            hourly_data = self.df_hourly[(self.df_hourly['time'] > newest_pivot['time']) & (self.df_hourly['time'] <= current_time)]

            min_low = hourly_data['l'].min()

            if min_low > newest_pivot['price']:
                return newest_pivot
            else:
                return None
                
    def find_first_pivot_after_entry(self, start_time, direction):
        depth = 4
        time_interval = pd.Timedelta(hours=1)
        depth_timedelta = depth * time_interval
        confirmed_pivots = self.zigzag_df.copy()
        confirmed_pivots['confirmation_time'] = confirmed_pivots['time'] + depth_timedelta

        if direction == 'bullish': 
            pivots_after_entry = confirmed_pivots[(confirmed_pivots['type'] == 'l') & 
                                                  (confirmed_pivots['time'] >= start_time)
                                                  ]

            if pivots_after_entry.empty:
                return None 

            pivots_after_entry.reset_index(drop=True, inplace=True)
            first_pivot_after_entry = pivots_after_entry.loc[0]

            return first_pivot_after_entry

        elif direction == 'bearish':
            pivots_after_entry = confirmed_pivots[(confirmed_pivots['type'] == 'h') & 
                                                  (confirmed_pivots['time'] >= start_time)
                                                  ]
            
            if pivots_after_entry.empty:
                return None

            pivots_after_entry.reset_index(drop=True, inplace=True)
            first_pivot_after_entry = pivots_after_entry.loc[0]

            return first_pivot_after_entry

    def check_exit_condition(self, current_row, stop_loss):
        # Determine if the trade should be exited based on stop loss being hit
        stop_loss = self.entry.stop_loss 

        if self.entry.signal in ['bull_eng', 'hammer']:
            exit_condition = current_row['l'] <= stop_loss
            return exit_condition
        elif self.entry.signal in ['bear_eng', 'shooting_star']:
            exit_condition = current_row['h'] >= stop_loss
            return exit_condition
        return False

