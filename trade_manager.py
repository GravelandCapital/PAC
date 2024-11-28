import pandas as pd 
from entry import Entry 

class TradeManager:
    def __init__(self, df_hourly, entry, zigzag_df, depth):
        print(f"Initializing TradeManager for instrument {entry.instrument}")
        self.df_hourly = df_hourly
        self.entry = entry
        self.zigzag_df = zigzag_df
        self.depth = depth
        self.pip_value = self.get_pip_value(entry.instrument)
        print(f"Set pip_value to {self.pip_value}")

    def get_pip_value(self, instrument):
        pip_value = 0.01 if "JPY" in instrument else 0.0001
        print(f"Calculated pip value for {instrument}: {pip_value}")
        return pip_value

    def check_order_execution(self):
        print("Checking order execution...")
        start_time = self.entry.order_time 
        end_time = start_time + pd.Timedelta(hours=24)
        hourly_data = self.df_hourly[(self.df_hourly['time'] > start_time) & (self.df_hourly['time'] <= end_time)]

        print(f"Order placed at {start_time}, checking for execution up to {end_time}")

        if self.entry.signal in ['bull_eng', 'hammer']:
            condition = hourly_data['l'] <= self.entry.price
        elif self.entry.signal in ['bear_eng', 'shooting_star']:
            condition = hourly_data['h'] >= self.entry.price
        else: 
            print("Invalid signal type.")
            return False 

        if condition.any():
            fill_time = hourly_data[condition].iloc[0]['time']
            self.entry.filled_time = fill_time
            self.entry.order_status = "FILLED"
            print(f"Order filled at {fill_time}")
            return True
        else: 
            self.entry.order_status = "CANCELLED"
            print("Order not filled within time frame; order cancelled.")
            return False

    def manage_trade(self):
        print("Starting trade management...")
        start_time = self.entry.filled_time
        if not start_time:
            print("Trade not filled yet.")
            return None  # Trade not filled yet

        # Subset hourly data after trade is filled
        hourly_data = self.df_hourly[self.df_hourly['time'] >= start_time].reset_index(drop=True)
        print(f"Managing trade from {start_time} onwards.")

        # Initialize variables
        stop_loss = self.entry.stop_loss
        print(f"Initial stop loss: {stop_loss}")

        direction = 'bullish' if self.entry.signal in ['bull_eng', 'hammer'] else 'bearish'

        first_pivot = self.find_first_pivot_after_entry(start_time, direction)

        if first_pivot is not None: 
            exit_pivot_price = first_pivot['price']
            exit_pivot_time = first_pivot['time']
            print(f"Exit pivot price: {exit_pivot_price} at {exit_pivot_time}")
        
        else: 
            print(f"No exit pivot found after entry at {start_time}.")
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
            print(f"\nProcessing candle at {current_time}: Open={current_open}, Close={current_close}, High={current_high}, Low={current_low}")

            if exit_pivot_price is not None:
                confirmation_time = exit_pivot_time + pd.Timedelta(hours=4)
                if direction == 'bullish':
                    if current_low < exit_pivot_price and current_close > exit_pivot_price and current_time >= confirmation_time:
                        exit_pivot_price = current_low
                        print(f"Updated exit pivot price to {exit_pivot_price} due to wicked pivot.")
                    elif current_close < exit_pivot_price and current_time >= confirmation_time:
                        exit_time = current_time 
                        exit_price = current_close 
                        print(f"Trade exited at {current_time} with exit price {current_close}")
                        return {'exit_time': current_time, 'exit_price': current_close}
                elif direction == 'bearish':
                    if current_high > exit_pivot_price and current_close < exit_pivot_price and current_time >= confirmation_time:
                        exit_pivot_price = current_high
                        print(f"Updated exit pivot price to {exit_pivot_price} due to wicked pivot.")
                    elif current_close > exit_pivot_price and current_time >= confirmation_time:
                        exit_time = current_time
                        exit_price = current_close
                        print(f"Trade exited at {current_time} with exit price {current_close}")
                        return {'exit_time': current_time, 'exit_price': current_close}

            if self.entry.signal in ['bull_eng', 'hammer']:
                direction = 'bullish'
                pivot = self.get_latest_pivot(current_time, direction)
                       
                if pivot is not None:
                    pivot_price = pivot['price']
                    print(f"Latest confirmed pivot high: {pivot_price} at {pivot['time']}")

                    # Check for wicked pivots, update pivot_price if necessary
                    if current_high > pivot_price and current_close < pivot_price: 
                        pivot_price = current_high
                        print(f"Updated pivot price to {pivot_price} due to wicked pivot.")

                    # Check if current close is greater than pivot_price
                    if current_close > pivot_price:
                        # Adjust stop loss to the low of the breakout candle minus pip value
                        breakout_low = current_low
                        stop_loss = breakout_low - self.pip_value
                        self.entry.stop_loss = stop_loss  # Update the entry's stop loss
                        print(f"Stop loss adjusted to {stop_loss} at {current_time}")

            elif self.entry.signal in ['bear_eng', 'shooting_star']:
                direction = 'bearish'
                pivot = self.get_latest_pivot(current_time, direction)

                if pivot is not None:
                    pivot_price = pivot['price']
                    print(f"Latest confirmed pivot low: {pivot_price} at {pivot['time']}")

                    # Check for wicked pivots, update pivot_price if necessary
                    if current_low < pivot_price and current_close > pivot_price:
                        pivot_price = current_low
                        print(f"Updated pivot price to {pivot_price} due to wicked pivot.")

                    # Check if current close is less than pivot_price
                    if current_close < pivot_price:
                        # Adjust stop loss to the high of the breakout candle plus pip value
                        breakout_high = current_high
                        stop_loss = breakout_high + self.pip_value
                        self.entry.stop_loss = stop_loss  # Update the entry's stop loss
                        print(f"Stop loss adjusted to {stop_loss} at {current_time}")

            # Check for exit condition
            if self.check_exit_condition(current_row, stop_loss):
                exit_time = current_time
                exit_price = stop_loss
                print(f"Trade exited at {exit_time} with exit price {exit_price}")
                return {'exit_time': exit_time, 'exit_price': exit_price}

        # If trade is still open after iterating through all data
        print("Trade still open after all data processed.")
        return None

    def get_latest_pivot(self, current_time, direction):
        time_interval = pd.Timedelta(hours=1)  # Adjust if your data has a different time interval
        depth_timedelta = self.depth * time_interval

        # Calculate confirmation time for each pivot
        confirmed_pivots = self.zigzag_df.copy()
        confirmed_pivots['confirmation_time'] = confirmed_pivots['time'] + depth_timedelta

        # Only consider pivots confirmed before or at the current time
        confirmed_pivots = confirmed_pivots[confirmed_pivots['confirmation_time'] <= current_time]

        if direction == 'bullish':
            # Get the most recent confirmed pivot high
            pivot_highs = confirmed_pivots[confirmed_pivots['type'] == 'h']
            if pivot_highs.empty:
                print("No pivot highs found.")
                return None
            pivot = pivot_highs.sort_values('confirmation_time', ascending=False).iloc[0]
            pivot_time = pivot['time']
            pivot_price = pivot['price']
            print(f"Initial pivot price: {pivot_price} at {pivot_time}")

            # Subset hourly data between pivot_time and current_time
            hourly_data = self.df_hourly[(self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] <= current_time)]

            # Initialize updated_pivot_price
            updated_pivot_price = pivot_price

            # Continue updating pivot_price until it no longer changes
            while True:
                # Find candles that wick above the pivot price but close below it
                wicking_candles = hourly_data[(hourly_data['h'] > updated_pivot_price) & (hourly_data['c'] < updated_pivot_price)]

                if wicking_candles.empty:
                    break  # No more wicks above the pivot price

                # Update pivot price to the maximum high of wicking candles
                new_pivot_price = wicking_candles['h'].max()
                if new_pivot_price == updated_pivot_price:
                    break  # Pivot price did not change, exit loop

                print(f"Updated pivot price from {updated_pivot_price} to {new_pivot_price} due to wicks.")
                updated_pivot_price = new_pivot_price

            # Check if any candle closes above the updated pivot price
            if (hourly_data['c'] >= updated_pivot_price).any():
                print("Pivot invalidated due to close above pivot price.")
                return None  # Pivot is invalidated

            # Update pivot price in the pivot data
            pivot['price'] = updated_pivot_price
            print(f"Final pivot price: {updated_pivot_price}")
            return pivot

        elif direction == 'bearish':
            # Similar logic for bearish scenario
            # Get the most recent confirmed pivot low
            pivot_lows = confirmed_pivots[confirmed_pivots['type'] == 'l']
            if pivot_lows.empty:
                print("No pivot lows found.")
                return None
            pivot = pivot_lows.sort_values('confirmation_time', ascending=False).iloc[0]
            pivot_time = pivot['time']
            pivot_price = pivot['price']
            print(f"Initial pivot price: {pivot_price} at {pivot_time}")

            # Subset hourly data between pivot_time and current_time
            hourly_data = self.df_hourly[(self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] <= current_time)]

            # Initialize updated_pivot_price
            updated_pivot_price = pivot_price

            # Continue updating pivot_price until it no longer changes
            while True:
                # Find candles that wick below the pivot price but close above it
                wicking_candles = hourly_data[(hourly_data['l'] < updated_pivot_price) & (hourly_data['c'] > updated_pivot_price)]

                if wicking_candles.empty:
                    break  # No more wicks below the pivot price

                # Update pivot price to the minimum low of wicking candles
                new_pivot_price = wicking_candles['l'].min()
                if new_pivot_price == updated_pivot_price:
                    break  # Pivot price did not change, exit loop

                print(f"Updated pivot price from {updated_pivot_price} to {new_pivot_price} due to wicks.")
                updated_pivot_price = new_pivot_price

            # Check if any candle closes below the updated pivot price
            if (hourly_data['c'] <= updated_pivot_price).any():
                print("Pivot invalidated due to close below pivot price.")
                return None  # Pivot is invalidated

            # Update pivot price in the pivot data
            pivot['price'] = updated_pivot_price
            print(f"Final pivot price: {updated_pivot_price}")
            return pivot

        else:
            print("Invalid direction.")
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

            print (f"First pivot after entry: {first_pivot_after_entry['price']} at {first_pivot_after_entry['time']}")
            return first_pivot_after_entry

        elif direction == 'bearish':
            pivots_after_entry = confirmed_pivots[(confirmed_pivots['type'] == 'h') & 
                                                  (confirmed_pivots['time'] >= start_time)
                                                  ]
            
            if pivots_after_entry.empty:
                return None

            pivots_after_entry.reset_index(drop=True, inplace=True)
            first_pivot_after_entry = pivots_after_entry.loc[0]

            print (f"First pivot after entry: {first_pivot_after_entry['price']} at {first_pivot_after_entry['time']}")
            return first_pivot_after_entry

    def check_exit_condition(self, current_row, stop_loss):
        print(f"Checking exit condition at time {current_row['time']} with stop loss {stop_loss}")
        # Determine if the trade should be exited based on stop loss being hit
        if self.entry.signal in ['bull_eng', 'hammer']:
            exit_condition = current_row['l'] <= stop_loss
            print(f"Bullish exit condition is {'met' if exit_condition else 'not met'}.")
            return exit_condition
        elif self.entry.signal in ['bear_eng', 'shooting_star']:
            exit_condition = current_row['h'] >= stop_loss
            print(f"Bearish exit condition is {'met' if exit_condition else 'not met'}.")
            return exit_condition
        print("Invalid signal type for exit condition.")
        return False
