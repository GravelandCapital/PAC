import pandas as pd 
from entry import Entry 

class TradeManager:
    def __init__(self, df_hourly, entry, zigzag_df):
        print(f"Initializing TradeManager for instrument {entry.instrument}")
        self.df_hourly = df_hourly
        self.entry = entry
        self.zigzag_df = zigzag_df  # Added zigzag_df to the constructor
        self.pip_value = self.get_pip_value(entry.instrument)
        print(f"Set pip_value to {self.pip_value}")
        self.current_pivot = None  # Initialize current_pivot to None
    
    def get_pip_value(self, instrument): 
        pip_value = 0.01 if "JPY" in instrument else 0.0001
        print(f"Calculated pip value for {instrument}: {pip_value}")
        return pip_value

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

            if self.entry.signal in ['bull_eng', 'hammer']:
                direction = 'bullish'

                # If no pivot is currently being monitored, find one
                if self.current_pivot is None:
                    pivot = self.get_latest_pivot(current_time, direction)
                    if pivot is not None:
                        self.current_pivot = {'pivot_price': pivot['price'], 'pivot_time': pivot['time']}
                        print(f"Monitoring new pivot high: {self.current_pivot['pivot_price']} at {self.current_pivot['pivot_time']}")
                else:
                    pivot_price = self.current_pivot['pivot_price']
                    pivot_time = self.current_pivot['pivot_time']

                    # Check if current_close > pivot_price
                    if current_close > pivot_price:
                        # Adjust stop loss to the low of the breakout candle minus pip value
                        breakout_low = current_low
                        stop_loss = breakout_low - self.pip_value
                        self.entry.stop_loss = stop_loss  # Update the entry's stop loss
                        print(f"Stop loss adjusted to {stop_loss} at {current_time}")
                        # Reset current_pivot to None to find the next pivot
                        self.current_pivot = None
                    # Else if current_high > pivot_price and current_close < pivot_price
                    elif current_high > pivot_price and current_close < pivot_price:
                        # Adjust pivot_price to current_high
                        self.current_pivot['pivot_price'] = current_high
                        print(f"Adjusted pivot price to {current_high} at {current_time}")
                    else:
                        # Continue monitoring the same pivot
                        pass

            elif self.entry.signal in ['bear_eng', 'shooting_star']:
                direction = 'bearish'

                # If no pivot is currently being monitored, find one
                if self.current_pivot is None:
                    pivot = self.get_latest_pivot(current_time, direction)
                    if pivot is not None:
                        self.current_pivot = {'pivot_price': pivot['price'], 'pivot_time': pivot['time']}
                        print(f"Monitoring new pivot low: {self.current_pivot['pivot_price']} at {self.current_pivot['pivot_time']}")
                else:
                    pivot_price = self.current_pivot['pivot_price']
                    pivot_time = self.current_pivot['pivot_time']

                    # Check if current_close < pivot_price
                    if current_close < pivot_price:
                        # Adjust stop loss to the high of the breakout candle plus pip value
                        breakout_high = current_high
                        stop_loss = breakout_high + self.pip_value
                        self.entry.stop_loss = stop_loss  # Update the entry's stop loss
                        print(f"Stop loss adjusted to {stop_loss} at {current_time}")
                        # Reset current_pivot to None to find the next pivot
                        self.current_pivot = None
                    # Else if current_low < pivot_price and current_close > pivot_price
                    elif current_low < pivot_price and current_close > pivot_price:
                        # Adjust pivot_price to current_low
                        self.current_pivot['pivot_price'] = current_low
                        print(f"Adjusted pivot price to {current_low} at {current_time}")
                    else:
                        # Continue monitoring the same pivot
                        pass

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
        # Filter pivots from zigzag_df before current_time
        pivots_before_current = self.zigzag_df[self.zigzag_df['time'] < current_time]

        if direction == 'bullish':
            # Get pivot highs
            pivot_highs = pivots_before_current[pivots_before_current['type'] == 'h']
            # Sort by time descending to get the most recent pivot high
            pivot_highs = pivot_highs.sort_values('time', ascending=False)

            for _, pivot in pivot_highs.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']
                # Check if pivot price is greater than current open
                if pivot_price > self.current_open:
                    # Check if pivot price is greater than all highs between pivot_time and current_time (excluding current_time)
                    highs_between = self.df_hourly[
                        (self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] < current_time)
                    ]['h']
                    if (highs_between <= pivot_price).all():
                        return pivot

        elif direction == 'bearish':
            # Get pivot lows
            pivot_lows = pivots_before_current[pivots_before_current['type'] == 'l']
            # Sort by time descending to get the most recent pivot low
            pivot_lows = pivot_lows.sort_values('time', ascending=False)

            for _, pivot in pivot_lows.iterrows():
                pivot_time = pivot['time']
                pivot_price = pivot['price']
                # Check if pivot price is less than current open
                if pivot_price < self.current_open:
                    # Check if pivot price is less than all lows between pivot_time and current_time (excluding current_time)
                    lows_between = self.df_hourly[
                        (self.df_hourly['time'] > pivot_time) & (self.df_hourly['time'] < current_time)
                    ]['l']
                    if (lows_between >= pivot_price).all():
                        return pivot
        return None

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
