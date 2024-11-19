import pandas as pd 
from entry import Entry 

class TradeManager:
    def __init__(self, df_hourly, entry):
        print(f"Initializing TradeManager for instrument {entry.instrument}")
        self.df_hourly = df_hourly
        self.entry = entry
        self.depth = 4  # Depth for pivot confirmation
        self.pip_value = self.get_pip_value(entry.instrument)
        print(f"Set depth to {self.depth}, pip_value to {self.pip_value}")

    def get_pip_value(self, instrument): 
        pip_value = 0.01 if "JPY" in instrument else 0.0001
        print(f"Calculated pip value for {instrument}: {pip_value}")
        return pip_value

    def check_order_execution(self):
        print("Checking order execution...")
        start_time = self.entry.time 
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
        pivots = []  # To store detected pivots

        # Iterate over each hourly bar
        for idx in range(len(hourly_data)):
            current_row = hourly_data.iloc[idx]
            current_time = current_row['time']
            current_close = current_row['c']
            current_high = current_row['h']
            current_low = current_row['l']
            print(f"\nProcessing candle at {current_time}: Close={current_close}, High={current_high}, Low={current_low}")

            # Update pivots up to current index
            pivots = self.detect_pivots(hourly_data.iloc[:idx + 1])
            print(f"Detected pivots up to now: {pivots}")

            # Get the most recent pivot against the trade
            if self.entry.signal in ['bull_eng', 'hammer']:
                # Get the most recent confirmed pivot high
                recent_pivot = self.get_recent_pivot(pivots, pivot_type='h')
                if recent_pivot:
                    print(f"Most recent pivot high: {recent_pivot}")
                else:
                    print("No pivot high found.")
                # Check if pivot high is greater than all future highs up to current index
                if recent_pivot and self.is_pivot_valid(hourly_data.iloc[recent_pivot['index']:idx + 1], recent_pivot, direction='bullish'):
                    pivot_price = recent_pivot['price']
                    print(f"Valid pivot high at price {pivot_price}")
                    # Check if price closes above pivot price
                    if current_close > pivot_price:
                        # Adjust stop loss to the low of the breakout candle minus pip value
                        breakout_low = current_low
                        stop_loss = breakout_low - self.pip_value
                        self.entry.stop_loss = stop_loss  # Update the entry's stop loss
                        print(f"Stop loss adjusted to {stop_loss} at {current_time}")

            elif self.entry.signal in ['bear_eng', 'shooting_star']:
                # Get the most recent confirmed pivot low
                recent_pivot = self.get_recent_pivot(pivots, pivot_type='l')
                if recent_pivot:
                    print(f"Most recent pivot low: {recent_pivot}")
                else:
                    print("No pivot low found.")
                # Check if pivot low is less than all future lows up to current index
                if recent_pivot and self.is_pivot_valid(hourly_data.iloc[recent_pivot['index']:idx + 1], recent_pivot, direction='bearish'):
                    pivot_price = recent_pivot['price']
                    print(f"Valid pivot low at price {pivot_price}")
                    # Check if price closes below pivot price
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

    def detect_pivots(self, data_slice):
        print(f"Detecting pivots in data slice of length {len(data_slice)}")
        pivots = []
        total_rows = len(data_slice)
        for i in range(self.depth, total_rows - self.depth):
            is_pivot_high = True
            is_pivot_low = True
            current_high = data_slice.iloc[i]['h']
            current_low = data_slice.iloc[i]['l']
            print(f"Checking for pivot at index {i}: High={current_high}, Low={current_low}")

            # Check for pivot high
            for j in range(i - self.depth, i + self.depth + 1):
                if data_slice.iloc[j]['h'] > current_high:
                    is_pivot_high = False
                    print(f"Not a pivot high at index {i} because candle at index {j} has higher high {data_slice.iloc[j]['h']}")
                    break

            # Check for pivot low
            for j in range(i - self.depth, i + self.depth + 1):
                if data_slice.iloc[j]['l'] < current_low:
                    is_pivot_low = False
                    print(f"Not a pivot low at index {i} because candle at index {j} has lower low {data_slice.iloc[j]['l']}")
                    break

            if is_pivot_high:
                print(f"Confirmed pivot high at index {i} with price {current_high}")
                pivots.append({'index': i, 'price': current_high, 'type': 'h'})
            if is_pivot_low:
                print(f"Confirmed pivot low at index {i} with price {current_low}")
                pivots.append({'index': i, 'price': current_low, 'type': 'l'})

        return pivots

    def get_recent_pivot(self, pivots, pivot_type):
        print(f"Getting most recent pivot of type '{pivot_type}'")
        # Get the most recent pivot of specified type
        for pivot in reversed(pivots):
            if pivot['type'] == pivot_type:
                print(f"Found recent pivot: {pivot}")
                return pivot
        print("No recent pivot found.")
        return None

    def is_pivot_valid(self, data_slice, pivot, direction):
        print(f"Validating pivot at index {pivot['index']} with price {pivot['price']} for direction '{direction}'")
        # Check if pivot price is greater/less than all future highs/lows up to current point
        if direction == 'bullish':
            future_highs = data_slice['h']
            is_valid = (future_highs <= pivot['price']).all()
            print(f"Pivot is {'valid' if is_valid else 'invalid'} for bullish direction.")
            return is_valid
        elif direction == 'bearish':
            future_lows = data_slice['l']
            is_valid = (future_lows >= pivot['price']).all()
            print(f"Pivot is {'valid' if is_valid else 'invalid'} for bearish direction.")
            return is_valid
        print("Invalid direction specified.")
        return False

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
