import pandas as pd

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
