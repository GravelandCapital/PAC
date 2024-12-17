import pandas as pd

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

