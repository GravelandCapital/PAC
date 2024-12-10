from dataclasses import dataclass
import pandas as pd

@dataclass
class Entry:
    instrument: str
    signal: str
    entry_type: str
    price: float
    order_time: pd.Timestamp
    row_index: int  # Index in the daily dataframe
    original_stop_loss: float = None 
    stop_loss: float = None
    take_profit: float = None
    order_status: str = "PENDING"
    filled_time: pd.Timestamp = None
    exit_time: pd.Timestamp = None
    exit_price: float = None