import pandas as pd
import numpy as np
import tpqoa
import os

api = tpqoa.tpqoa("oanda.cfg")

instruments = ["AUD_JPY", "AUD_USD", "CAD_JPY", "EUR_GBP", "EUR_JPY", "EUR_USD", "EUR_NZD", "GBP_JPY", "GBP_USD", "USD_CAD", "USD_CHF", "USD_JPY", "XAG_USD", "XAU_USD"]

writer = pd.ExcelWriter("EngulfingFibo.xlsx", engine="xlsxwriter")

def fetch_data(instrument, granularity, start, end):
    df = api.get_history(instrument = instrument, granularity=granularity, start=start, end=end, price = "M")
    df.drop(["complete", "volume"], axis=1, inplace=True)
    df.reset_index(inplace=True)
    df["time"] = pd.to_datetime(df["time"])     
    df.sort_values("time", inplace=True)
    print(df.head(5))
    return df

def save_to_excel(df, instrument, granularity, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{instrument}_{granularity}.xlsx")
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=granularity, index=False)
    print (f"Data for {instrument} and {granularity} saved to {file_path}")

output_dir = r"C:\Users\grave\OneDrive\Coding\fxdata"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for instrument in instruments:
    df_daily = fetch_data(instrument, "D", "2016-01-01" , "2023-12-31")
    df_hourly = fetch_data(instrument, "H1", "2016-01-01", "2023-12-31")

    save_to_excel(df_daily, instrument, "D", output_dir)
    save_to_excel(df_hourly, instrument, "H1", output_dir)