import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_URL = "https://data.ny.gov/resource/t6yz-b64h.json"

CHUNK_SIZE = 50000

load_dotenv("../.secrets")
APP_TOKEN = os.getenv("NYS_CRZ")

def fetch_chunk(offset: int, where: str) -> list[dict]:
    params = {
        "$limit": CHUNK_SIZE,
        "$offset": offset,
        "$where": where,
        # Optional: only pull columns you care about (edit as needed)
        "$select": ",".join([
            "toll_10_minute_block",
            "toll_date",
            "toll_hour",
            "detection_region",
            "detection_group",
            "vehicle_class",
            "crz_entries",
            "excluded_roadway_entries",
        ]),
        "$order": "toll_10_minute_block ASC",
    }

    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    r = requests.get(BASE_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

def get_max_timestamp_brooklyn():
    params = {
        "$select": "max(toll_10_minute_block) as max_ts",
        "$where": "detection_region = 'Brooklyn'",
    }

    headers = {}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    out = r.json()

    max_ts = pd.to_datetime(out[0]["max_ts"], utc=True)
    return max_ts

def load_last_n_days_brooklyn(n_days=14) -> pd.DataFrame:
    max_ts = get_max_timestamp_brooklyn()
    start_ts = max_ts - pd.Timedelta(days=n_days)

    start_str = start_ts.strftime("%Y-%m-%dT%H:%M:%S.000")
    end_str = max_ts.strftime("%Y-%m-%dT%H:%M:%S.000")

    where = (
        f"detection_region = 'Brooklyn' "
        f"AND toll_10_minute_block >= '{start_str}' "
        f"AND toll_10_minute_block <= '{end_str}'"
    )

    offset = 0
    chunks = []

    while True:
        print(f"Fetching rows {offset + 1}–{offset + CHUNK_SIZE}")
        data = fetch_chunk(offset, where)

        if not data:
            break

        chunks.append(pd.DataFrame(data))
        offset += CHUNK_SIZE

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["toll_10_minute_block"] = pd.to_datetime(
        df["toll_10_minute_block"], utc=True, errors="coerce"
    )

    return df

def aggr_data(df):
    # half hour blocks
    df['toll_timestamp'] = pd.to_datetime(df['toll_10_minute_block'], errors="coerce").dt.floor("30min")
    
    # add entries cols
    df['traffic_volume'] = (df['crz_entries'].astype(int) + df['excluded_roadway_entries'].astype(int))
    
    data = df[['toll_timestamp','traffic_volume']].groupby('toll_timestamp').sum('traffic_volume').reset_index()
    
    return data

def make_lstm_eval_windows(series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i-seq_len:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def mdl_pipe(mdl, scaler, data, SEQ_LEN=48):
    # Ensure sorted
    data = data.sort_values("toll_timestamp").reset_index(drop=True)

    # Extract series
    series = (
        data["traffic_volume"]
        .astype(float)
        .to_numpy()
    )

    # Scale
    series_scaled = scaler.transform(series.reshape(-1, 1)).flatten()

    # Create LSTM windows
    X_test, y_test = make_lstm_eval_windows(series_scaled, SEQ_LEN)

    # Reshape for LSTM
    X_test = X_test.reshape(X_test.shape[0], SEQ_LEN, 1)

    # Predict
    y_pred_scaled = mdl.predict(X_test, verbose=0).flatten()

    # Inverse scale
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # --- TIMESTAMP ALIGNMENT ---
    # y_pred corresponds to timestamps starting at index SEQ_LEN
    timestamps = data.loc[SEQ_LEN:, "toll_timestamp"].reset_index(drop=True)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Return a dataframe (BEST PRACTICE for Streamlit)
    pred_df = pd.DataFrame({
        "toll_timestamp": timestamps,
        "traffic_volume_true": y_true,
        "traffic_volume_pred": y_pred
    })

    return pred_df, mae, rmse

def mdl_plt(y_true,y_pred,n=200):
    plt.figure(figsize=(12, 4))
    plt.gcf().patch.set_facecolor("#0b1e39")
    plt.gca().set_facecolor("#0b1e39")
    plt.plot(y_true[:n], label="Actual",color="#F1948A",linewidth=2)
    plt.plot(y_pred[:n], label="Predicted",color="#85C1E9")
    plt.legend()
    plt.title(f"LSTM – First {n} predictions",
              color="white",
              fontsize=14,
              fontweight="bold"
             )
    plt.xlabel("Validation index", color="white")
    plt.ylabel("Traffic volume", color="white")
    plt.tick_params(colors="white")
    # Grid (major + minor)
    plt.grid(which="major", color="gray", linewidth=0.3)
    plt.minorticks_on()
    plt.grid(which="minor", color="gray", linewidth=0.2)
    # Legend
    leg = plt.legend()
    for t in leg.get_texts():
        t.set_color("gray")

    # Remove spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.show()
    
def mdl_plt_streamlit(y_true, y_pred, n=200):
    fig, ax = plt.subplots(figsize=(12, 4))

    # Background
    fig.patch.set_facecolor("#0b1e39")
    ax.set_facecolor("#0b1e39")

    # Lines
    ax.plot(y_true[:n], label="Actual", color="#F1948A", linewidth=2)
    ax.plot(y_pred[:n], label="Predicted", color="#85C1E9")

    # Title & labels
    ax.set_title(
        f"LSTM – First {n} predictions",
        color="white",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel("Validation index", color="white")
    ax.set_ylabel("Traffic volume", color="white")

    # Ticks
    ax.tick_params(colors="white")

    # Grid
    ax.grid(which="major", color="gray", linewidth=0.3)
    ax.minorticks_on()
    ax.grid(which="minor", color="gray", linewidth=0.2)

    # Legend
    leg = ax.legend()
    for t in leg.get_texts():
        t.set_color("gray")

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig