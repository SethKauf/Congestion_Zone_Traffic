# src/app.py
import joblib
import lstm_functions as lf
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf

# -----------------------------------------------------------------------------
# Paths / config
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../src
REPO_ROOT = HERE.parent                         # .../congestion_zone_traffic

COL_TIMESTAMP = "toll_timestamp"
COL_PRED = "traffic_volume_pred"
SEQ_LEN = 48

st.set_page_config(
    page_title="NYC Congestion Pricing - Traffic Volume",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _find_artifact(filename: str) -> Path:
    """
    Find a file robustly on Streamlit Cloud by checking common locations
    and finally falling back to a repo-wide rglob.
    """
    candidates = [
        REPO_ROOT / filename,
        HERE / filename,
        REPO_ROOT / "artifacts" / filename,
        REPO_ROOT / "models" / filename,
        REPO_ROOT / "assets" / filename,
        HERE / "artifacts" / filename,
        HERE / "models" / filename,
        HERE / "assets" / filename,
    ]

    for p in candidates:
        if p.exists():
            return p

    # last resort: search the repo (can be slower, but repo is usually small)
    hits = list(REPO_ROOT.rglob(filename))
    if hits:
        return hits[0]

    # Not found: raise with a very explicit message
    raise FileNotFoundError(
        f"Could not find '{filename}'.\n"
        f"Checked:\n- " + "\n- ".join(str(c) for c in candidates) + "\n\n"
        f"REPO_ROOT contents:\n"
        + "\n".join(f"- {p.name}" for p in sorted(REPO_ROOT.iterdir()))
        + "\n\n"
        f"SRC (HERE) contents:\n"
        + "\n".join(f"- {p.name}" for p in sorted(HERE.iterdir()))
    )


@st.cache_resource(show_spinner=True)
def load_artifacts():
    """
    Load model + scaler once per server process.
    Uses multiple fallbacks and prints helpful diagnostics on failure.
    """
    model_path = _find_artifact("lstm_model.keras")
    scaler_path = _find_artifact("scaler.pkl")

    # Helpful debug info (visible in app)
    st.sidebar.caption("### Artifact paths (debug)")
    st.sidebar.code(f"MODEL_PATH = {model_path}\nSCALER_PATH = {scaler_path}")

    # Load scaler first (fast)
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to load scaler with joblib.\n"
            f"Path: {scaler_path}\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e

    # Load keras model (try both keras entrypoints)
    keras_errors = []
    for loader_name, loader_fn in [
        ("tf.keras.models.load_model", tf.keras.models.load_model),
        ("keras.saving.load_model", getattr(tf.keras, "keras", None).models.load_model if getattr(tf.keras, "keras", None) else None),
    ]:
        if loader_fn is None:
            continue
        try:
            # compile=False avoids needing loss/optimizer objects
            model = loader_fn(model_path, compile=False)
            return model, scaler
        except Exception as e:
            keras_errors.append(f"{loader_name} -> {type(e).__name__}: {e}")

    raise RuntimeError(
        "Failed to load Keras model.\n\n"
        f"Path attempted: {model_path}\n\n"
        "Most common causes:\n"
        "- The .keras file is NOT actually in the deployed repo (committed locally but not pushed)\n"
        "- The file is tracked by Git LFS but Streamlit Cloud didn’t pull LFS objects\n"
        "- Version mismatch / custom objects required (less likely if compile=False)\n\n"
        "Loader errors:\n- " + "\n- ".join(keras_errors)
    )


@st.cache_data(show_spinner=True, ttl=60 * 10)
def get_predictions(last_n_days: int):
    """
    Returns:
      pred_df_small: DataFrame with [toll_timestamp, traffic_volume_pred] for filtering/ranking
      y_true: np.array of true values aligned to predictions (for diagnostic plot)
      y_pred: np.array of predicted values (for diagnostic plot)
      mae, rmse: floats
    """
    raw = lf.load_last_n_days_brooklyn(n_days=last_n_days)
    feat = lf.aggr_data(raw)

    mdl, scaler = load_artifacts()

    pred_df_full, mae, rmse = lf.mdl_pipe(mdl, scaler, feat, SEQ_LEN=SEQ_LEN)

    pred_df_full = pred_df_full.copy()
    pred_df_full[COL_TIMESTAMP] = pd.to_datetime(pred_df_full[COL_TIMESTAMP], errors="coerce")

    y_true = pred_df_full["traffic_volume_true"].to_numpy()
    y_pred = pred_df_full["traffic_volume_pred"].to_numpy()

    pred_df_small = pred_df_full[[COL_TIMESTAMP, COL_PRED]].sort_values(COL_TIMESTAMP)
    return pred_df_small, y_true, y_pred, mae, rmse


def _normalize_to_ny_naive(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    # tz-aware -> convert then drop tz
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    # tz-naive -> assume UTC then convert to NY then drop tz
    return ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("NYC Congestion Pricing - Predicted Traffic Volume Coming From Brooklyn")

with st.sidebar:
    st.header("Controls")
    last_n_days = st.slider("Pull last N days of raw data", 1, 60, 14)

    st.divider()
    st.header("Filter View")
    start_date = st.date_input("Start Date", value=(datetime.now() - timedelta(days=7)).date())
    end_date = st.date_input("End date", value=datetime.now().date())

    st.divider()
    st.header("Ranking (within selected range)")
    st.caption("Ranks timestamps by predicted volume (Lowest -> Highest).")
    top_n = st.slider("Show N lowest / N highest", 3, 25, 10)

with st.spinner("Loading data and generating predictions..."):
    pred, y_true, y_pred, mae, rmse = get_predictions(last_n_days)

# Validate output
needed = {COL_TIMESTAMP, COL_PRED}
if not needed.issubset(pred.columns):
    st.error(f"Prediction output must include columns: {sorted(list(needed))}")
    st.stop()

# Normalize timestamps to NY time (tz-naive for easy filtering)
pred = pred.copy()
pred[COL_TIMESTAMP] = _normalize_to_ny_naive(pred[COL_TIMESTAMP])

# Filter by date range (inclusive)
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
pred_f = pred[(pred[COL_TIMESTAMP] >= start_dt) & (pred[COL_TIMESTAMP] <= end_dt)].copy()

if pred_f.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# --- Main Layout ---
left, right = st.columns([2, 1])

with left:
    st.subheader("Predicted traffic volume over time")

    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:,.2f}")
    m2.metric("RMSE", f"{rmse:,.2f}")

    n_points = st.slider("Number of points to display", 50, 500, 200, step=50)

    # Your helper should return a matplotlib figure
    fig = lf.mdl_plt_streamlit(y_true, y_pred, n=n_points)
    st.pyplot(fig, clear_figure=True)

    with st.expander("Show prediction data (first 200 rows)"):
        st.text(pred_f.head(200).to_string(index=False))

with right:
    st.subheader("Summary")
    st.metric("Rows in view", f"{len(pred_f):,}")
    st.metric(
        "Range",
        f"{pred_f[COL_TIMESTAMP].min():%Y-%m-%d %H:%M} → {pred_f[COL_TIMESTAMP].max():%Y-%m-%d %H:%M}",
    )
    st.metric("Avg predicted volume", f"{pred_f[COL_PRED].mean():,.0f}")
    st.metric("Min predicted volume", f"{pred_f[COL_PRED].min():,.0f}")
    st.metric("Max predicted volume", f"{pred_f[COL_PRED].max():,.0f}")

st.divider()
st.subheader("Lowest → Highest predicted timestamps (within selected range)")

ranked = pred_f.sort_values(COL_PRED, ascending=True).reset_index(drop=True)

lowest = ranked.head(top_n).copy()
lowest.insert(0, "Rank (lowest)", range(1, len(lowest) + 1))

highest = ranked.tail(top_n).sort_values(COL_PRED, ascending=False).copy()
highest.insert(0, "Rank (highest)", range(1, len(highest) + 1))

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Lowest")
    st.text(lowest.to_string(index=False))

with c2:
    st.markdown("### Highest")
    st.text(highest.to_string(index=False))
