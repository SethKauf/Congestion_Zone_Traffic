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
    """Find file by checking common locations, then repo-wide search."""
    candidates = [
        REPO_ROOT / filename,      # repo root (your case)
        HERE / filename,           # src/
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

    hits = list(REPO_ROOT.rglob(filename))
    if hits:
        return hits[0]

    raise FileNotFoundError(
        f"Could not find '{filename}'.\n\n"
        f"Checked:\n- " + "\n- ".join(str(c) for c in candidates) + "\n\n"
        f"Repo root ({REPO_ROOT}) contains:\n"
        + "\n".join(f"- {p.name}" for p in sorted(REPO_ROOT.iterdir()))
        + "\n\n"
        f"Src dir ({HERE}) contains:\n"
        + "\n".join(f"- {p.name}" for p in sorted(HERE.iterdir()))
    )


def _peek_text(path: Path, n: int = 200) -> str:
    """Read first n bytes as text (best-effort) for debugging."""
    try:
        b = path.read_bytes()[:n]
        return b.decode("utf-8", errors="replace")
    except Exception as e:
        return f"<could not read bytes: {type(e).__name__}: {e}>"


def _is_git_lfs_pointer(path: Path) -> bool:
    """Detect Git LFS pointer files (common Streamlit Cloud deploy gotcha)."""
    head = _peek_text(path, 200)
    return "git-lfs.github.com/spec" in head or head.strip().startswith("version https://git-lfs.github.com/spec")


@st.cache_resource(show_spinner=True)
def load_artifacts():
    """
    Load model + scaler once per server process.
    Includes strong diagnostics for Streamlit Cloud deployments.
    """
    model_path = _find_artifact("lstm_model.keras")
    scaler_path = _find_artifact("scaler.pkl")

    # Sidebar debug
    with st.sidebar:
        st.caption("### Artifact debug")
        st.code(
            "Resolved paths:\n"
            f"MODEL  = {model_path}\n"
            f"SCALER = {scaler_path}"
        )
        try:
            st.write("Model size (bytes):", model_path.stat().st_size)
        except Exception:
            pass
        try:
            st.write("Scaler size (bytes):", scaler_path.stat().st_size)
        except Exception:
            pass

        # LFS pointer check
        if _is_git_lfs_pointer(model_path):
            st.error("MODEL FILE LOOKS LIKE A GIT LFS POINTER (NOT THE REAL .keras FILE).")
            st.code(_peek_text(model_path, 300))

    # If the model is an LFS pointer, fail with a clear message
    if _is_git_lfs_pointer(model_path):
        raise RuntimeError(
            "Your 'lstm_model.keras' appears to be a Git LFS pointer file in the deployed environment.\n\n"
            "Fix:\n"
            "- Ensure the REAL binary is committed (not LFS), OR\n"
            "- Configure Streamlit Cloud to pull LFS objects (not always available), OR\n"
            "- Store the model as a normal GitHub file under 100MB, OR\n"
            "- Host the model elsewhere and download it at startup.\n"
        )

    # Load scaler (usually straightforward)
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to load scaler with joblib.\n"
            f"Path: {scaler_path}\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e

    # Load model (most compatible approach for TF/Keras in the cloud)
    # - compile=False avoids requiring loss/optimizer objects
    # - safe_mode=False helps if Keras 3 thinks something is unsafe to deserialize
    try:
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except TypeError:
            # older signature: safe_mode not supported
            model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        # show more diagnostics in sidebar
        with st.sidebar:
            st.caption("### Model load diagnostics")
            st.code(_peek_text(model_path, 300))

        raise RuntimeError(
            "Failed to load Keras model from lstm_model.keras.\n\n"
            f"Path: {model_path}\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            "Common fixes:\n"
            "1) Re-save the model in your training env with:\n"
            "   model.save('lstm_model.keras', include_optimizer=False)\n"
            "2) Ensure the file is NOT Git LFS pointer (must be real binary)\n"
            "3) If you used custom layers/losses/metrics, pass custom_objects on load\n"
        ) from e

    return model, scaler


@st.cache_data(show_spinner=True, ttl=60 * 10)
def get_predictions(last_n_days: int):
    """
    Returns:
      pred_df_small: DataFrame with [toll_timestamp, traffic_volume_pred]
      y_true: np.array aligned to predictions
      y_pred: np.array predictions
      mae, rmse: floats
    """
    raw = lf.load_last_n_days_brooklyn(n_days=last_n_days)
    feat = lf.aggr_data(raw)

    mdl, scaler = load_artifacts()

    pred_df_full, mae, rmse = lf.mdl_pipe(mdl, scaler, feat, SEQ_LEN=SEQ_LEN)

    pred_df_full = pred_df_full.copy()
    pred_df_full[COL_TIMESTAMP] = pd.to_datetime(pred_df_full[COL_TIMESTAMP], errors="coerce")

    y_true = pred_df_full["traffic_volume_true"].to_numpy()
    y_pred = pred_df_full[COL_PRED].to_numpy()

    pred_df_small = pred_df_full[[COL_TIMESTAMP, COL_PRED]].sort_values(COL_TIMESTAMP)
    return pred_df_small, y_true, y_pred, mae, rmse


def _normalize_to_ny_naive(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        return ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
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

needed = {COL_TIMESTAMP, COL_PRED}
if not needed.issubset(pred.columns):
    st.error(f"Prediction output must include columns: {sorted(list(needed))}")
    st.stop()

pred = pred.copy()
pred[COL_TIMESTAMP] = _normalize_to_ny_naive(pred[COL_TIMESTAMP])

start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
pred_f = pred[(pred[COL_TIMESTAMP] >= start_dt) & (pred[COL_TIMESTAMP] <= end_dt)].copy()

if pred_f.empty:
    st.warning("No data in the selected date range.")
    st.stop()

left, right = st.columns([2, 1])

with left:
    st.subheader("Predicted traffic volume over time")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:,.2f}")
    m2.metric("RMSE", f"{rmse:,.2f}")

    n_points = st.slider("Number of points to display", 50, 500, 200, step=50)

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
