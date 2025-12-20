import joblib
import lstm_functions as lf
import pandas as pd
import streamlit as st

from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras


# ---------------------------
# Basic config
# ---------------------------
st.set_page_config(page_title="NYC Congestion Pricing - Traffic Volume", layout="wide")

COL_TIMESTAMP = "toll_timestamp"
COL_PRED = "traffic_volume_pred"


# ---------------------------
# Helpers: find files anywhere in repo + debug
# ---------------------------
HERE = Path(__file__).resolve().parent          # .../src
REPO_ROOT = HERE.parent                         # .../congestion_zone_traffic


def _repo_tree_preview(root: Path, max_depth: int = 4) -> str:
    """Small tree preview for debugging what's actually in Streamlit Cloud."""
    lines = []
    root = root.resolve()
    for p in sorted(root.rglob("*")):
        try:
            rel = p.relative_to(root)
        except Exception:
            rel = p
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if p.is_dir():
            continue
        lines.append(str(rel))
    return "\n".join(lines[:500])  # cap output


def find_first(root: Path, patterns: list[str]) -> Path | None:
    """Search root recursively for the first file matching any glob pattern."""
    for pat in patterns:
        matches = list(root.rglob(pat))
        if matches:
            # prefer the shortest path (often closest to repo root)
            matches = sorted(matches, key=lambda x: len(x.parts))
            return matches[0]
    return None


@st.cache_resource(show_spinner=False)
def resolve_artifact_paths() -> tuple[Path, Path]:
    """
    Finds model + scaler anywhere in the repo.
    Tries common names/locations first, then recursive search.
    """
    # Common expected locations first
    candidates_model = [
        REPO_ROOT / "lstm_model.keras",
        REPO_ROOT / "models" / "lstm_model.keras",
        REPO_ROOT / "artifacts" / "lstm_model.keras",
        HERE / "lstm_model.keras",
        HERE / "models" / "lstm_model.keras",
    ]
    candidates_scaler = [
        REPO_ROOT / "scaler.pkl",
        REPO_ROOT / "models" / "scaler.pkl",
        REPO_ROOT / "artifacts" / "scaler.pkl",
        HERE / "scaler.pkl",
        HERE / "models" / "scaler.pkl",
    ]

    model_path = next((p for p in candidates_model if p.exists()), None)
    scaler_path = next((p for p in candidates_scaler if p.exists()), None)

    # If not found, do a recursive search
    if model_path is None:
        model_path = find_first(REPO_ROOT, ["*.keras", "*.h5", "*.hdf5"])
    if scaler_path is None:
        scaler_path = find_first(REPO_ROOT, ["scaler.pkl", "*scaler*.pkl", "*.pkl"])

    if model_path is None or scaler_path is None:
        # Expose what files actually exist on the server
        with st.expander("Debug: repo file listing (Streamlit Cloud)"):
            st.code(_repo_tree_preview(REPO_ROOT), language="text")

        missing = []
        if model_path is None:
            missing.append("model (.keras/.h5)")
        if scaler_path is None:
            missing.append("scaler (.pkl)")

        raise FileNotFoundError(
            "Could not locate required artifact(s): " + ", ".join(missing) + "\n\n"
            "Make sure these files are committed to GitHub (and not ignored), or if using Git LFS, "
            "ensure Streamlit Cloud can pull the real files (not pointer stubs)."
        )

    return model_path, scaler_path


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model + scaler once per app session."""
    model_path, scaler_path = resolve_artifact_paths()

    # show resolved paths in sidebar for sanity
    st.session_state["resolved_model_path"] = str(model_path)
    st.session_state["resolved_scaler_path"] = str(scaler_path)

    # Inference-only load to avoid optimizer/metrics deserialization issues
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, compile=False)

    scaler = joblib.load(scaler_path)
    return model, scaler


# ---------------------------
# Predictions
# ---------------------------
@st.cache_data(show_spinner=True, ttl=60 * 10)
def get_predictions(last_n_days: int):
    raw = lf.load_last_n_days_brooklyn(n_days=last_n_days)
    feat = lf.aggr_data(raw)

    mdl, scaler = load_artifacts()
    pred_df_full, mae, rmse = lf.mdl_pipe(mdl, scaler, feat, SEQ_LEN=48)

    pred_df_full = pred_df_full.copy()
    pred_df_full[COL_TIMESTAMP] = pd.to_datetime(pred_df_full[COL_TIMESTAMP])

    y_true = pred_df_full["traffic_volume_true"].to_numpy()
    y_pred = pred_df_full["traffic_volume_pred"].to_numpy()

    pred_df_small = pred_df_full[[COL_TIMESTAMP, COL_PRED]].sort_values(COL_TIMESTAMP)
    return pred_df_small, y_true, y_pred, mae, rmse


# ---------------------------
# UI
# ---------------------------
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

    st.divider()
    st.header("Artifacts (resolved)")
    st.caption("This is where Streamlit Cloud actually found them.")
    st.text(st.session_state.get("resolved_model_path", "(not resolved yet)"))
    st.text(st.session_state.get("resolved_scaler_path", "(not resolved yet)"))


# Run predictions with clear errors
try:
    with st.spinner("Loading data and generating predictions..."):
        pred, y_true, y_pred, mae, rmse = get_predictions(last_n_days)
except Exception as e:
    st.exception(e)
    st.stop()

# Normalize timestamps to NY time
ts = pd.to_datetime(pred[COL_TIMESTAMP], errors="coerce")

if getattr(ts.dt, "tz", None) is not None:
    pred[COL_TIMESTAMP] = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
else:
    pred[COL_TIMESTAMP] = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.tz_localize(None)

# Filter by date range
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
pred_f = pred[(pred[COL_TIMESTAMP] >= start_dt) & (pred[COL_TIMESTAMP] <= end_dt)].copy()

if pred_f.empty:
    st.warning("No data in the selected date range.")
    st.stop()

# Layout
left, right = st.columns([2, 1])

with left:
    st.subheader("Predicted traffic volume over time")
    st.metric("MAE", f"{mae:,.2f}")
    st.metric("RMSE", f"{rmse:,.2f}")

    n_points = st.slider("Number of points to display", 50, 500, 200, step=50)
    fig = lf.mdl_plt_streamlit(y_true, y_pred, n=n_points)
    st.pyplot(fig, clear_figure=True)

    with st.expander("Show prediction data"):
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
