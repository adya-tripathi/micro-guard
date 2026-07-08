"""
MicroGuard: Real-Time Anomaly Detector
Streamlit dashboard powered by a from-scratch multivariate log-Gaussian model.
"""
 
import streamlit as st
import pandas as pd
import numpy as np
 
from gaussian import log_gaussian, load_model
from data_simulator import stream_rows, load_dataset
 
st.set_page_config(page_title="MicroGuard", layout="wide")
st.title("🛠️ MicroGuard: Real-Time Anomaly Detector")
st.caption("Multivariate Gaussian anomaly detection, built from scratch — no sklearn in the inference path.")
 
DATA_PATH = "smart_manufacturing_data.csv"   
MODEL_PATH = "model_params.npz"
 
mu, sigma, default_eps, features,scaler_mean,scaler_scale = load_model(MODEL_PATH)
 
st.sidebar.header("Model Info")
st.sidebar.write("Features used:", features)
st.sidebar.write("Trained epsilon (log-space):", round(default_eps, 4))
 
epsilon = st.sidebar.slider(
    "Anomaly threshold (log-probability)",
    min_value=float(default_eps - 5),
    max_value=float(default_eps + 5),
    value=float(default_eps),
    step=0.1,
)
 
speed = st.sidebar.slider("Playback delay (seconds/row)", 0.0, 1.0, 0.05, 0.01)
run = st.toggle("Start live stream", value=False)
 
# ---- Layout placeholders ----
status_placeholder = st.empty()
col1, col2 = st.columns(2)
chart_placeholder = col1.empty()
table_placeholder = col2.empty()
 
history = []
MAX_HISTORY = 200
 
if run:
    stream = stream_rows(DATA_PATH, features, delay=speed, loop=True)
    for x, row in stream:
        processed_row = row[features].copy()
        
        processed_row["predicted_remaining_life"] = np.log1p(processed_row["predicted_remaining_life"])
        x_unscaled = processed_row.values
        x_scaled = (x_unscaled - scaler_mean)/scaler_scale

        custom_score = log_gaussian(x_scaled.reshape(1, -1), mu, sigma)[0]
        from scipy.stats import multivariate_normal
        scipy_score = multivariate_normal.logpdf(x_scaled, mean=mu, cov=sigma)

        score = scipy_score
        is_anomaly = score < epsilon
 
        history.append(score)
        if len(history) > MAX_HISTORY:
            history.pop(0)
 
        with status_placeholder.container():
            if is_anomaly:
                st.error(f"⚠️ ANOMALY DETECTED — log-score = {score:.2f} (threshold {epsilon:.2f})")
            else:
                st.success(f"✅ System Normal — log-score = {score:.2f} (threshold {epsilon:.2f})")
 
        chart_placeholder.line_chart(pd.DataFrame({"log_score": history}))
        table_placeholder.dataframe(row[features].to_frame().T, use_container_width=True)
 
        if not st.session_state.get("keep_running", True):
            break
else:
    st.info("Toggle **Start live stream** in the sidebar to begin monitoring.")
    df_preview = load_dataset(DATA_PATH)
    st.write("Dataset preview:", df_preview.sample(10))
 