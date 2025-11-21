"""RadarGuru Streamlit Frontend

Run:
  streamlit run radar_guru_app.py

Purpose:
Branded interface for RadarGuru product demonstration.
Features:
- Scenario selection (empty, metal_object, clutter, hidden)
- Adjustable clutter & SNR
- Model selection (original / feature-engineered / augmented)
- Tabs: Single Frame, Batch Simulation, Hidden Mode Lab, Metrics
- Adaptive threshold suggestion based on batch score distribution
- Feature vector visualization for FE models
- Export current frame & decision to JSON
"""
import os
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import importlib
import radar_simulation
from radar_simulation import (
    generate_range_doppler_heatmap,
    denoise_background_subtract,
    extract_features,
)
import joblib

importlib.reload(radar_simulation)

MODELS_DIR = os.path.join('.', 'models')
MODEL_FILES = {
    'Original SVM': 'metal_classifier_svm.joblib',
    'Feature-Engineered SVM': 'metal_classifier_svm_fe.joblib',
    'Original SVM (Augmented)': 'metal_classifier_svm_aug.joblib',
    'Feature-Engineered SVM (Augmented)': 'metal_classifier_svm_fe_aug.joblib',
}

@st.cache_resource(show_spinner=False)
def load_models():
    loaded = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
            except Exception as e:
                st.warning(f"Failed loading {fname}: {e}")
    return loaded

models = load_models()

# -------------------- SIDEBAR CONFIG --------------------
st.sidebar.image("https://static.streamlit.io/logo.png", width=80)
st.sidebar.markdown("# RadarGuru")
st.sidebar.caption("Dekho Jo Chhupa Hai – Hidden metal detection demo")
scenario = st.sidebar.selectbox("Scenario", ['empty','metal_object','clutter','hidden'], index=1)
metal_flag = st.sidebar.checkbox("Metal present?", value=(scenario in ['metal_object','hidden']))
clutter_level = st.sidebar.slider("Clutter Level", 0.0, 0.5, 0.15, 0.01)
snr_db = st.sidebar.slider("SNR (dB)", 4, 30, 18, 1)
img_size = st.sidebar.selectbox("Image Size", ['64x64','128x64'], index=0)
range_bins, doppler_bins = (64,64) if img_size=='64x64' else (128,64)
model_choice = st.sidebar.selectbox("Model", list(models.keys()) if models else ["No models"])
base_threshold = st.sidebar.slider("Base Threshold", 0.0, 1.0, 0.50, 0.01)
use_hidden_mode = st.sidebar.checkbox("Hidden Mode (force 0.10 threshold)")
adaptive_threshold = 0.10 if use_hidden_mode else base_threshold
st.sidebar.write(f"**Active Threshold:** {adaptive_threshold:.2f}")
show_features = st.sidebar.checkbox("Show Feature Vector (FE models)", value=True)

st.title("RadarGuru – Real-time Synthetic Radar Metal Detection")
st.markdown("### Explore detection across scenarios with adaptive thresholds and engineered features.")

# Tabs
single_tab, batch_tab, hidden_tab, metrics_tab = st.tabs(["Single Frame","Batch Simulation","Hidden Mode Lab","Metrics / Export"])

# Utility inference function
def infer(model_name, model_obj, raw_hm, proc_hm):
    if model_obj is None:
        return 0.0
    if 'Feature-Engineered' in model_name:
        fv = extract_features(raw_hm, proc_hm, k_top=10).reshape(1,-1)
        return model_obj.predict_proba(fv)[0,1]
    flat = proc_hm.reshape(1,-1)
    return model_obj.predict_proba(flat)[0,1]

# -------------------- SINGLE FRAME --------------------
with single_tab:
    st.subheader("Single Frame Inference")
    hm = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario=scenario, metal=metal_flag, clutter_level=clutter_level, snr_db=snr_db)
    proc = denoise_background_subtract(hm, method='median', kernel_size=5)
    model = models.get(model_choice)
    prob = infer(model_choice, model, hm, proc)
    decision = prob >= adaptive_threshold
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability", f"{prob:.3f}")
    c2.metric("Threshold", f"{adaptive_threshold:.2f}")
    c3.metric("Decision", "METAL" if decision else "NON-METAL")
    col_raw, col_proc = st.columns(2)
    with col_raw:
        fig_r, ax_r = plt.subplots(figsize=(4,4))
        ax_r.imshow(hm, origin='lower', aspect='auto', cmap='inferno')
        ax_r.set_title("Raw Heatmap")
        st.pyplot(fig_r)
    with col_proc:
        fig_p, ax_p = plt.subplots(figsize=(4,4))
        ax_p.imshow(proc, origin='lower', aspect='auto', cmap='inferno')
        ax_p.set_title("Processed Heatmap")
        st.pyplot(fig_p)
    if show_features and 'Feature-Engineered' in model_choice:
        st.markdown("**Feature Vector**")
        fv = extract_features(hm, proc, k_top=10)
        labels = ['raw_mean','raw_std','raw_max','raw_energy','raw_frac>0.5','proc_mean','proc_std','proc_max','proc_energy','proc_frac>0.5'] + [f'top_{i+1}' for i in range(10)]
        st.dataframe({"feature": labels, "value": [round(x,5) for x in fv.tolist()]})

# -------------------- BATCH SIMULATION --------------------
with batch_tab:
    st.subheader("Batch Simulation / Threshold Suggestion")
    batch_count = st.number_input("Frames to simulate", 5, 200, 30, 5)
    simulate_btn = st.button("Simulate Batch")
    if simulate_btn:
        model = models.get(model_choice)
        probs = []
        raw_list = []
        proc_list = []
        for i in range(batch_count):
            hm_b = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario=scenario, metal=metal_flag, clutter_level=clutter_level, snr_db=snr_db)
            proc_b = denoise_background_subtract(hm_b, method='median', kernel_size=5)
            raw_list.append(hm_b); proc_list.append(proc_b)
            probs.append(infer(model_choice, model, hm_b, proc_b))
        probs = np.array(probs)
        suggested_thr = np.percentile(probs, 20)  # choose threshold capturing top 80% probabilities
        st.write(f"Suggested threshold (retain ≥80% detections): {suggested_thr:.2f}")
        decision_rate = np.mean(probs >= adaptive_threshold)
        st.write(f"Current threshold decision rate: {decision_rate*100:.1f}%")
        fig_hist, ax_h = plt.subplots(figsize=(6,3))
        ax_h.hist(probs, bins=15, color='orange', alpha=0.8)
        ax_h.axvline(adaptive_threshold, color='red', label='Active Threshold')
        ax_h.axvline(suggested_thr, color='green', linestyle='--', label='Suggested')
        ax_h.set_xlabel('Probability'); ax_h.set_ylabel('Count'); ax_h.legend()
        st.pyplot(fig_hist)
        st.caption("You can switch to Hidden Mode for higher recall (threshold 0.10) if many weak positives exist.")

# -------------------- HIDDEN MODE LAB --------------------
with hidden_tab:
    st.subheader("Hidden Mode Laboratory")
    lab_count = st.slider("Hidden frames (metal)", 10, 150, 50, 10)
    clutter_count = st.slider("Clutter frames (non-metal)", 10, 150, 50, 10)
    run_lab = st.button("Run Hidden Lab")
    if run_lab:
        model = models.get(model_choice)
        probs_hidden = []
        probs_clutter = []
        for _ in range(lab_count):
            hm_h = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='hidden', metal=True, clutter_level=0.25, snr_db=8)
            proc_h = denoise_background_subtract(hm_h, method='median', kernel_size=5)
            probs_hidden.append(infer(model_choice, model, hm_h, proc_h))
        for _ in range(clutter_count):
            hm_c = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='clutter', metal=False, clutter_level=0.3, snr_db=8)
            proc_c = denoise_background_subtract(hm_c, method='median', kernel_size=5)
            probs_clutter.append(infer(model_choice, model, hm_c, proc_c))
        ph = np.array(probs_hidden); pc = np.array(probs_clutter)
        recall_at_thr = np.mean(ph >= adaptive_threshold)
        false_pos_rate = np.mean(pc >= adaptive_threshold)
        st.write(f"Hidden Metal Recall @ {adaptive_threshold:.2f}: {recall_at_thr*100:.1f}%")
        st.write(f"Clutter False Positive Rate @ {adaptive_threshold:.2f}: {false_pos_rate*100:.1f}%")
        fig_lab, ax_lab = plt.subplots(figsize=(6,3))
        ax_lab.hist(ph, bins=15, alpha=0.6, label='Hidden Metal')
        ax_lab.hist(pc, bins=15, alpha=0.6, label='Clutter')
        ax_lab.axvline(adaptive_threshold, color='red', label='Threshold')
        ax_lab.set_xlabel('Probability'); ax_lab.set_ylabel('Count'); ax_lab.legend()
        st.pyplot(fig_lab)
        st.caption("Aim: High hidden recall with acceptable clutter FPR. Adjust base threshold or enable Hidden Mode.")

# -------------------- METRICS / EXPORT --------------------
with metrics_tab:
    st.subheader("Metrics & Export")
    export_btn = st.button("Export Last Single Frame Decision")
    if export_btn and 'hm' in locals():
        out = {
            'scenario': scenario,
            'metal_flag': metal_flag,
            'clutter_level': clutter_level,
            'snr_db': snr_db,
            'probability': float(prob),
            'threshold_used': float(adaptive_threshold),
            'decision': 'metal' if decision else 'non-metal',
            'model': model_choice,
        }
        import json
        os.makedirs('exports', exist_ok=True)
        path = os.path.join('exports', f'radarguru_frame_{int(time.time())}.json')
        with open(path,'w') as f:
            json.dump(out, f, indent=2)
        st.success(f"Exported to {path}")
    st.markdown("**System Notes**")
    st.write("Models loaded:", list(models.keys()))
    st.write("Engineered features enhance hidden detection; switch models to compare recall vs precision trade-offs.")

st.footer = st.caption("© 2025 RadarGuru – Loha detection powered by synthetic radar simulation")
