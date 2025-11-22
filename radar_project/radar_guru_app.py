"""RadarGuru Streamlit Frontend

Run:
    streamlit run radar_guru_app.py

Purpose:
Simplified interface focused on three core sections:
1. Generation – create a synthetic radar frame and understand parameters.
2. Classification – view model probability, feature stats (if engineered), batch distribution.
3. Hidden Detection – evaluate hidden vs clutter recall & false positives.

Removed extra tabs and wording; concise explanations for demo.
"""
import os
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
st.sidebar.caption("Hidden + Metal target detection demo")
scenario = st.sidebar.selectbox("Scene Type", ['empty','metal_object','clutter','hidden'], index=1, help="Overall pattern: hidden=faint stripe, metal_object=clear blob, clutter=weak blobs, empty=no target")
clutter_level = st.sidebar.slider("Clutter", 0.0, 0.5, 0.15, 0.01, help="Speckle intensity. Higher = more background interference.")
snr_db = st.sidebar.slider("SNR dB", 4, 30, 18, 1, help="Higher SNR = cleaner signal; lower = noisier.")
img_size = st.sidebar.selectbox("Image Size", ['64x64','128x64'], index=0)
range_bins, doppler_bins = (64,64) if img_size=='64x64' else (128,64)
model_choice = st.sidebar.selectbox("Model", list(models.keys()) if models else ["No models"], help="Choose detector variant.")
base_threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.50, 0.01, help="Probability cutoff for METAL.")
use_hidden_mode = st.sidebar.checkbox("Hidden Focus (0.10)", help="Force threshold=0.10 to catch faint hidden targets.")
adaptive_threshold = 0.10 if use_hidden_mode else base_threshold
st.sidebar.write(f"Active Threshold: **{adaptive_threshold:.2f}**")
show_features = st.sidebar.checkbox("Show Features (FE models)", value=False)

# Derive metal flag implicitly from scenario to avoid confusion
metal_flag = scenario in ['metal_object','hidden']

st.title("RadarGuru – Synthetic Radar Demo")
st.markdown(
    "**Flow** → 1) Create Frame  2) Detect Metal  3) Stress Test Hidden.\n"
    "If probability ≥ threshold ⇒ METAL; else NON-METAL. Hidden mode lowers the threshold to catch faint stripes."
)

with st.expander("Quick Guide"):
    st.markdown(
        "Scene Type defines pattern; clutter & SNR shape difficulty. Heatmap shows echoes; processed map highlights peaks. Probability comes from chosen model; threshold converts probability to label. Hidden focus sets threshold=0.10 for faint stripe detection."
    )
with st.expander("Key Terms"):
    st.markdown(
        "**Probability**: Model score in [0,1] estimating chance at least one metal target is present given current frame features.\n"
        "**Threshold**: Cutoff applied to probability. If probability ≥ threshold ⇒ label METAL else NON-METAL. Lower threshold increases detections (recall) but can raise false alarms.\n"
        "**SNR (dB)**: Signal-to-Noise Ratio. Higher SNR = stronger target vs noise; lower SNR makes blobs harder to distinguish. Noise std ≈ 10^(-SNR/20). We reduce SNR to make hidden cases challenging."
    )

# Simplified three tabs
gen_tab, clf_tab, hidden_tab = st.tabs(["1. Create Frame","2. Metal Detection","3. Hidden Stress Test"])

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
with gen_tab:
    st.subheader("Create Frame")
    st.caption("Generate synthetic radar-like frame. 'hidden' = faint stripe + weak blob. 'metal_object' = strong blob. 'clutter' = scattered weak blobs.")
    generate_btn = st.button("Generate Frame")
    if generate_btn or 'hm' not in locals():
        hm = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario=scenario, metal=metal_flag, clutter_level=clutter_level, snr_db=snr_db)
        proc = denoise_background_subtract(hm, method='median', kernel_size=5)
    col1, col2 = st.columns(2)
    with col1:
        fig_r, ax_r = plt.subplots(figsize=(4,4))
        im_r = ax_r.imshow(hm, origin='lower', aspect='auto', cmap='inferno')
        ax_r.set_title("Raw Heatmap")
        plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
        st.pyplot(fig_r)
    with col2:
        fig_p, ax_p = plt.subplots(figsize=(4,4))
        im_p = ax_p.imshow(proc, origin='lower', aspect='auto', cmap='inferno')
        ax_p.set_title("Processed (Background Subtracted)")
        # Overlay top-k intensity points (processed) for intuitiveness if using FE model
        if 'Feature-Engineered' in model_choice:
            flat_idx = np.argsort(proc.ravel())[::-1][:10]
            ys, xs = np.unravel_index(flat_idx, proc.shape)
            ax_p.scatter(xs, ys, s=30, edgecolors='white', facecolors='none', linewidths=1.2, label='Top peaks')
            ax_p.legend(loc='lower right', fontsize=8)
        plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
        st.pyplot(fig_p)
    raw_stats = {'mean': float(np.mean(hm)), 'std': float(np.std(hm)), 'max': float(np.max(hm)), 'energy': float(np.sum(hm*hm)), 'frac>0.5': float(np.mean(hm>0.5))}
    proc_stats = {'mean': float(np.mean(proc)), 'std': float(np.std(proc)), 'max': float(np.max(proc)), 'energy': float(np.sum(proc*proc)), 'frac>0.5': float(np.mean(proc>0.5))}
    st.markdown("**Stats (Raw vs Processed)**")
    st.json({'raw': raw_stats, 'processed': proc_stats})
    st.caption("Raw = target blobs + clutter + noise (then normalized & lightly blurred). Processed = raw − median background to emphasize local peaks (candidate metallic returns).")
    with st.expander("Interpret"):
        st.markdown(
            "Raw: target echoes + clutter + noise. Processed: peaks isolated. Strong tight blob ⇒ metal. Faint horizontal band ⇒ hidden. Scattered weak blobs ⇒ clutter."
        )
    with st.expander("What did these parameters do?"):
        st.write(
            f"Scenario: '{scenario}' determines number/type of blobs. Metal flag: {metal_flag} boosts amplitude. Clutter level adds speckle intensity ≈ {clutter_level}. SNR={snr_db} dB sets Gaussian noise std (lower SNR ⇒ noisier frame)."
        )
        st.write("Normalization rescales to 0..1 so feature distributions stay consistent across runs.")

with clf_tab:
    st.subheader("Metal Detection")
    if 'hm' not in locals():
        st.info("Generate a frame in the Generation tab first.")
    else:
        model = models.get(model_choice)
        prob = infer(model_choice, model, hm, proc)
        decision = prob >= adaptive_threshold
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability", f"{prob:.3f}")
        c2.metric("Threshold", f"{adaptive_threshold:.2f}")
        c3.metric("Decision", "METAL" if decision else "NON-METAL")
        # Inline concise definitions
        st.caption("Probability = model confidence metal is present. Threshold = cutoff; ≥ gives METAL. Hidden Focus forces threshold=0.10 for faint stripe recall.")
        # Simple horizontal bar showing probability and threshold
        fig_pb, ax_pb = plt.subplots(figsize=(4.5,0.5))
        ax_pb.barh([0], [prob], color='orange')
        ax_pb.axvline(adaptive_threshold, color='red', linestyle='--', label='Threshold')
        ax_pb.set_xlim(0,1)
        ax_pb.set_yticks([])
        ax_pb.set_xlabel('Probability')
        ax_pb.legend(loc='upper right', fontsize=7)
        st.pyplot(fig_pb)
        if 'Feature-Engineered' in model_choice and show_features:
            st.markdown("**Engineered Feature Vector**")
            fv = extract_features(hm, proc, k_top=10)
            labels = ['raw_mean','raw_std','raw_max','raw_energy','raw_frac>0.5','proc_mean','proc_std','proc_max','proc_energy','proc_frac>0.5'] + [f'top_{i+1}' for i in range(10)]
            st.dataframe({"feature": labels, "value": [round(x,5) for x in fv.tolist()]})
            with st.expander("Feature meanings"):
                st.markdown(
                    "- raw_mean/std: overall energy & contrast before background removal.\n"
                    "- raw_energy: sum of squares (emphasizes strong pixels).\n"
                    "- raw_frac>0.5: density of high-intensity pixels.\n"
                    "- processed_*: same stats after background subtraction (higher target separation).\n"
                    "- top_k: strongest pixel intensities from processed map capturing peak prominence."
                )
        batch_count = st.slider("Batch test count", 5, 100, 25, 5, help="How many frames to simulate (same settings) to see probability spread.")
        batch_btn = st.button("Batch Test")
        if batch_btn:
            model = models.get(model_choice)
            probs = []
            for _ in range(batch_count):
                hm_b = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario=scenario, metal=metal_flag, clutter_level=clutter_level, snr_db=snr_db)
                proc_b = denoise_background_subtract(hm_b, method='median', kernel_size=5)
                probs.append(infer(model_choice, model, hm_b, proc_b))
            probs = np.array(probs)
            suggested_thr = float(np.percentile(probs, 20))
            st.write(f"Suggested threshold (capture ≥80% positives): {suggested_thr:.2f}")
            fig_hist, ax_h = plt.subplots(figsize=(6,3))
            ax_h.hist(probs, bins=15, color='steelblue', alpha=0.8)
            ax_h.axvline(adaptive_threshold, color='red', label='Active')
            ax_h.axvline(suggested_thr, color='green', linestyle='--', label='Suggested')
            ax_h.set_xlabel('Probability'); ax_h.set_ylabel('Count'); ax_h.legend()
            st.pyplot(fig_hist)
            st.caption("Suggested threshold = 20th percentile (≈ keep 80%). Lower = more recall; higher = fewer false alarms.")

with hidden_tab:
    st.subheader("Hidden Stress Test")
    st.caption("Faint stripe + weak blob vs clutter. Gauge recall vs false alarms.")
    lab_hidden = st.slider("Hidden metals", 10, 150, 50, 10)
    lab_clutter = st.slider("Clutter non-metals", 10, 150, 50, 10)
    run_hidden = st.button("Evaluate Hidden/Clutter")
    if run_hidden:
        model = models.get(model_choice)
        probs_hidden = []
        probs_clutter = []
        for _ in range(lab_hidden):
            hm_h = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='hidden', metal=True, clutter_level=0.25, snr_db=8)
            proc_h = denoise_background_subtract(hm_h, method='median', kernel_size=5)
            probs_hidden.append(infer(model_choice, model, hm_h, proc_h))
        for _ in range(lab_clutter):
            hm_c = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='clutter', metal=False, clutter_level=0.3, snr_db=8)
            proc_c = denoise_background_subtract(hm_c, method='median', kernel_size=5)
            probs_clutter.append(infer(model_choice, model, hm_c, proc_c))
        ph = np.array(probs_hidden); pc = np.array(probs_clutter)
        recall = float(np.mean(ph >= adaptive_threshold))
        fpr = float(np.mean(pc >= adaptive_threshold))
        c1, c2 = st.columns(2)
        c1.metric("Hidden Recall", f"{recall*100:.1f}%")
        c2.metric("Clutter FPR", f"{fpr*100:.1f}%")
        # Confusion matrix style summary
        tp = int(np.sum(ph >= adaptive_threshold))
        fn = int(ph.shape[0] - tp)
        fp = int(np.sum(pc >= adaptive_threshold))
        tn = int(pc.shape[0] - fp)
        st.markdown("**Counts (Hidden=Positive, Clutter=Negative)**")
        st.json({"TP_hidden_detected": tp, "FN_hidden_missed": fn, "FP_clutter_false_alarm": fp, "TN_clutter_correct": tn})
        fig_hd, ax_hd = plt.subplots(figsize=(6,3))
        ax_hd.hist(ph, bins=15, alpha=0.6, label='Hidden')
        ax_hd.hist(pc, bins=15, alpha=0.6, label='Clutter')
        ax_hd.axvline(adaptive_threshold, color='red', label='Threshold')
        ax_hd.set_xlabel('Probability'); ax_hd.set_ylabel('Count'); ax_hd.legend()
        st.pyplot(fig_hd)
        st.caption(f"Hidden Recall {recall*100:.1f}%, Clutter FPR {fpr*100:.1f}%. Lower threshold ⇒ ↑ recall & ↑ false alarms.")

    # Note: Batch simulation moved to Classification tab; export removed for simplicity.

st.footer = st.caption("© 2025 RadarGuru")
