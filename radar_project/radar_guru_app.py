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
    hidden_stripe_score,
)
import joblib

importlib.reload(radar_simulation)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILES = {
    'Original SVM': 'metal_classifier_svm.joblib',
    'Feature-Engineered SVM': 'metal_classifier_svm_fe.joblib',
    'Original SVM (Augmented)': 'metal_classifier_svm_aug.joblib',
    'Feature-Engineered SVM (Augmented)': 'metal_classifier_svm_fe_aug.joblib',
}

@st.cache_resource(show_spinner=False)
def load_models():
    """Load available models from MODELS_DIR; return dict name->estimator.
    Adds per-file diagnostics so user can understand missing models scenario.
    """
    loaded = {}
    diagnostics = []
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                loaded[name] = joblib.load(path)
                diagnostics.append(f"✅ {fname} loaded")
            except Exception as e:
                diagnostics.append(f"⚠️ {fname} error: {e}")
        else:
            diagnostics.append(f"❌ {fname} not found at {path}")
    # Store diagnostics in session_state for potential display
    st.session_state['model_load_report'] = diagnostics
    return loaded

models = load_models()

# -------------------- SESSION STATE INIT --------------------
if 'frame_scenario' not in st.session_state:
    st.session_state['frame_scenario'] = None
if 'hm' not in st.session_state:
    st.session_state['hm'] = None
if 'proc' not in st.session_state:
    st.session_state['proc'] = None

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
if not models:
    with st.sidebar.expander("Model Diagnostics"):
        st.write("Models directory:", MODELS_DIR)
        st.write("Load attempts:")
        for line in st.session_state.get('model_load_report', []):
            st.write(line)
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
gen_tab, clf_tab, hidden_tab = st.tabs(["1. Create Frame","2. Metal Detection","3. Hidden Object Detection"])

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
    generate_btn = st.button("Generate / Refresh Frame")
    need_new = generate_btn or st.session_state['hm'] is None or st.session_state['frame_scenario'] != scenario
    if need_new:
        hm_new = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario=scenario, metal=metal_flag, clutter_level=clutter_level, snr_db=snr_db)
        proc_new = denoise_background_subtract(hm_new, method='median', kernel_size=5)
        st.session_state['hm'] = hm_new
        st.session_state['proc'] = proc_new
        st.session_state['frame_scenario'] = scenario
    hm = st.session_state['hm']
    proc = st.session_state['proc']
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
    # End generation tab content

with clf_tab:
    st.subheader("Metal Detection")
    if st.session_state['hm'] is None:
        st.info("Generate a frame in the Generation tab first.")
    else:
        model = models.get(model_choice)
        hm = st.session_state['hm']
        proc = st.session_state['proc']
        prob = infer(model_choice, model, hm, proc)
        show_hidden_controls = use_hidden_mode or st.session_state.get('frame_scenario') == 'hidden'
        if show_hidden_controls:
            stripe_threshold = st.slider("Hidden Stripe Threshold", 0.5, 5.0, 1.2, 0.1, help="Heuristic stripe score cutoff; lower = more sensitive.")
            stripe_metrics = hidden_stripe_score(proc)
            stripe_score = stripe_metrics['score']
        else:
            stripe_metrics = None
            stripe_score = 0.0
        decision = (prob >= adaptive_threshold) or (show_hidden_controls and stripe_score >= (stripe_threshold if show_hidden_controls else 999))
        c1, c2, c3 = st.columns(3)
        c1.metric("Probability", f"{prob:.3f}")
        c2.metric("Threshold", f"{adaptive_threshold:.2f}")
        c3.metric("Decision", "METAL" if decision else "NON-METAL")
        st.caption("Probability = model confidence; METAL if probability ≥ threshold or hidden stripe score crosses cutoff (hidden mode).")
        if stripe_metrics is not None:
            st.markdown("**Hidden Stripe Heuristic**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Stripe Score", f"{stripe_metrics['score']:.2f}")
            s2.metric("Peak z", f"{stripe_metrics['peak_z']:.2f}")
            s3.metric("Band frac", f"{stripe_metrics['band_frac']:.2f}")
            s4.metric("Contrast", f"{stripe_metrics['contrast']:.2f}")
            st.caption("Stripe score combines peak prominence, band length and contrast; probability OR stripe triggers METAL.")
            with st.expander("How stripe heuristic works"):
                st.markdown(
                    "We scan row intensities of the processed map, smooth them, then measure: \n"
                    "- Peak z: strongest row intensity prominence vs global mean/std.\n"
                    "- Band frac: longest contiguous run of elevated rows (stripe continuity).\n"
                    "- Contrast: peak row vs surrounding neighborhood.\n"
                    "Composite score = 0.5*peak_z + 0.3*(band_frac*10) + 0.2*contrast. Lower threshold increases sensitivity to faint stripes."
                )
        fig_pb, ax_pb = plt.subplots(figsize=(4.5,0.5))
        ax_pb.barh([0], [prob], color='orange')
        ax_pb.axvline(adaptive_threshold, color='red', linestyle='--', label='Threshold')
        ax_pb.set_xlim(0,1)
        ax_pb.set_yticks([])
        ax_pb.set_xlabel('Probability')
        ax_pb.legend(loc='upper right', fontsize=7)
        st.pyplot(fig_pb)
        if ('Feature-Engineered' in model_choice) and show_features:
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
        show_batch = st.checkbox("Show batch probability distribution", value=False, help="Simulate multiple frames & visualize probability spread.")
        if show_batch:
            batch_count = st.slider("Batch test count", 5, 100, 25, 5, help="How many frames to simulate (same settings).")
            batch_btn = st.button("Run Batch Simulation")
        else:
            batch_btn = False
        if batch_btn and show_batch:
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
    st.subheader("Hidden Object Detection")
    st.markdown("Single hidden frame detection and optional batch stress test.")
    colh1, colh2 = st.columns(2)
    with colh1:
        hidden_snr = st.slider("Hidden Frame SNR (dB)", 4, 20, 8, 1)
    with colh2:
        hidden_clutter = st.slider("Hidden Clutter Level", 0.0, 0.5, 0.25, 0.01)
    colt1, colt2 = st.columns(2)
    with colt1:
        hidden_prob_thr = st.slider("Probability Threshold (P_thr)", 0.0, 1.0, 0.10, 0.01)
    with colt2:
        hidden_stripe_thr = st.slider("Stripe Threshold (S_thr)", 0.5, 5.0, 1.20, 0.1)
    detect_btn = st.button("Run Hidden Detection")
    if detect_btn:
        model = models.get(model_choice)
        hm_hidden = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='hidden', metal=True, clutter_level=hidden_clutter, snr_db=hidden_snr)
        proc_hidden = denoise_background_subtract(hm_hidden, method='median', kernel_size=5)
        prob_hidden = infer(model_choice, model, hm_hidden, proc_hidden)
        stripe_vals = hidden_stripe_score(proc_hidden)
        stripe_score_hidden = stripe_vals['score']
        decision_hidden = (prob_hidden >= hidden_prob_thr) or (stripe_score_hidden >= hidden_stripe_thr)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Probability", f"{prob_hidden:.3f}")
        mc2.metric("Stripe Score", f"{stripe_score_hidden:.2f}")
        mc3.metric("Decision", "METAL" if decision_hidden else "NON-METAL")
        with st.expander("Hidden Frame Visuals"):
            vc1, vc2 = st.columns(2)
            with vc1:
                fig_hr, ax_hr = plt.subplots(figsize=(4,4))
                imhr = ax_hr.imshow(hm_hidden, origin='lower', aspect='auto', cmap='inferno'); ax_hr.set_title('Raw Hidden Heatmap'); plt.colorbar(imhr, ax=ax_hr, fraction=0.046, pad=0.04)
                st.pyplot(fig_hr)
            with vc2:
                fig_hp, ax_hp = plt.subplots(figsize=(4,4))
                imhp = ax_hp.imshow(proc_hidden, origin='lower', aspect='auto', cmap='inferno'); ax_hp.set_title('Processed Hidden Heatmap'); plt.colorbar(imhp, ax=ax_hp, fraction=0.046, pad=0.04)
                st.pyplot(fig_hp)
        with st.expander("Stripe Components"):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Peak z", f"{stripe_vals['peak_z']:.2f}")
            sc2.metric("Band frac", f"{stripe_vals['band_frac']:.2f}")
            sc3.metric("Contrast", f"{stripe_vals['contrast']:.2f}")
            st.caption("Composite score = 0.5*peak_z + 0.3*(band_frac*10) + 0.2*contrast.")
    with st.expander("Batch Stress Test (Hidden vs Clutter)"):
        st.caption("Evaluate recall (hidden detected) vs false alarms (clutter misclassified). Uses same thresholds defined above.")
        lab_hidden = st.slider("Hidden samples", 10, 150, 40, 10)
        lab_clutter = st.slider("Clutter samples", 10, 150, 40, 10)
        run_batch = st.button("Run Batch Stress Test")
        show_hist = st.checkbox("Show probability histograms", value=True)
        if run_batch:
            model = models.get(model_choice)
            p_hidden = []
            p_clutter = []
            s_hidden = []
            s_clutter = []
            for _ in range(lab_hidden):
                hm_h = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='hidden', metal=True, clutter_level=hidden_clutter, snr_db=hidden_snr)
                proc_h = denoise_background_subtract(hm_h, method='median', kernel_size=5)
                p_hidden.append(infer(model_choice, model, hm_h, proc_h))
                s_hidden.append(hidden_stripe_score(proc_h)['score'])
            for _ in range(lab_clutter):
                hm_c = generate_range_doppler_heatmap(range_bins, doppler_bins, scenario='clutter', metal=False, clutter_level=hidden_clutter, snr_db=hidden_snr)
                proc_c = denoise_background_subtract(hm_c, method='median', kernel_size=5)
                p_clutter.append(infer(model_choice, model, hm_c, proc_c))
                s_clutter.append(hidden_stripe_score(proc_c)['score'])
            p_hidden = np.array(p_hidden); p_clutter = np.array(p_clutter)
            s_hidden = np.array(s_hidden); s_clutter = np.array(s_clutter)
            hidden_detect = (p_hidden >= hidden_prob_thr) | (s_hidden >= hidden_stripe_thr)
            clutter_false = (p_clutter >= hidden_prob_thr) | (s_clutter >= hidden_stripe_thr)
            recall = float(hidden_detect.mean()); fpr = float(clutter_false.mean())
            bc1, bc2 = st.columns(2)
            bc1.metric("Hidden Recall", f"{recall*100:.1f}%")
            bc2.metric("Clutter FPR", f"{fpr*100:.1f}%")
            tp = int(hidden_detect.sum()); fn = int(lab_hidden - tp)
            fp = int(clutter_false.sum()); tn = int(lab_clutter - fp)
            st.json({"TP_hidden_detected": tp, "FN_hidden_missed": fn, "FP_clutter_false_alarm": fp, "TN_clutter_correct": tn})
            if show_hist:
                fig_hh, ax_hh = plt.subplots(figsize=(6,3))
                ax_hh.hist(p_hidden, bins=15, alpha=0.5, label='Hidden P')
                ax_hh.hist(p_clutter, bins=15, alpha=0.5, label='Clutter P')
                ax_hh.axvline(hidden_prob_thr, color='red', label='P_thr')
                ax_hh.set_xlabel('Probability'); ax_hh.legend(); st.pyplot(fig_hh)
                fig_sh, ax_sh = plt.subplots(figsize=(6,3))
                ax_sh.hist(s_hidden, bins=15, alpha=0.5, label='Hidden S')
                ax_sh.hist(s_clutter, bins=15, alpha=0.5, label='Clutter S')
                ax_sh.axvline(hidden_stripe_thr, color='purple', label='S_thr')
                ax_sh.set_xlabel('Stripe Score'); ax_sh.legend(); st.pyplot(fig_sh)
            st.caption("Adjust thresholds to balance recall vs false alarms. Lower values increase sensitivity.")

st.footer = st.caption("© 2025 RadarGuru")
