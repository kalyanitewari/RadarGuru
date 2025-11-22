# RadarGuru Deep Overview: What, Why, How

This single file explains the entire project end‑to‑end in a structured "What / Why / How" manner for each component. Use it as a mastery guide before presenting, extending, or onboarding collaborators.

---
## 1. Mission
**What:** Detect metallic objects (including weak, partially hidden ones) in synthetic radar range–Doppler heatmaps.
**Why:** Hidden metallic signatures matter in safety (security screening), operations (logistics asset tracking), and navigation (robotics in degraded visibility). Missing faint objects is risk; over‑triggering on clutter wastes resources.
**How:** Generate controllable synthetic radar frames, preprocess to enhance targets, engineer robust features, apply classical SVM models, augment for edge cases, expose thresholds and evaluation via an interactive UI.

---
## 2. Core Design Principles
| Principle | Why | How Implemented |
|-----------|-----|-----------------|
| Fast Iteration | Real radar acquisition is slow/expensive | Synthetic generator (`radar_simulation.py`) with adjustable scenario + SNR + clutter |
| Explainability | Stakeholders need traceability | Feature vectors (stats + top‑k) are interpretable vs opaque embeddings |
| Hidden Sensitivity | Baseline missed faint metals | Threshold sweeps + augmentation + feature engineering |
| Modularity | Swap models / preprocessing easily | Separate modules: generation, features, training, evaluation, UI |
| Reproducibility | Stable comparisons & demos | Fixed RNG seeds, saved datasets (CSV + NPY), joblib models, plots |
| Low Complexity | Avoid heavy infra for prototype | Classical ML (SVM) before deep models given environment issues |

---
## 3. Synthetic Data Generation (`radar_simulation.py`)
### What
Creates 2D range–Doppler-like heatmaps plus auxiliary 1D profiles (range / Doppler). Scenarios: `empty`, `metal_object`, `clutter`, `hidden`.
### Why
Enables controlled variation (object count, clutter intensity, SNR) to stress test classification strategies without hardware.
### How
1. Start with zero array; optionally add clutter (Gaussian speckle scaled by `clutter_level`).
2. Sample number of objects based on scenario (0, 1, or random 1–2 for clutter). Each object → Gaussian blob with random center, size, amplitude (amplitude boosted if `metal=True`).
3. Hidden scenario adds faint elongated stripe across Doppler axis (models smeared/occluded return).
4. Add thermal noise by converting `snr_db` to linear std and injecting Gaussian noise.
5. Normalize 0..1, apply mild Gaussian blur to mimic point-spread function.
6. Optional FFT (`apply_fft`) for visualization/testing transform domain.
7. Dataset builder loops, chooses scenario distribution to balance metal vs non-metal, saves `.npy`, PNG, and `labels.csv` manifest.
8. `denoise_background_subtract()`: median/gaussian filter background removal to enhance contrast for feature extraction.
9. `extract_features()`: numeric feature vector: raw stats, processed stats, top‑k intensities.

---
## 4. Feature Engineering
### What
Compact vector (≈20 features) summarizing energy distribution and salient peaks.
### Why
Hidden metallic signals are faint, elongated, sometimes broad—flattening raw pixels loses structured differences. Handcrafted descriptors emphasize relative intensities and spread.
### How
- Raw stats: mean (overall energy), std (contrast), max (peak), energy (sum of squares), fraction above 0.5 (density of strong returns).
- Processed stats: same metrics post background subtraction (robust to clutter).
- Top‑k intensities: sorted peak values capturing target prominence vs background.
- Padding ensures fixed dimensionality when fewer than k pixels.

---
## 5. Model Choices
### What
RBF SVM pipelines: baseline with flatten + PCA; engineered variant with features only; augmented versions adding hidden/clutter samples.
### Why
- **SVM**: Works well with small/medium feature spaces; kernel handles nonlinear separation; faster to iterate versus solving deep learning environment issues (TensorFlow DLL on Python 3.12).
- **PCA baseline**: Reduces dimensionality from raw pixels; preserves variance but misses semantic structure of faint targets.
- **Feature-engineered SVM**: Encodes physically meaningful attributes improving hidden recall.
### How
Pipeline steps: StandardScaler → (PCA baseline only) → SVC(probability=True, class_weight='balanced'). `class_weight` compensates minor label imbalance and protects recall.

---
## 6. Dataset Evolution
| Stage | What | Why | How Outcome |
|-------|------|-----|-------------|
| Prototype | 300 mixed samples | Quick feasibility test | Balanced generation, internal split | Established baseline metrics |
| Expanded | 400 train/val + 100 test | Stability & reproducibility | Stratified split | Reduced variance, consistent evaluation |
| Augmented | Added hidden + clutter | Teach edge cases | Programmatic injection functions | Hidden recall → ~1.00 @ threshold 0.10 |

---
## 7. Hidden Detection Challenge
### What
Baseline hidden recall ≈0.10 at threshold 0.50; metallic faint signatures often predicted non-metal.
### Why
Flatten+PCA retains variance but not contextual shape differences of elongated, low-amplitude returns.
### How (Improvements)
1. Feature engineering: increased discrimination of weak elongated returns.
2. Threshold sweep: systematically evaluated precision/recall vs probability cutoff; selected 0.10 for hidden mode.
3. Augmentation: added synthetic hidden/clutter frames to training set; improved recall robustness.
4. Separate operating modes: standard (0.50), hidden (0.10), clutter (0.20–0.30).

---
## 8. Threshold Strategy
### What
Adaptive threshold depending on scene type and clutter level.
### Why
Single global threshold over-penalizes recall in hidden scenes; too low threshold in clutter inflates false positives.
### How
- Sweep produced `threshold_sweep_hidden.png` and JSON metrics.
- FE/augmented models hold precision tolerably high at low threshold.
- Potential dynamic formula: `thr = base_thr * (1 - clutter_factor*0.3)` to raise threshold in heavy clutter.

---
## 9. Evaluation Artifacts
### What
Plots: confusion matrices, ROC, PR curves, threshold sweep; JSON metric files (`hidden_baseline_metrics.json`, `hidden_threshold_sweep.json`, `hidden_augmented_metrics.json`).
### Why
Visual & machine-readable records allow comparison over iterations and parameter changes.
### How
Notebook cells save PNGs under `models/`. Metrics serialized via `json.dump` for downstream ingestion or dashboarding.

---
## 10. Streamlit App (`radar_guru_app.py`)
### What
Interactive UI with tabs: Single Frame, Batch Simulation, Hidden Mode Lab, Metrics/Export.
### Why
Non-technical stakeholders need tangible exploration; threshold tuning benefits from visual distribution of probabilities.
### How
- Generates scenarios live, shows raw vs processed heatmaps.
- Computes feature vector + probability using selected model.
- Allows threshold slider adjustment and batch histograms.
- Hidden Lab focuses on faint object probability distributions.

---
## 11. Architecture Flow (Runtime)
**What:** Sequence from raw acquisition to decision & alert.
**Why:** Ensures each transformation adds value (noise suppression, feature amplification) without unnecessary latency.
**How:**
1. Generate / Acquire frame.
2. Background subtraction + denoise.
3. Feature extraction.
4. SVM probability.
5. Adaptive threshold (context: hidden vs clutter vs standard).
6. Temporal smoothing (optional rolling average or majority vote).
7. Emit alert & log metrics.

---
## 12. Performance & Latency
**What:** Keep inference <50 ms GPU / <120 ms CPU.
**Why:** Real-time responsiveness essential in screening / navigation scenarios.
**How:** Lightweight features (<25 floats), SVM classification (<3 ms), efficient NumPy operations, small input shape (64×64). Future optimization via vectorized background updates and possible CNN quantization.

---
## 13. Reliability & Monitoring
**What:** Track drift, false alert spikes, threshold efficacy.
**Why:** Synthetic-trained model may degrade on real data; need observability early.
**How:** Emit per-frame metrics (probability, smoothed_prob, clutter_factor, dynamic_threshold). Periodic calibration frames update precision estimates; threshold adjustments logged for audit.

---
## 14. Security & Safety
**What:** Guard against malformed input and alert flooding.
**Why:** Stability and trust in operational deployment.
**How:** Validate array shapes, clamp values, rate-limit alerts (K consecutive frames), fallback to baseline energy heuristic if model unavailable.

---
## 15. Extensibility Roadmap
| Phase | What | Why | How |
|-------|------|-----|-----|
| Short | Probability calibration | Stable threshold choices | Isotonic/Platt on validation probs |
| Short | FE vs CNN comparison | Potential accuracy gain | Export FE features into CNN input or hybrid model |
| Medium | Temporal modeling | Smooth noise & micro-dropouts | LSTM/TCN on frame feature sequences |
| Medium | Drift detection | Maintain performance | KS test on feature distributions |
| Long | Multi-sensor fusion | Disambiguate clutter | Combine radar probability with camera or RF sensor |
| Long | Active learning loop | Continual improvement | Collect uncertain frames for labeling |

---
## 16. Component Responsibilities Recap
| Component | What | Why | How |
|-----------|------|-----|-----|
| `radar_simulation.py` | Synthetic frame & feature generator | Rapid controlled data | Gaussian blobs + noise + stats extraction |
| Classification Notebook | Train baseline/FE/augmented models | Establish metrics & artifacts | Pipelines + joblib + plots |
| Hidden Detection Notebook | Stress hidden/clutter cases | Tune thresholds & measure recall | Generate eval sets + sweeps |
| `radar_guru_app.py` | Interactive demo & experimentation | Stakeholder engagement | Streamlit tabs & sliders |
| `deployment_design.md` | Operational blueprint | Bridge prototype to production | Flowchart + threshold logic |

---
## 17. Key Decisions & Trade-offs
| Decision | Rationale | Downsides | Future Mitigation |
|----------|-----------|-----------|------------------|
| Use SVM over CNN | Simplicity, fast iteration | Potential ceiling on performance | Later integrate CNN once env stable |
| Feature engineering focus | Recover hidden recall | Manual tuning overhead | Auto-search / learned embeddings |
| Low threshold for hidden | Maximize recall | More false positives risk | Adaptive + temporal smoothing |
| Augment with synthetic hidden/clutter | Teach edge cases cheaply | Synthetic bias risk | Mix with real captures |

---
## 18. Lessons Learned
- Hidden recall gap often due to representation, not classification algorithm alone.
- Threshold sweeps provide objective operating point selection—avoid arbitrary cutoffs.
- Augmentation is powerful if it preserves distribution characteristics (don’t over-clone trivial patterns).
- Classical ML remains competitive when paired with strong features.

---
## 19. Quick Reference Cheat Sheet
| Task | Steps |
|------|-------|
| Retrain baseline | Load data → flatten+PCA → SVM fit → save model |
| Generate features | Denoise → stats + top‑k → assemble vector |
| Hidden evaluation | Generate hidden/clutter set → predict probs → threshold analysis |
| Add augmentation | Generate extra hidden/clutter frames → concatenate → retrain pipelines |
| Adjust threshold | Sweep probs vs metrics → choose F1/recall balancing point |

---
## 20. Minimal Reproduction Flow (CLI Style)
1. Generate dataset: `generate_dataset('data/trainval', 400)` & `generate_dataset('data/test', 100)`.
2. Train baseline & FE SVM (notebooks or extracted script).
3. Run hidden detection notebook → baseline metrics & sweep.
4. Augment training → retrain → re-evaluate hidden/clutter.
5. Launch Streamlit app for demo threshold tuning.

---
## 21. Mental Model Summary
Think of each frame as a noisy 2D energy field: we amplify contrast (background subtraction), condense it into descriptive numbers (features), then map those to probability. Hidden mode simply shifts the acceptable confidence threshold because faint metals produce lower raw scores; augmentation and features reshape the probability distribution so those faint signals stand out more.

---
## 22. Final One-Liner
"RadarGuru turns faint radar hints into actionable detection by combining physics-inspired feature engineering with adaptive thresholds." 

---
© 2025 RadarGuru
