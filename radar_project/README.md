# RadarGuru – Synthetic Radar Metal & Hidden Object Detection

**Tagline:** *RadarGuru: Dekho Jo Chhupa Hai* (See what’s hidden)

RadarGuru is an end‑to‑end synthetic radar experimentation platform: we generate range–Doppler style heatmaps, engineer robust features, train classical ML (SVM) models, and expose interactive threshold tuning for partially hidden metallic targets. This README explains the problem, real‑world relevance, the thought process, terminology, dataset & model evolution, plots, and how to reproduce everything locally.

---
## 1. Problem Statement
Detect metallic objects (e.g., toolboxes, vehicles, infrastructure pieces) in radar imagery where signals may be weak or partially occluded by clutter (vegetation, packaging, shielding). In hidden scenarios the target return is low SNR and spatially smeared, causing a naive classifier to miss true metals (low recall). Goal: maximize recall for hidden metals while controlling false positives in clutter.

### Real‑Life Use Cases
- Security screening: concealed metallic threats behind low‑density materials.
- Industrial logistics: verifying presence of tagged metallic pallets amidst reflective clutter.
- Robotics navigation: identifying metallic structural landmarks in rain/fog (low SNR).
- Infrastructure inspection: detecting corroded metallic plates partially covered by debris.

### Practical Challenge
Radar reflections vary with geometry, material, multipath, and noise. Hidden or partially occluded objects produce weak, elongated returns easily lost when thresholds/feature sets are tuned only for strong canonical signatures.

---
## 2. Key Concepts & Terms
- **Range–Doppler Heatmap:** 2D array approximating target energy across distance (range) and relative velocity (Doppler). We simulate patterns rather than perform real FFT processing here.
- **Clutter:** Background reflections (noise, vegetation, random scatterers) that can mimic faint targets.
- **Hidden Target:** Metallic object with attenuated / blurred signature (lower SNR, elongated Gaussian) simulating occlusion or partial shielding.
- **SNR (Signal‑to‑Noise Ratio):** Higher SNR → clearer target blob; low SNR → faint spread, harder to classify.
- **Feature Engineering:** Deriving descriptive statistics (mean, max, energy, fraction above threshold, top‑k intensities) from raw + preprocessed heatmaps.
- **PCA:** Dimensionality reduction used only in baseline model (flattened input) to retain variance with fewer components.
- **SVM (Support Vector Machine):** A robust classical model with RBF kernel; good at separating classes in moderate feature spaces without huge data.
- **Recall vs Precision:** Recall (sensitivity) crucial for hidden metals; precision must remain acceptable to avoid clutter false alarms.
- **Threshold Sweep:** Evaluating performance across probability cutoffs to select operating points (e.g., 0.10 for hidden mode).

---
## 3. Thought Process – Storyline
| Phase | What We Did | Why | Outcome |
|-------|-------------|-----|---------|
| 1. Simulation | Built synthetic generator (`radar_simulation.py`) for scenarios: empty, metal, clutter, hidden. | Fast iteration; no hardware needed. | Produced initial 300 sample dataset. |
| 2. Baseline Dataset Expansion | Increased pool to 400 train/val + 100 separate test. | Improve generalization & stable metrics. | Better separation; reproducible splits. |
| 3. Model Selection | Attempted CNN (TensorFlow DLL issues on local Python 3.12). Pivoted to SVM pipeline (StandardScaler + PCA). | Reliability & lower complexity. | Baseline SVM trained; decent visible metal accuracy. |
| 4. Feature Engineering | Added statistical + top‑k intensity features from raw & processed map. | Capture shape + energy patterns resilient to noise. | Hidden recall improved dramatically in FE variant. |
| 5. Hidden Baseline & Threshold Sweep | Evaluated hidden detection with baseline model; recall near 0.10 at default threshold 0.50. Swept thresholds. | Quantify cost of lowering threshold. | Identified lower threshold (≈0.10) for recall boost. |
| 6. Augmentation | Added synthetic hidden + clutter examples into training. | Teach classifier edge cases & ambiguous signatures. | Achieved hidden recall ≈1.00 at threshold 0.10; precision ≈0.86. |
| 7. UI & Productization | Built Streamlit app (`radar_guru_app.py`) with Single Frame, Batch Simulation, Hidden Mode Lab, Metrics tabs. | Enable interactive exploration & threshold tuning. | Usable demo for non‑technical stakeholders. |
| 8. Deployment Design | Authored `deployment_design.md` with pipeline steps, adaptive threshold logic, monitoring metrics. | Clarify real integration pathway. | Ready for PDF export & review. |

---
## 4. Dataset Evolution
| Stage | Samples (Train+Val) | Test | Hidden / Clutter Augment | Notes |
|-------|---------------------|------|--------------------------|-------|
| Initial | 300 mixed | (inline split) | None | Quick prototype metrics. |
| Expanded | 400 train/val | 100 test | None | Stable separation; produce official metrics. |
| Augmented | 400 base + hidden/clutter injected → >400 effective | 100 test | Added hidden metal + clutter non‑metal | Boosted hidden recall w/ modest precision trade‑off. |

---
## 5. Model Evolution & Rationale
| Model | Input Representation | Why | Hidden Recall (approx) | Notes |
|-------|----------------------|-----|------------------------|-------|
| Baseline SVM | Flattened heatmap + PCA | Simple, fast, low memory | ~0.10 (threshold 0.50) | Misses weak elongated signatures. |
| FE SVM | Engineered features (stats + top‑k) | Encodes energy distribution & local peaks | ~0.86 (threshold 0.10) | Large recall gain with threshold tuning. |
| Augmented FE / Orig | Same as respective | Exposure to hidden/clutter patterns | ~1.00 (threshold 0.10) | Precision ~0.86; trade studied via sweep. |

Numbers above drawn from hidden evaluation notebooks & `models/` metrics JSON.

---
## 6. Interpreting the Plots (`models/`)
| File | Meaning | Insight |
|------|---------|---------|
| `confusion_matrix_svm_val.png` / `confusion_matrix_svm_test.png` | Baseline confusion on validation/test | Shows initial misclassifications of faint metals. |
| `roc_curve_svm_val.png` / `roc_curve_svm_test.png` | ROC curves baseline | Good AUC for strong metals; hidden weakness unseen here. |
| `pr_curve_svm_val.png` / `pr_curve_svm_test.png` | Precision–Recall baseline | Precision stable; recall drops when threshold high. |
| `confusion_matrix_hidden_svm.png` | Hidden scenario confusion | Highlights low baseline hidden recall at default threshold. |
| `roc_curve_hidden_svm.png` | Hidden ROC | Operating point shift required for sensitivity. |
| `pr_curve_hidden_svm.png` | Hidden PR curve | Illustrates recall–precision trade‑off when lowering threshold. |
| `threshold_sweep_hidden.png` | Sweep of metrics vs threshold | Guides selection of ≈0.10 for hidden mode. |
| `pr_curve_svm_val.png` (FE/Aug not stored separately) | Combined context | FE & augmentation push curve upward (not all plots saved). |

If adding more experiments, replicate confusion/ROC/PR saving for FE and augmented models for side‑by‑side comparison.

---
## 7. Threshold Strategy
| Mode | Recommended Threshold | Rationale |
|------|-----------------------|-----------|
| Standard | 0.50 | Balanced decision for typical metal presence. |
| Hidden | 0.10 | Maximizes recall for faint elongated returns; acceptable precision (~0.86 augmented). |
| High Clutter | 0.20–0.30 | Dampens false positives when environment noisy. |

Use Batch Simulation + Hidden Lab tabs to visualize probability distributions and refine thresholds on new synthetic mixes.

---
## 8. Architecture Overview
High‑level pipeline (detailed in `deployment_design.md`):
1. Acquire raw frame.
2. Preprocess (background subtraction + denoise).
3. Extract features (stats + top‑k intensities from raw & processed).
4. SVM probability inference.
5. Adaptive threshold (standard vs hidden vs clutter context).
6. Decision + optional temporal smoothing & metrics logging.

`deployment_design.md` includes a Mermaid flowchart and monitoring suggestions (drift, false positive rate, latency).

---
## 9. Reproducing Everything Locally
### Prerequisites
Windows, Python 3.12 (or 3.10/3.11 if TensorFlow experiments desired), PowerShell.

### Clone & Environment
```powershell
git clone <your-fork-or-ssh-url> RadarGuru
cd RadarGuru/radar_project
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Run the App
```powershell
streamlit run radar_guru_app.py
```
Navigate to http://localhost:8501.

### Retrain Baseline / FE / Augmented
Open `classification_model.ipynb` and execute cells sequentially:
1. Load data paths & imports.
2. Train baseline SVM (flatten+PCA) & evaluate confusion/ROC.
3. Compute engineered features & train FE SVM.
4. Perform augmentation (hidden/clutter injection) & retrain augmented models.
5. Save models & plots to `models/`.

### Hidden Evaluation
Run `hidden_object_detection.ipynb`:
1. Generate hidden/clutter test frames.
2. Baseline evaluation & metrics JSON.
3. Threshold sweep storing `hidden_threshold_sweep.json` & plot.
4. Augmented models comparison; save improved metrics.

### Simulation Sandbox
Use `radar_simulation.ipynb` to inspect generated heatmaps and verify feature distributions visually.

### Deployment PDF
```powershell
pandoc deployment_design.md -o deployment_design.pdf
```

---
## 10. Minimal Inference Example
```python
import joblib
from radar_simulation import generate_range_doppler_heatmap, denoise_background_subtract, extract_features
model = joblib.load('models/metal_classifier_svm_fe_aug.joblib')
raw = generate_range_doppler_heatmap(64, 64, 'hidden', metal=True, clutter_level=0.25, snr_db=8)
proc = denoise_background_subtract(raw, method='median', kernel_size=5)
feat = extract_features(raw, proc, k_top=10).reshape(1,-1)
prob = model.predict_proba(feat)[0,1]
decision = prob >= 0.10  # hidden mode threshold
print(f"prob={prob:.3f} hidden_detected={decision}")
```

---
## 11. Lessons Learned
- Feature engineering dramatically outperformed naive flatten+PCA for faint signals.
- Threshold selection must be context‑aware; a single global 0.50 cutoff under‑detects hidden metals.
- Augmentation (hidden + clutter examples) stabilized recall without catastrophic precision loss.
- Classical ML (SVM) provided fast iteration vs troubleshooting deep learning environment issues.

---
## 12. Glossary (Quick)
| Term | Short Definition |
|------|------------------|
| Range | Distance bin from radar. |
| Doppler | Relative velocity bin (frequency shift). |
| Heatmap | 2D energy/intensity representation. |
| Clutter | Non‑target background reflections. |
| Hidden | Attenuated / smeared target signature. |
| Recall | Fraction of true metals correctly detected. |
| Precision | Fraction of detected metals that are true. |
| SVM | Margin‑based classifier using kernel mapping. |
| PCA | Linear dimensionality reduction retaining variance. |
| Threshold Sweep | Evaluate metrics across probability cutoffs. |

---
## 13. Roadmap / Next Steps
- Export FE & augmented plots for side‑by‑side comparisons.
- Add probability calibration (isotonic / Platt scaling).
- Introduce temporal smoothing (rolling majority vote).
- Serve model via FastAPI; add CI for lint/tests.
- Integrate drift detection on feature distributions.
- Port simulation to GPU for large batch generation.

---
## 14. File Map
| File | Purpose |
|------|---------|
| `radar_simulation.py` | Heatmap generation, preprocessing, feature extraction. |
| `classification_model.ipynb` | Train/evaluate baseline, FE, augmented SVM models. |
| `hidden_object_detection.ipynb` | Hidden/clutter evaluation & threshold sweep. |
| `radar_simulation.ipynb` | Visual inspection of synthetic frames. |
| `deployment_design.md` | Production pipeline & adaptive threshold design. |
| `radar_guru_app.py` | Streamlit UI (Single, Batch, Hidden Lab, Metrics). |
| `models/` | Saved `.joblib` models + plots + JSON metrics. |
| `data/` | Generated arrays/images for train/val/test. |

---
## 15. License & Attribution
Add an appropriate license (e.g., MIT) before distribution. Synthetic data & code for educational/demo purposes.

---
© 2025 RadarGuru
