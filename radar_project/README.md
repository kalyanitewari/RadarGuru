# RadarGuru – Synthetic Radar Metal Detection

**Tagline:** *RadarGuru: Dekho Jo Chhupa Hai* (See what’s hidden)

This project simulates a real-time radar pipeline for detecting metallic objects (including partially hidden ones) using synthetic range–Doppler heatmaps, feature engineering, and classical ML models.

## Core Capabilities
- Synthetic heatmap generation for scenarios: `empty`, `metal_object`, `clutter`, `hidden`.
- Dataset creation (balanced metal / non-metal) with saved `.npy` arrays and preview PNGs.
- Classification models: Original SVM (flatten+PCA), Feature-Engineered SVM, Augmented variants with hidden/clutter samples.
- Hidden object evaluation: recall vs false positives with threshold sweeps.
- Streamlit frontends: `streamlit_app.py` (generic demo) & `radar_guru_app.py` (branded, advanced tabs & threshold suggestion).
- Deployment design document (`deployment_design.md`) outlining pipeline, thresholds, monitoring.

## File Map
| File | Purpose |
|------|---------|
| `radar_simulation.py` | Heatmap generation, preprocessing, feature extraction |
| `classification_model.ipynb` | Dataset load, train base + FE + augmented models |
| `hidden_object_detection.ipynb` | Baseline + improved hidden/clutter evaluation & threshold sweeps |
| `deployment_design.md` | Production pipeline & flowchart (convertible to PDF) |
| `streamlit_app.py` | Original interactive demo |
| `radar_guru_app.py` | Branded RadarGuru interface (Single, Batch, Hidden Lab, Export) |
| `models/` | Saved `.joblib` models & evaluation plots |
| `data/` | Generated training, test, and hidden evaluation artifacts |

## Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Run Streamlit (RadarGuru)
```powershell
streamlit run radar_guru_app.py
```
Open browser at http://localhost:8501.

### Jupyter Notebooks
```powershell
python -m notebook
```

## Using the Models
Saved models (if present):
- `metal_classifier_svm.joblib` (original)
- `metal_classifier_svm_fe.joblib` (feature-engineered)
- `metal_classifier_svm_aug.joblib` (augmented original)
- `metal_classifier_svm_fe_aug.joblib` (augmented FE)

Example inference snippet:
```python
import numpy as np, joblib
from radar_simulation import generate_range_doppler_heatmap, denoise_background_subtract, extract_features
model = joblib.load('models/metal_classifier_svm_fe_aug.joblib')
hm = generate_range_doppler_heatmap(64,64,'hidden', metal=True, clutter_level=0.25, snr_db=8)
proc = denoise_background_subtract(hm, method='median', kernel_size=5)
fv = extract_features(hm, proc, k_top=10).reshape(1,-1)
prob = model.predict_proba(fv)[0,1]
print('Probability metal:', prob)
```

## Threshold Guidance
| Mode | Threshold | Effect |
|------|-----------|--------|
| Standard | 0.50 | Balanced precision/recall |
| Hidden | 0.10 | Maximizes hidden recall |
| High Clutter | 0.20–0.30 | Reduces false positives |

Use the **Batch Simulation** and **Hidden Mode Lab** tabs in `radar_guru_app.py` to empirically tune thresholds.

## Exporting Deployment PDF
```powershell
pandoc deployment_design.md -o deployment_design.pdf
```

## Roadmap (Optional Enhancements)
- Probability calibration (isotonic) for stable thresholds.
- Temporal modeling (LSTM / TCN) for multi-frame aggregation.
- Real capture ingestion and domain adaptation.
- FastAPI microservice for /predict + /metrics endpoints.

## License & Attribution
Provide license terms here (e.g., MIT). Synthetic data generation logic and RadarGuru UI provided as demonstration code.

---
© 2025 RadarGuru
