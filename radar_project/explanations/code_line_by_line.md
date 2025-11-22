# RadarGuru Code Walkthrough (Line by Line)

This document explains the core generation, classification, and hidden detection logic. The classification and hidden detection were originally implemented in notebooks; here we map notebook cells to conceptual Python segments. Each section lists source lines (or grouped logical blocks) followed by commentary.

---
## 1. Generation: `radar_simulation.py`

```python
1  """
2  Helper functions to generate synthetic 1D/2D radar-like signals and heatmaps.
3  """
4  import os
5  import numpy as np
6  from scipy import ndimage
7  import matplotlib.pyplot as plt
```
Lines 1–3: Module docstring for purpose. 4–7: Imports (filesystem, numeric, image filtering, plotting).

```python
10 _rng = np.random.default_rng(42)
```
Line 10: Module-level deterministic RNG for reproducibility.

```python
13 def _gaussian_blob(shape, center, sigma, amplitude=1.0):
14     x = np.arange(shape[1])
15     y = np.arange(shape[0])
16     xx, yy = np.meshgrid(x, y)
17     cx, cy = center
18     g = amplitude * np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * sigma ** 2))
19     return g
```
Lines 13–19: Internal helper producing a 2D Gaussian used for target blobs.

```python
22 def generate_range_doppler_heatmap(range_bins=128, doppler_bins=64,
23                                    scenario='empty', metal=False,
24                                    clutter_level=0.1, snr_db=20):
25     """Create a synthetic range-Doppler 2D heatmap."""
```
Lines 22–25: Main generator signature with adjustable size, scenario, metal flag, clutter amplitude, and SNR.

Key internal logic (summarized):
- Initialize zero heatmap (shape). Add clutter if scenario demands or clutter_level > 0.
- Decide number of objects based on scenario.
- For each object: sample random center, sigma, amplitude (boost if metal), add Gaussian blob.
- If hidden and metal: add faint elongated stripe to simulate smeared signature.
- Add thermal noise based on `snr_db` (converted to linear amplitude).
- Normalize and apply light Gaussian blur to approximate sensor PSF.

Important lines (selected):
```python
31 heatmap = np.zeros(shape, dtype=float)        # base canvas
34 clutter = _rng.normal(scale=clutter_level, size=shape); heatmap += np.abs(clutter)
43 n_objects = ...                               # scenario-dependent count
51 blob = _gaussian_blob(...); heatmap += blob   # add each target blob
57 if scenario == 'hidden' and metal: ...        # elongated weak feature
62 noise_std = 10 ** (-snr_db / 20.0)            # convert dB to linear std
63 heatmap += np.abs(_rng.normal(scale=noise_std, size=shape))
66 heatmap = heatmap / (np.max(heatmap) + 1e-12) # prevent divide-by-zero
69 heatmap = ndimage.gaussian_filter(heatmap, sigma=1.0)
```
Each normalization ensures values remain in 0..1.

```python
74 def apply_fft(signal_2d):
75     f = np.fft.fftshift(np.fft.fft2(signal_2d))
76     mag = np.abs(f)
77     mag = np.log1p(mag)
78     mag = mag / (mag.max() + 1e-12)
79     return mag
```
Lines 74–79: Simulated transform to produce log-magnitude FFT (for visualization only).

```python
82 def visualize_heatmap(heatmap, title=None, cmap='inferno', show=True, savepath=None):
83     plt.figure(figsize=(5, 4))
84     plt.imshow(heatmap, aspect='auto', origin='lower', cmap=cmap)
85     plt.colorbar(label='Normalized amplitude')
86     ...
91     if savepath: ... plt.savefig(...)
94     if show: plt.show() else: plt.close()
```
Lines 82–94: Flexible plotting helper used during dataset creation & CLI demo.

```python
97 def generate_dataset(out_dir, n_samples=300, img_size=(64, 64), seed=42):
```
Generates balanced metal/non-metal dataset:
- Half metal: scenarios biased toward `metal_object`.
- Half non-metal: mix of `empty` and `clutter`.
- Saves `.npy` arrays + PNG previews + CSV manifest.

Important lines:
```python
107 half = n_samples // 2
111 if i < half: label = 1 ... else: label = 0
118 heatmap = generate_range_doppler_heatmap(...)
121 np.save(npy_path, heatmap.astype(np.float32))
123 visualize_heatmap(... show=False, savepath=png_path)
131 writer.writerow(['npy_path','png_path','label'])
```

```python
136 def denoise_background_subtract(heatmap, method='median', kernel_size=5):
141 bg = ndimage.median_filter(...) or gaussian_filter(...)
142 proc = heatmap - bg
143 proc = proc - proc.min(); proc /= proc.max() (if >0)
146 return proc
```
Median/gaussian filtering acts as background model; subtracting emphasizes targets.

Range profile & Doppler spectrum 1D functions (lines ~149–192) mirror 2D logic with single dimension arrays for variety.

```python
194 def extract_features(raw_heatmap, processed_heatmap=None, k_top=10):
```
Feature vector includes:
- Raw stats (mean/std/max/energy/fraction > 0.5)
- Processed stats (same, or zeros if absent)
- Top-k intensities from processed (or raw fallback) sorted descending.
Pad if fewer than k available.

```python
214 if __name__ == '__main__':
215   # CLI argument parsing, generate one demo heatmap & FFT, save images
```
Allows quick standalone testing.

---
## 2. Classification Workflow (Notebook `classification_model.ipynb`)

The notebook trains baseline and engineered SVM models, plus augmented variants. Below is a distilled Python equivalent with line commentary.

### 2.1 Data Loading & Splitting
```python
# Load or generate datasets
project_root = os.path.abspath('.')                    # base path
trainval_dir = os.path.join(project_root,'data','trainval')
if not os.path.exists(trainval_csv): generate_dataset(...)
X_trainval, y_trainval = load_from_csv(trainval_csv)   # load arrays
X_train, X_val, y_train, y_val = train_test_split(... stratify=y_trainval)
X_test, y_test = load_from_csv(test_csv)               # independent test set
```
Generates 400 train/val if missing; uses stratified split for balanced classes; holds out 100 test for final metrics.

### 2.2 Baseline SVM
```python
X_train_flat = X_train.reshape(N_train, -1)            # flatten H*W
svm_clf = Pipeline([
  ('scaler', StandardScaler(with_mean=True)),          # normalize per feature
  ('pca', PCA(n_components=0.95, svd_solver='full')),  # retain 95% variance
  ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
])
svm_clf.fit(X_train_flat, y_train)                     # train
svm_prob_val = svm_clf.predict_proba(X_val_flat)[:,1]  # probability of metal
svm_pred_val = svm_clf.predict(X_val_flat)             # hard labels
cm_val = confusion_matrix(y_val, svm_pred_val)         # validation confusion
```
Key metrics (accuracy, ROC, PR) are computed for both val and test; saved to `models/` as PNG plus serialized model via `joblib`.

### 2.3 Feature-Engineered SVM
```python
proc = denoise_background_subtract(h, method='median', kernel_size=5)   # background removal
fv = extract_features(h, proc, k_top=10)                                # build feature vector
svm_fe = Pipeline([
  ('scaler', StandardScaler()),
  ('svc', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
])
svm_fe.fit(X_train_fe, y_train)                                         # train on engineered features
```
Removes PCA since feature count is already compact. Uses same RBF SVM.

### 2.4 Augmentation for Hidden & Clutter
```python
for i in range(n_hidden_metal): generate_range_doppler_heatmap(... 'hidden', metal=True)   # positive additions
for i in range(n_clutter_nonmetal): generate_range_doppler_heatmap(... 'clutter', metal=False)  # negative additions
X_train_aug = concat(original_train, aug_proc)                         # use processed maps for flatten variant
svm_clf_aug.fit(X_train_aug_flat, y_train_aug)                         # retrain original pipeline
fe_train_aug = build_engineered_block(aug_raw, aug_proc)               # features for FE variant
svm_fe_aug.fit(X_train_fe_aug, y_train_fe_aug)                         # retrain FE pipeline
```
Augmentation increases representation of hidden signatures to boost recall.

### 2.5 Validation of Augmented Models
Predict on untouched validation set to avoid contamination; observe accuracy & confusion matrices to confirm no catastrophic overfitting.

Artifacts written:
- `metal_classifier_svm.joblib` (baseline)
- `metal_classifier_svm_fe.joblib` (feature engineered)
- `metal_classifier_svm_aug.joblib` (augmented original)
- `metal_classifier_svm_fe_aug.joblib` (augmented FE)

---
## 3. Hidden Detection Workflow (Notebook `hidden_object_detection.ipynb`)

Evaluates baseline, performs threshold sweep, and compares augmented models.

### 3.1 Baseline Hidden vs Clutter Evaluation
```python
clf = joblib.load('models/metal_classifier_svm.joblib')  # baseline SVM
for hidden samples: hm = generate_range_doppler_heatmap(... 'hidden', metal=True)
proc = denoise_background_subtract(hm, 'median')         # process
X_flat = X_proc.reshape(N, -1)                           # flatten for SVM pipeline
prob = clf.predict_proba(X_flat)[:,1]
pred = (prob >= 0.5).astype(int)                         # standard threshold
cm = confusion_matrix(y, pred)                           # reveals low hidden recall
```
Plots (confusion, ROC, PR) saved; baseline metrics persisted as JSON.

### 3.2 Threshold Sweep (Original vs Feature-Engineered)
```python
orig_prob = orig_clf.predict_proba(X_flat)[:,1]          # probabilities original model
fe_prob   = fe_clf.predict_proba(fe_X)[:,1]              # probabilities FE model
for t in np.linspace(0.05, 0.95, 19):                    # sweep thresholds
  pred = (prob >= t).astype(int)
  precision_recall_fscore_support(...)                   # compute precision/recall/F1
best = max(results, key=lambda r: r['f1'])               # pick best F1 threshold
```
Saves `hidden_threshold_sweep.json` and `threshold_sweep_hidden.png` summarizing trade-offs. FE model shows higher recall at lower thresholds.

### 3.3 Augmented Models Evaluation
```python
orig_aug = joblib.load('metal_classifier_svm_aug.joblib')
fe_aug   = joblib.load('metal_classifier_svm_fe_aug.joblib')
prob_orig_aug = orig_aug.predict_proba(X_flat)[:,1]
pred_orig_aug = (prob_orig_aug >= 0.10)                  # chosen threshold for recall
prob_fe_aug = fe_aug.predict_proba(fe_X)[:,1]
pred_fe_aug = (prob_fe_aug >= 0.10)
cm_orig_aug = confusion_matrix(y, pred_orig_aug)
cm_fe_aug   = confusion_matrix(y, pred_fe_aug)
```
Metrics stored in `hidden_augmented_metrics.json` to show improved recall (≈1.00) with acceptable precision (≈0.86) at threshold 0.10.

---
## 4. Summary of Responsibilities
- `radar_simulation.py`: Data synthesis, preprocessing, feature extraction primitives, CLI demo.
- Classification Notebook: Dataset build, baseline SVM, feature engineering, augmentation, artifact saving.
- Hidden Detection Notebook: Baseline hidden performance, threshold analysis, augmented evaluation.

---
## 5. Suggested Next Step
If you want this as fully runnable `.py` scripts (extracted from notebooks) I can generate `classification_workflow.py` and `hidden_detection_workflow.py` next, each with inline comments.

---
© 2025 RadarGuru
