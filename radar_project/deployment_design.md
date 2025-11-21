# Deployment Design (Real-time Radar Pipeline)

This design document specifies a production-ready real-time radar pipeline for metal vs non‑metal object discrimination, hidden object detection support, and lifecycle/monitoring strategy. It is intentionally concise for PDF export.

---
## 1. Scope & Goals
Primary objective: ingest streaming radar returns and output a stable per-frame or per-event metal probability with actionable alerts while minimizing latency (<50 ms target on edge GPU; <120 ms CPU). Hidden/occluded targets must be surfaced by combining enhanced feature engineering and adaptive thresholds.

---
## 2. System Context
Sensors deliver bursts of IQ samples over fast time and slow time. The pipeline converts these into range–Doppler heatmaps. An application node performs preprocessing, inference, and alerting; a monitoring layer stores metadata and performance metrics.

---
## 3. Data Flow (High-Level)
1. Acquire raw IQ buffer (fast-time chirps × slow-time chirps).
2. Apply windowing (e.g., Hanning) and 1D FFT (range) per chirp; stack.
3. Apply Doppler FFT (slow-time) to form range–Doppler cube; take magnitude.
4. Log-compress, normalize, and optional dynamic range clipping.
5. Background subtraction (running median or exponential moving average) to suppress static clutter.
6. Denoise (median or light Gaussian) and optional contrast stretch.
7. Resize / pad to model input (64×64) and retain float32.
8. Feature extraction (engineered stats + top‑k peaks) OR flatten + PCA path.
9. Run inference (Feature-Engineered SVM or CNN if available) → probability p(metal).
10. Threshold decision (adaptive; default 0.10 for hidden recall; may differ for standard scenes).
11. Temporal smoothing (rolling majority vote or EWMA of probabilities over last M frames).
12. Alert generation (if smoothed probability ≥ threshold for K consecutive frames).
13. Logging & metrics emission.

---
## 4. Detailed Preprocessing Steps
| Step | Purpose | Notes |
|------|---------|-------|
| Range FFT | Distance separation | Per chirp; zero-pad to improve resolution if latency budget allows |
| Doppler FFT | Velocity separation | Window slow-time dimension to reduce leakage |
| Magnitude + Log | Stabilize dynamic range | Use log1p(mag) then normalize |
| Background Subtraction | Remove static clutter | Maintain running median map; update every N frames with decay |
| Denoising | Suppress speckle | Median 3×3; fall back to Gaussian σ≈1 if metal peak broadening acceptable |
| Normalization | Model consistency | Scale to [0,1]; optional per-frame percentile normalization |
| Resize | Standard input size | Bilinear or area resample to 64×64 |
| Feature Extraction | Robust hidden detection | Stats (mean,std,max,energy,>0.5 fraction) raw + processed + top‑k intensities |

---
## 5. Inference & Thresholding Logic
Pseudo-code decision:
```
prob = model.predict(frame_features)
prob_smoothed = 0.6*prob + 0.4*prev_prob_smoothed
dynamic_thr = base_thr * (1 - clutter_factor*0.3)  # raise threshold in high clutter?
if prob_smoothed >= dynamic_thr:
      metal_frame_counter += 1
else:
      metal_frame_counter = max(0, metal_frame_counter-1)
alert = (metal_frame_counter >= K)
```
Recommended base thresholds:
| Scenario | Base Threshold | Rationale |
|----------|----------------|-----------|
| Standard scene | 0.50 | Balanced precision/recall |
| Hidden detection mode | 0.10 | Maximizes metal recall (observed 0.86–1.00) |
| High clutter surge | 0.20–0.30 | Mitigate false positives |

Adaptive inputs: clutter_factor from mean background energy; optional SNR estimate to modulate threshold.

---
## 6. Flowchart
```mermaid
flowchart LR
   A[Raw IQ Buffer] --> B[Range FFT]\n  B --> C[Doppler FFT]\n  C --> D[Magnitude + Log Norm]\n  D --> E[Background Subtraction]\n  E --> F[Denoise / Smooth]\n  F --> G[Resize 64x64]\n  G --> H{Feature Path?}
   H -->|Engineered| I[Extract Stats + Top-k]
   H -->|CNN| J[Prepare Tensor]
   I --> K[Feature-Eng SVM Prob]
   J --> L[CNN Prob]
   K --> M[Temporal Smoothing]
   L --> M[Temporal Smoothing]
   M --> N[Adaptive Threshold]
   N --> O{Alert?}
   O -->|Yes| P[Emit Event + Log]
   O -->|No| Q[Continue Stream]
   P --> R[Metrics Store]
   Q --> R
```

---
## 7. Latency & Performance Considerations
| Component | Approx Budget (ms) | Optimization |
|-----------|--------------------|--------------|
| FFTs (Range+Doppler) | 10–20 | Use FFTW/cuFFT; pre-plan sizes |
| Background & Denoise | 3–8 | Vectorized ops; fused kernels |
| Feature Extraction | 1–2 | Precompute thresholds; avoid Python loops (NumPy) |
| SVM Inference | < 3 | Keep feature dimension small (~20) |
| CNN Inference (optional) | 10–25 | Quantization (INT8) and tensor RT acceleration |
| Smoothing & Threshold | < 1 | Simple arithmetic |
Total target < 50 ms GPU / < 120 ms CPU.

---
## 8. Reliability & Monitoring
Metrics to emit per frame / window:
- latency_ms, model_prob, smoothed_prob, dynamic_threshold
- clutter_factor, background_energy, false_alert_count
- rolling_precision_estimate (from periodic labeled calibration frames)
Health checks: alert rate spike, sustained high threshold misses, probability collapse (<0.05 for >N frames).
Fallback: if feature SVM fails (exception or drift), switch to CNN path or simpler threshold on energy + top-k peaks.

---
## 9. Limitations
| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Synthetic bias | Domain shift | Collect real data; fine-tune |
| Hidden edge cases (extreme occlusion) | Missed detections | Lower threshold + multi-frame aggregation |
| Clutter bursts (rain, machinery) | False positives | Adaptive threshold & clutter_factor scaling |
| Temperature / hardware drift | Gradual performance decay | Periodic background recalibration |
| Latency spikes under load | Delayed alerts | Back-pressure + micro-batching |

---
## 10. Improvement Roadmap
Short term:
- Probability calibration (isotonic) for stable threshold selection.
- Add 10–20% real field samples in training mix.
Medium term:
- Temporal CNN/LSTM for sequence-level classification (improves noisy single frames).
- Online drift detection (Kolmogorov–Smirnov on feature distributions).
Long term:
- Multi-sensor fusion (radar + passive RF or camera) for disambiguation.
- Active learning loop: uncertain frames queued for human labeling.

---
## 11. Security & Safety Considerations
- Validate frame dimensions & numeric ranges before inference to prevent malformed input crashes.
- Rate-limit alert emissions to avoid flooding downstream systems.
- Store only derived statistical features (optional) for privacy; raw frames rotated/dropped after retention window.

---
## 12. PDF Export Instructions
From project root in PowerShell:
```
pandoc .\radar_project\deployment_design.md -o .\radar_project\deployment_design.pdf
```
Alternative: VS Code Markdown: Right-click → Export (if extension installed). Mermaid rendering may require plugin; if absent, export ASCII fallback.

---
## 13. ASCII Fallback Flow (If Mermaid Unavailable)
```
Raw IQ -> Range FFT -> Doppler FFT -> Log/Norm -> Background Subtract -> Denoise -> Resize -> Feature Path?
    |-> Engineered Features -> SVM Prob
    |-> CNN Tensor ---------> CNN Prob
Both -> Temporal Smooth -> Adaptive Threshold -> Alert? -> Metrics
```

---
## 14. Summary
This pipeline balances recall for hidden metallic objects (low base threshold with engineered features) against precision (temporal smoothing + adaptive threshold scaling). Modular layers allow substitution of model components (SVM vs CNN) and incorporation of calibration, drift monitoring, and multi-sensor fusion.

---
*End of Document*
