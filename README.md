# CMAPSS Turbofan Engine Remaining Useful Life (RUL) Prediction

**Deep Learning Project for Predictive Maintenance**

This project implements an end-to-end **CNN-LSTM hybrid deep learning pipeline** to predict the Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS benchmark dataset (FD001, FD002, FD003).

---

## Results Summary

| Dataset | Operating Conditions | Fault Modes | Test Engines | **RMSE** | **MAE** | **PHM08 Score** |
|---------|----------------------|-------------|--------------|----------|---------|-----------------|
| **FD001** | 1 (Sea Level)       | 1 (HPC)    | 100          | **17.04** | 13.55  | 676.4          |
| **FD002** | 6                   | 1 (HPC)    | 259          | **29.83** | 20.46  | 11406.7        |
| **FD003** | 1 (Sea Level)       | 2 (HPC+Fan)| 100          | **15.41** | 11.79  | 744.3          |

- **FD001 & FD003** (single operating condition) achieve competitive RMSE in the mid-teens, which is solid for a clean CNN-LSTM without heavy hyper-parameter tuning or advanced domain adaptation.
- **FD002** is significantly harder (6 operating conditions). The higher error is expected; literature typically reports 20–30+ RMSE for basic models on this subset.

These numbers demonstrate practical ability to:
- Handle multivariate time-series sensor data
- Build and train deep sequential models
- Evaluate with industry-standard metrics (RMSE + NASA PHM08 asymmetric score)
- Produce clear visualizations of model behaviour

---

## Project Structure

```
cmapss_rul_project/
├── data/
│   └── raw/                  # Original CMAPSS .txt files
│       ├── train_FD00*.txt
│       ├── test_FD00*.txt
│       └── RUL_FD00*.txt
├── src/
│   ├── preprocess.py         # Loading, RUL labeling, normalization, windowing
│   ├── model.py              # CNN-LSTM and LSTM architectures (PyTorch)
│   ├── train_eval.py         # Training loop, RMSE, PHM08 score
│   └── run_experiment.py     # Full pipeline runner + visualization generation
├── models/                   # Saved .pt weights
│   ├── FD001_cnn_lstm.pt
│   ├── FD002_cnn_lstm.pt
│   └── FD003_cnn_lstm.pt
├── results/
│   ├── FD00*_metrics.json
│   ├── FD00*_predictions.csv
│   └── summary_metrics.csv
├── visualizations/
│   ├── FD00*_training_curve.png
│   └── FD00*_predictions.png   # scatter, error hist, sorted comparison
├── requirements.txt
└── README.md
```

---

## Method Highlights (No Hard-coded High/Medium Labels)

- **Piecewise-linear RUL target** capped at 125 cycles (standard in CMAPSS literature).
- **Sensor selection**: 14 informative sensors for FD001/FD003; +3 operating settings for FD002.
- **Min-Max normalization** fitted **only on training data**.
- **Sliding windows** of length 30 cycles.
- **Deep Learning model**: Hybrid **CNN-LSTM**
  - 2× Conv1D (64 → 32 filters) + BatchNorm + MaxPool for local pattern extraction
  - 2-layer LSTM (64 hidden) for temporal degradation modelling
  - Fully-connected head → continuous RUL regression
- **Training**: Adam + ReduceLROnPlateau + Early Stopping on validation RMSE.
- **Evaluation metrics**:
  - RMSE / MAE
  - NASA PHM08 Score (asymmetric: late predictions are penalized more heavily)

No rule-based or hard-coded “High / Medium / Low” health labels were used. The model learns continuous degradation directly from sensor sequences.

---

## How to Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full experiment (FD001 → FD003)
cd cmapss_rul_project
PYTHONPATH=src python src/run_experiment.py
```

Results, models and plots will be written automatically under `results/`, `models/` and `visualizations/`.

---

## Visualizations Included

For each subset you will find:

1. **Training curve** – train loss + validation RMSE over epochs (with early-stopping marker).
2. **Prediction diagnostics** (3 panels):
   - Predicted vs True RUL scatter
   - Error (Pred − True) histogram
   - Sorted true vs predicted RUL comparison across all test engines

---

## References

- Saxena & Goebel (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.
- Standard CMAPSS evaluation protocol (piecewise RUL, PHM08 score).
- Hybrid CNN-LSTM architectures widely used in recent RUL literature.

---

**Author**: Prepared as a portfolio project for data-analysis / predictive-maintenance skill demonstration for Tata ELSXI hackathon
