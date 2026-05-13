# Ventilator-Weaning-in-ARDS-Code
History-Conditioned Temporal Machine Learning for Personalized Ventilator Weaning in ARDS
## 📁 Project Files

This project is organized into **four Python modules**, each representing one major stage of the Temporal Double Machine Learning pipeline.  
You can keep them separate or merge them into a single file depending on your workflow.

### 🔹 1. `temporal_dml_pipeline_part1.py`
Contains:
- Full import stack  
- Global configuration  
- Database connection  
- ARDS cohort extraction  
- Time‑series & lab extraction from MIMIC‑IV  

### 🔹 2. `temporal_dml_pipeline_part2.py`
Contains:
- Temporal feature engineering  
- Resampling  
- Rolling windows  
- Trend extraction  
- Weaning‑trial detection  
- Missing‑data handling (Median, MICE, Indicator)  
- Neural models (LSTM, TCN)  
- Training loop for temporal networks  

### 🔹 3. `temporal_dml_pipeline_part3.py`
Contains:
- Ensemble nuisance estimator  
  - XGBoost  
  - Random Forest  
  - LSTM  
  - TCN  
  - Ridge meta‑learner  
- Propensity score estimator  

### 🔹 4. `temporal_dml_pipeline_part4.py`
Contains:
- Temporal Double Machine Learning (CausalForestDML)  
- HCA estimation  
- Evaluation utilities (AUROC, AUPRC, Brier)  
- Serialization helpers  
- Main guard  

---

## 📌 Combined Version

If preferred, all four parts can be merged into a single file:

