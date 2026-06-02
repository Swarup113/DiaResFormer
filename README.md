# DiaResFormer: Explainable Diabetes Prediction System

## Overview

DiaResFormer is an Explainable Artificial Intelligence (XAI)-driven diabetes prediction framework developed for early diabetes screening in women. The framework combines the strengths of TabResNet and FT-Transformer architectures to capture complex feature interactions within tabular healthcare data while maintaining interpretability through multiple XAI techniques.

The system was developed using multiple diabetes datasets collected from different geographical and clinical settings and includes a web-based decision support interface for real-world usability.

---

## Live Demo

The application is hosted on HuggingFace Spaces: https://huggingface.co/spaces/dewanjee/DiaResFormer

---
## Key Features

- Hybrid TabResNet + FT-Transformer architecture (ResFormer)
- Multi-dataset diabetes prediction framework
- Synthetic data generation using Hybrid TVAE-CTGAN
- Extensive benchmarking against traditional ML and modern tabular DL models
- Multiple Explainable AI (XAI) techniques
- Cross-domain and external-domain validation
- Web-based decision support system
- Confidence score and explainability visualization support

---

## System Summary

| Component      | Description                                        |
| -------------- | -------------------------------------------------- |
| Model          | ResFormer (TabResNet + FT-Transformer)             |
| Task           | Binary diabetes prediction                         |
| Data Sources   | Frankfurt, PIMA, Pabna datasets                    |
| Interface      | Flask-based web application                        |
| Explainability | SHAP, PFI, ALE, Counterfactuals, Attention Rollout |

---

## Dataset Information

| Dataset   | Region     | Samples | Diabetic | Non-Diabetic |
| --------- | ---------- | ------- | -------- | ------------ |
| Frankfurt | Germany    | 2000    | 684      | 1316         |
| PIMA      | USA        | 768     | 268      | 500          |
| Pabna     | Bangladesh | 465     | 372      | 93           |

---

## Data Processing Pipeline

| Step           | Method                                     |
| -------------- | ------------------------------------------ |
| Data Cleaning  | Removal of invalid zero medical values     |
| Outliers       | IQR + Isolation Forest (intersection rule) |
| Missing Values | MICE / KNN Imputation                      |
| Scaling        | Standardization                            |
| Validation     | Statistical consistency checks             |

---

## Synthetic Data Generation

To address imbalance and improve robustness, multiple generative models were tested:

| Method              |
| ------------------- |
| SMOTE-ENN           |
| TVAE                |
| CTGAN               |
| TabDDPM             |
| Hybrid TVAE + CTGAN |

The **Hybrid TVAE–CTGAN** approach produced the most realistic and balanced synthetic samples based on statistical similarity metrics (KS-test, JSD, Wasserstein distance) and TSTR evaluation.

---

## Model Performance

### Benchmark Models

| Category  | Models                                                          |
| --------- | --------------------------------------------------------------- |
| ML Models | Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost |
| DL Models | TabNet, TabResNet, SAINT, FT-Transformer, TabPFN                |
| Proposed  | **ResFormer (Hybrid Model)**                                    |

---

## Final Results

| Dataset   | Accuracy (ResFormer) |
| --------- | -------------------- |
| Frankfurt | >98%                 |
| PIMA      | >98%                 |
| Pabna     | >98%                 |

The model showed consistent performance across all datasets with balanced precision, recall, and AUROC.

---

## Proposed ResFormer Architecture
ResFormer (TabResNet + FTTransformer Hybrid)

The proposed architecture achieved comparatively balanced and consistent performance across multiple evaluation metrics.

### ResFormer combines:

- Residual learning capabilities of TabResNet
- Attention mechanisms of FT-Transformer

### This hybrid design enables:

- Better feature representation
- Improved modeling of tabular relationships
- Enhanced generalization across datasets

## Model Architecture

| Hyperparameter     | Value |
| ------------------ | ----- |
| Token Dimension    | 128   |
| Residual Blocks    | 2     |
| Transformer Blocks | 2     |
| Attention Heads    | 4     |
| Dropout            | 0.1   |
| Optimizer          | AdamW |
| Learning Rate      | 3e-4  |
| Batch Size         | 256   |
| Epochs             | 150   |

---

## Explainability (XAI)

| Method            | Purpose                       |
| ----------------- | ----------------------------- |
| SHAP              | Feature contribution analysis |
| PFI               | Global feature importance     |
| ALE               | Feature effect visualization  |
| Counterfactuals   | “What-if” reasoning           |
| Attention Rollout | Transformer interpretability  |

---

## Cross & External Validation

| Test Type           | Description                          |
| ------------------- | ------------------------------------ |
| Cross-domain        | Train on one dataset, test on others |
| External validation | Tested on unseen clinical datasets   |

Results showed strong generalization across different population distributions.

---

## Web Application

A Flask-based web system was developed for real-time prediction and explanation.

| Feature             | Description                  |
| ------------------- | ---------------------------- |
| Input Form          | Clinical feature entry       |
| Prediction          | Diabetes risk output         |
| Confidence Score    | Model certainty              |
| Explainability View | Feature-level interpretation |

---

## Tech Stack

Python • Flask • PyTorch • Scikit-learn • NumPy • Matplotlib • XAI Toolkits

---

## Note

This system is a decision-support tool for early diabetes risk screening. It is not a replacement for clinical diagnosis.
