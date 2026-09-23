# RT-DeepNIDS

**A Real-Time Hybrid Network Intrusion Detection System for IT and IoT Environments**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-2.x-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-brightgreen" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
</p>

> Undergraduate capstone research — Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh.

---

## Overview

Signature-based intrusion detection systems only catch what they have already seen. They need constant database updates, they are blind to zero-day exploits, and in high-volume IT/IoT networks they drown analysts in false positives.

**RT-DeepNIDS** is an end-to-end, explainable NIDS framework built around a **Dual-Pipeline data optimization strategy** that resolves the long-standing conflict between *detection precision* and *inference speed*:

- **Pipeline A — High Precision (SMOTE):** synthesizes minority attack signatures for maximum accuracy in offline / cloud settings.
- **Pipeline B — High Speed (Tomek Links + Instance Hardness Threshold):** aggressively removes redundant and ambiguous benign flows, cutting training latency by roughly **10×** and making deep models viable on edge hardware.

On top of this sit hybrid deep learning engines (**CNN+GRU**, **CNN+Transformer**) and fast ensemble baselines (**XGBoost**, **Random Forest**, **Decision Tree**). The framework is validated on three benchmark corpora, stress-tested for zero-day robustness through strict cross-dataset transfer, made transparent with **SHAP**, and finally deployed as a live traffic-monitoring prototype.

---

## Key Contributions

1. **Dual-Pipeline balancing strategy** — SMOTE for precision, Tomek+IHT for speed, evaluated head-to-head in a full ablation study.
2. **Hybrid detection engine** — spatial (Conv1D), temporal (GRU) and attention-based (Transformer) architectures benchmarked against tree ensembles.
3. **~10× latency reduction** — CNN+GRU training drops from 2534s to 266s with no catastrophic accuracy loss, enabling real-time deployment.
4. **Zero-day cross-domain validation** — models trained *only* on IT traffic (merged CIC-IDS) detect unseen IoT attacks (ToN-IoT-v3) with **no retraining**.
5. **Explainable AI via SHAP** — proves decisions are driven by universal network features (`protocol`, `in_bytes`, `out_bytes`) rather than dataset-specific artifacts.
6. **RT-DeepNIDS prototype** — a lightweight interactive dashboard performing live inference, threat classification and instant alert generation.

---

## Methodology

<p align="center">
  <img src="Methodology%20diagram/Methodology%20diagram-1.png" alt="RT-DeepNIDS methodology workflow" width="850">
</p>

<p align="center"><i>Overall workflow of the proposed architecture for real-time and cross-domain network intrusion detection.</i></p>

**Phase 1 — Preprocessing.** Median imputation for missing/infinite values; removal of structural leakage columns (IP addresses, MAC addresses, port numbers) so models cannot memorize topology; variance and correlation filtering; Min-Max normalization.

**Phase 2 — Dual-Pipeline optimization.** 70/15/15 train-validation-test split, then the training partition is routed through either SMOTE (Pipeline A) or Tomek+IHT (Pipeline B).

**Phase 3 — Detection engines.** Decision Tree (CART), Random Forest, XGBoost, Hybrid CNN+GRU, CNN+Transformer.

**Phase 4 — Evaluation, explainability, deployment.** Accuracy / Precision / Recall / F1 / ROC-AUC / confusion matrices → SHAP interpretability → RT-DeepNIDS live prototype.

---

## Datasets

| Dataset | Features | Approx. Samples | Context |
|---|---|---|---|
| CIC-IDS-2017 | 78 | ~2.8 M | Network flow-based features |
| CIC-IDS-2018 | 80 | ~16 M | Updated flow-based features |
| ToN-IoT-v3 | ~470 | ~2.4 M | IoT telemetry and network logs |

For the zero-day experiment the **merged CIC-IDS corpus is the source domain** and **ToN-IoT-v3 is the target domain**.

> **Note:** the raw datasets are *not* included in this repository due to size limits. CIC-IDS-2017 and CIC-IDS-2018 are available from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/); ToN-IoT-v3 is available from the [UNSW Canberra Cyber repository](https://research.unsw.edu.au/projects/toniot-datasets). Download them and place the CSVs under `data/` as described below.

---

## Results

### Intra-domain — Pipeline A (SMOTE)

| Dataset | Model | Accuracy (%) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CIC-IDS-2017 | Decision Tree | 99.87 | 0.99 | 0.99 | 0.99 |
| CIC-IDS-2017 | Random Forest | 99.87 | 0.99 | 0.99 | 0.99 |
| CIC-IDS-2017 | **XGBoost** | **99.94** | 1.00 | 0.99 | 0.99 |
| CIC-IDS-2017 | Hybrid CNN+GRU | 99.83 | 0.99 | 0.98 | 0.99 |
| CIC-IDS-2018 | Decision Tree | 99.87 | 0.99 | 0.99 | 0.99 |
| CIC-IDS-2018 | Random Forest | 99.88 | 0.99 | 0.99 | 0.99 |
| CIC-IDS-2018 | **XGBoost** | **99.94** | 1.00 | 1.00 | 1.00 |
| CIC-IDS-2018 | Hybrid CNN+GRU | 99.83 | 0.99 | 0.98 | 0.99 |
| ToN-IoT-v3 | Decision Tree | 99.33 | 0.99 | 0.99 | 0.99 |
| ToN-IoT-v3 | **Random Forest** | **99.37** | 0.99 | 0.99 | 0.99 |
| ToN-IoT-v3 | XGBoost | 99.25 | 0.99 | 0.99 | 0.99 |
| ToN-IoT-v3 | Hybrid CNN+GRU | 98.98 | 0.98 | 0.98 | 0.98 |

### Intra-domain — Pipeline B (Tomek+IHT)

| Dataset | Model | Accuracy (%) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| CIC-IDS 2017 & 2018 | Decision Tree | 98.69 | 0.98 | 0.98 | 0.98 |
| CIC-IDS 2017 & 2018 | Random Forest | 98.76 | 0.98 | 0.98 | 0.98 |
| CIC-IDS 2017 & 2018 | XGBoost | 97.58 | 0.97 | 0.97 | 0.97 |
| CIC-IDS 2017 & 2018 | Hybrid CNN+GRU | 95.74 | 0.95 | 0.95 | 0.95 |
| ToN-IoT-v3 | Decision Tree | 99.33 | 0.99 | 0.99 | 0.99 |
| ToN-IoT-v3 | Random Forest | 99.30 | 0.99 | 0.99 | 0.99 |
| ToN-IoT-v3 | XGBoost | 99.10 | 0.99 | 0.99 | 0.99 |
| ToN-IoT-v3 | Hybrid CNN+GRU | 96.28 | 0.96 | 0.96 | 0.96 |

### Zero-day cross-dataset generalization

Trained on merged CIC-IDS (IT) → evaluated directly on unseen ToN-IoT-v3 (IoT), **no retraining, no fine-tuning**.

| Model | Accuracy (%) | Precision | Recall | F1 |
|---|---|---|---|---|
| **Random Forest** | **96.60** | 0.92 | 0.92 | 0.92 |
| XGBoost | 96.50 | 0.92 | 0.92 | 0.92 |
| Decision Tree | 95.10 | 0.92 | 0.92 | 0.92 |
| CNN+Transformer | 93.50 | 0.88 | 0.88 | 0.88 |
| Hybrid CNN+GRU | 92.30 | 0.89 | 0.89 | 0.89 |

All architectures held AUC > 0.92 across completely unseen environments.

### Ablation study — accuracy vs. latency

| Pipeline | Model | Accuracy (%) | Train Time (s) | Test Time (s) |
|---|---|---|---|---|
| A: SMOTE | Random Forest | 99.87 | 97.32 | 0.74 |
| A: SMOTE | XGBoost | 99.94 | 144.11 | 1.60 |
| A: SMOTE | Hybrid CNN+GRU | 99.83 | 2534.33 | 8.70 |
| B: Tomek+IHT | Random Forest | 98.76 | 25.77 | 0.89 |
| B: Tomek+IHT | XGBoost | 97.58 | 29.41 | 1.14 |
| B: Tomek+IHT | **Hybrid CNN+GRU** | 95.74 | **266.35** | 5.86 |

The headline trade-off: SMOTE buys near-perfect accuracy at extreme computational cost, while Tomek+IHT gives back **~9.5× faster training** for roughly 4 accuracy points — the exchange that makes edge deployment possible.

---

## Explainable AI

SHAP analysis consistently identified `protocol`, `out_bytes` and `in_bytes` as the dominant drivers of anomaly detection across both domains. This is the evidence that the cross-domain transfer is real generalization rather than coincidence: the models rely on protocol-agnostic traffic dynamics, not on memorized dataset-specific metadata.

---

## Real-Time Deployment

Beyond offline training and evaluation, RT-DeepNIDS is deployed as an interactive, explainable **Streamlit dashboard** that operationalizes the trained models for live intrusion detection. The dashboard supports two inference modes, **CSV simulation**, which replays exported test traffic through the models and **live network packet capture** via Scapy, which
extracts per-flow features from real traffic in real time. Every verdict is made transparent through on-demand **SHAP** explanations, and a cross-flow heuristic layer flags port-scan activity ahead of model inference.

- **Module & setup guide:** [`Real_Time_Deployment/`](./Real_Time_Deployment)
- **Live demo (Hugging Face Spaces):** https://sadiamehrinrahi-rt-deepnids.hf.space/

> The hosted demo runs in CSV-simulation mode, as cloud environments cannot
> capture live packets; live capture is available in the local deployment.

---

## Repository Structure

```
RT-DeepNIDS/
├── CIC-IDS-2017/                     # Pipeline, training and evaluation for CIC-IDS-2017
├── CIC-IDS-2017(Tomek+IHT vs SMOTE)/ # Dual-pipeline ablation experiments
├── CIC-IDS-2018/
│   ├── preprocess.py                 # Cleaning, leakage removal, normalization, resampling
│   ├── models.py                     # RF / DT / XGBoost / CNN+GRU / CNN+Transformer builders
│   ├── train.py                      # Training loop, evaluation, artifact export
│   └── plots/                        # Confusion matrices, ROC curves
├── TON-IOT-V3/                       # IoT telemetry experiments
├── TON-IOT-V3 FL/                    # ToN-IoT feature-level variant
├── Cross Validation pipeline/        # Zero-day cross-dataset generalization
├── Real_Time_Deployment/             # Live Streamlit dashboard (CSV sim + Scapy live capture)
│   ├── app.py                        # Streamlit frontend (UI, live dashboard, SHAP panel)
│   ├── backend.py                    # Feature engineering, flow tracking, model loading
│   ├── requirements.txt              # Deployment dependencies
│   └── README.md                     # Setup, usage and live-demo link
├── Demo plots/                       # Ablation and comparison figures
├── Methodology diagram/              # Architecture diagram assets
├── docs/                             # Manuscript, presentation, supporting material
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- A CUDA-enabled GPU is strongly recommended for the deep learning pipelines
- 16 GB RAM minimum (the reference environment used an Intel Core i7, 64 GB DDR4 and an NVIDIA RTX-series GPU)

### Installation

```bash
git clone https://github.com/<your-username>/RT-DeepNIDS.git
cd RT-DeepNIDS

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Preparing the data

Download the three benchmark datasets from the links in the [Datasets](#datasets) section and arrange them as:

```
data/
├── cic-ids-2017/
├── cic-ids-2018/
└── ton-iot-v3/
```

### Running the pipeline

```bash
# 1. Clean, normalize and balance a dataset
python preprocess.py --dataset cic-ids-2018 --pipeline smote      # or: tomek_iht

# 2. Train and evaluate a model
python train.py --dataset cic-ids-2018 --model cnn_gru --pipeline tomek_iht

# 3. Zero-day cross-domain evaluation (train on IT, test on IoT)
python train.py --source cic-ids-merged --target ton-iot-v3 --model random_forest --cross-domain
```

Metrics, confusion matrices and ROC curves are written to the corresponding `plots/` directory.

---

## Experimental Configuration

| Category | Parameter | Setting |
|---|---|---|
| Hardware | Processor / Memory / GPU | Intel Core i7 · 64 GB DDR4 · NVIDIA RTX (CUDA) |
| Software | Language & libraries | Python 3.9+ · TensorFlow, Keras, scikit-learn, XGBoost, imbalanced-learn, SHAP |
| Deep Learning | Optimizer | Adam (initial LR 0.0005) |
| Deep Learning | Loss | Categorical Cross-Entropy |
| Deep Learning | Batch size | 256–1024 (dynamically adjusted) |
| Deep Learning | Regularization | Dropout 0.2–0.4, Early Stopping |
| Ensemble ML | Estimators | up to 200 trees (Random Forest, XGBoost) |
| Ensemble ML | XGBoost depth / LR | max_depth 8 · learning_rate 0.1 |

---

## Roadmap

- [ ] **Model compression for edge computing** — quantization and pruning of the CNN+GRU and Transformer variants for on-gateway inference.
- [ ] **Big data streaming integration** — Apache Kafka / PySpark ingestion to replace the simulated live stream with real enterprise-scale traffic.
- [ ] **Adversarial threat defense** — expand the training corpus with adversarial attack datasets and harden the models against AI-generated deceptive packets.

---

## Team

| Name | Role | GitHub |
|---|---|---|
| **Sanjida Khanom** | Author · Project Lead, Architechture design and project plan preparation, Deep learning architecture, cross-domain evaluation , Writing lead| [@sanjida-khanom](https://github.com/sanjida-khanom) |
| **Isitiyak Hasan Maruf** | Author · Data preprocessing and Pipeline A (SMOTE, Tomek+IHT) for dataset CIC-IDS-2017 | [@MarufKhan-ops](https://github.com/MarufKhan-ops) |
| **Sadia Mehrin Rahi** | Author · Pipeline B (Tomek+IHT), ablation study , Real-Time-Deployment | [@SadiaMehrinRahi](https://github.com/sadia-mehrin-rahi) |
| **Ayesha Siddika** | Author · Ensemble ML baselines,Data preprocessing and Pipeline A (SMOTE) for dataset CIC-IDS-2018  | [@ayesha099-git](https://github.com/ayesha099-git) |

**Supervision & Acknowledgement**

We gratefully acknowledge the Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT) for computational resources and a supportive research environment. Special thanks to **Md. Saifur Rahman** (Assistant Professor and Chairman, CSE, BUBT) and **Dr. Md. Shafiqul Islam** (Assistant Professor, CSE, UAP) for their guidance and technical insight throughout this work.

---



## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

The benchmark datasets remain subject to the licensing terms of their original providers (Canadian Institute for Cybersecurity and UNSW Canberra Cyber).
