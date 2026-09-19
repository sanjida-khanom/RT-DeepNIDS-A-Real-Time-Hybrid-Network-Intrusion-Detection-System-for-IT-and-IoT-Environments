# RT-DeepNIDS — Real-Time Deployment

Real-Time module of **RT-DeepNIDS: A Real-Time Hybrid Network Intrusion
Detection System for IT and IoT Environments**.

This directory contains the interactive **Streamlit dashboard** that serves the
trained models for live intrusion detection — the deployment layer of the
project (Novelty 05: Real-Time Deployment).

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Space-blue?logo=huggingface)](https://sadiamehrinrahi-rt-deepnids.hf.space/)

---

## Live Demo

A hosted version of the dashboard is deployed on **Hugging Face Spaces**:

** https://sadiamehrinrahi-rt-deepnids.hf.space/**

The hosted demo runs in **CSV Simulation** mode (cloud environments cannot
capture live network packets). For **live packet capture** with Scapy, run the
app locally following the setup steps below.

---

## Features

- **Two traffic sources**
  - **CSV Simulation** — replays exported test traffic through the trained
    models (reliable, matches training accuracy).
  - **Live Network Interface** — captures real packets via **Scapy**, builds
    per-flow features on the fly, and classifies them in real time.
- **All models supported** — Decision Tree, Random Forest, XGBoost,
  Hybrid CNN-GRU, and CNN+Transformer.
- **Four evaluation modes** — CIC-IDS-2017, CIC-IDS-2018, ToN-IoT-v3, and
  Cross-Domain (zero-day robustness test).
- **Cross-flow port-scan detector** — a heuristic layer that catches
  nmap / hping3 style scans before the ML model.
- **Explainable AI (SHAP)** — every attack verdict can be explained on demand.

---

## Project Structure

```
.
├── app.py               # Streamlit frontend (UI, live dashboard, SHAP panel)
├── backend.py           # Feature engineering, flow tracking, model loading
├── requirements.txt     # Python dependencies
└── Real_Time_Export/    # Trained models & scalers (NOT in git — see below)
    ├── SMOTE/
    │   ├── CIC_IDS_2017/   (RF/DT/XGB .pkl, CNN_GRU_model.h5, scaler.pkl, live_traffic_sample.csv)
    │   ├── CIC_IDS_2018/
    │   └── TON_IOT_V3/
    ├── Tomek_IHT/          (same structure as SMOTE/)
    └── Cross_Validation/   (CNN_Transformer + CNN_GRU + ML models, scaler, sample)
```

> **Note:** `Real_Time_Export/` (trained `.h5` / `.pkl` model files) is **not**
> committed to GitHub because the files exceed GitHub's size limits. Place your
> exported models in this folder locally before running.

---

## Setup & Run (Local)

### 1. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
> For **live packet capture** on Windows, also install
> [Npcap](https://npcap.com).

### 3. Run
Live capture needs administrator / root privileges:
```bash
# Windows: run terminal as Administrator, then
streamlit run app.py

# Linux / macOS
sudo streamlit run app.py
```
The dashboard opens at `http://localhost:8501`.

---

## Usage

1. In the sidebar, pick a **Monitoring Mode**, **Dataset**, **Balancing
   Technique**, and **Model**.
2. Choose a **Traffic Source**:
   - *CSV Simulation* — no admin needed, best for demos.
   - *Live Network Interface* — select your network adapter (needs admin + Npcap).
3. Press **START** to begin monitoring; watch live KPIs, threat classification,
   and the traffic log.
4. Open the **SHAP** panel to see why a verdict was made.

---

## Note on Live-Capture Accuracy

CSV Simulation reproduces the exact feature set used in training, so it matches
the reported accuracy. Live capture reconstructs flow features from raw packets;
some fields (retransmission counts, throughput aggregates, DNS/HTTP metadata)
cannot be derived from raw packets and are imputed. Live-capture predictions are
therefore approximate — an inherent limitation of all real-time NIDS, not a bug.

---

## Team

**Supervisor:** Md. Saifur Rahman, Assistant Professor, Dept. of CSE, BUBT

| Member | ID |
|--------|-----|
| Ayesha Siddika | 22234103099 |
| Sanjida Khanom | 22234103103 |
| Ihsanul Hossain Rafsan | 22234103112 |
| Sadia Mehrin Rahi | 22234103122 |
| Istiyak Hasan Maruf | 22234103130 |

Bangladesh University of Business and Technology (BUBT) — Department of Computer
Science and Engineering.
