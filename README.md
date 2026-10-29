# 💳 Online Payment Fraud Detection System

[![CI Pipeline](https://github.com/nandeesh2799/OnlinePaymentFraudDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/nandeesh2799/OnlinePaymentFraudDetection/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Oversampling-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-7%20Passed-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

An enterprise-grade, real-time **Financial Fraud Detection System** designed to identify fraudulent online payments, suspicious account liquidations, and transfer discrepancies. Combines an **Imbalanced-Learn SMOTE** resampled **RandomForestClassifier** with a domain-informed heuristic risk engine and an interactive high-contrast **Streamlit** dashboard.

---

## ⚡ Core Capabilities

- **Hybrid Ensemble Architecture**: Merges machine learning probability scores with domain rules (e.g., origin balance depletion ratio >= 90%, zero-destination mismatches, high-velocity cashouts).
- **Extreme Class Imbalance Mitigation**: Implements **Synthetic Minority Over-sampling Technique (SMOTE)** inside an automated `imblearn` pipeline to prevent false negatives on rare fraudulent events.
- **Deep Feature Engineering**: Computes balance deltas (`orig_delta`, `dest_delta`), transaction gaps (`orig_amount_gap`, `dest_amount_gap`), origin depletion rates, and account emptying flags.
- **Self-Healing Dataset & Pipeline**: If `data/fraud.csv` or `models/fraud_model.pkl` is absent upon startup, the system automatically synthesizes a schema-compliant dataset and initiates local training.
- **Production Containerization**: Fully reproducible Docker container with multi-stage caching and healthcheck probes on `/_stcore/health`.

---

## 🧠 Model Architecture & Signal Flow

```
  Incoming Transaction
  [ type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest ]
                           │
                           ▼
          ┌──────────────────────────────────┐
          │     Domain Feature Engineering   │
          │  - orig_delta & dest_delta       │
          │  - orig/dest amount gaps         │
          │  - origin depletion ratio        │
          └────────────────┬─────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
  ┌──────────────────────┐    ┌──────────────────────┐
  │   SMOTE + RF Model   │    │  Rule Risk Engine    │
  │   (Learned Signals)  │    │  (Heuristic Weights) │
  └──────────┬───────────┘    └──────────┬───────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
              [ Ensemble Blending Engine ]
                           │
                           ▼
            Fraud Probability & Alert Status
               (Legitimate / Fraudulent)
```

---

## 📂 Project Structure

```text
OnlinePaymentFraudDetection/
├── app.py                      # Interactive Streamlit dashboard
├── data/
│   └── fraud.csv               # Historical transactions dataset
├── models/
│   └── fraud_model.pkl         # Serialized SMOTE + RandomForest pipeline
├── src/
│   ├── preprocessing.py        # Schema validation, feature engineering, and rules
│   ├── train_model.py          # Imbalanced pipeline training with SMOTE
│   └── predict.py              # Blended inference engine and safe loader
├── tests/
│   └── test_fraud_detection.py # Comprehensive unit test suite (7 tests)
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration with healthchecks
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated test and build verification
└── requirements.txt            # Python dependencies
```

---

## 🧪 Testing & Verification

Run the test suite with pytest:

```bash
python3 -m pytest tests/ -v
```

Tests validate schema coercion, feature engineering arithmetic, domain heuristic scores, synthetic data generation, ColumnTransformer tensor shapes, SMOTE training, and inference output bounds.

---

## 🚀 Setup & Execution

### Option 1: Local Environment

```bash
# Clone the repository
git clone https://github.com/nandeesh2799/OnlinePaymentFraudDetection.git
cd OnlinePaymentFraudDetection

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model (optional, app automatically trains if model is absent)
python3 src/train_model.py

# Launch Streamlit dashboard
streamlit run app.py
```

### Option 2: Docker Containerization

Run with Docker Compose:

```bash
# Build and run container
docker compose up --build

# Open browser at http://localhost:8501
```

---

## 👨‍💻 Author & Engineering Showcase

**Nandeesh N K**  
Full-Stack Developer & AI Systems Engineer — Bengaluru, India  
- 🌐 **Interactive Portfolio:** [NandeeshOS (Live Demo)](https://nandeesh2799-portfolio.vercel.app)
- 💼 **LinkedIn:** [linkedin.com/in/nandeesh-n-k-729773270](https://linkedin.com/in/nandeesh-n-k-729773270)
- 🐙 **GitHub Profile:** [github.com/nandeesh2799](https://github.com/nandeesh2799)
- 📫 **Direct Email:** [nandeeshnk506@gmail.com](mailto:nandeeshnk506@gmail.com)

---

## 📜 License
This project is licensed under the MIT License.
