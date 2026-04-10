# Online Payment Fraud Detection

Streamlit web app + clean ML backend for detecting potentially fraudulent online payment transactions.

## Project overview

This project trains a fraud detection model on `data/fraud.csv` (if present). If the dataset is missing, it automatically generates a small synthetic dataset with the same schema so the project stays runnable end-to-end.

## Folder structure

```
fraud/
├── app.py                 # Streamlit UI (kept stable)
├── data/
│   └── fraud.csv          # Dataset (optional; auto-generated if missing)
├── models/
│   └── fraud_model.pkl    # Saved trained pipeline (preprocess + model)
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── preprocessing.py   # Shared preprocessing (single source of truth)
│   ├── train_model.py     # Train + save RandomForest with SMOTE
│   └── predict.py         # Load model + predict (same preprocessing)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train model

```bash
python src/train_model.py
```

This saves the trained model pipeline to `models/fraud_model.pkl`.

## Run Streamlit app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).
