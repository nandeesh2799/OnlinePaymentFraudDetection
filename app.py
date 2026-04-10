"""
Streamlit UI for Online Payment Fraud Detection.

IMPORTANT:
- UI layout is kept stable; only backend/model logic is standardized and fixed.
"""

from __future__ import annotations

import traceback

import streamlit as st

from src.predict import load_model, predict_transaction


@st.cache_resource(show_spinner=False)
def _get_model():
    return load_model()


def _render_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg-main: #142b63;
            --bg-deep: #0f2355;
            --bg-panel: rgba(17, 34, 78, 0.88);
            --bg-input: rgba(62, 51, 29, 0.55);
            --border-soft: rgba(90, 140, 220, 0.18);
            --border-bright: rgba(49, 196, 255, 0.35);
            --text-main: rgba(255,255,255,0.96);
            --text-muted: rgba(214, 225, 245, 0.68);
            --cyan: #2bd7ff;
            --cyan-soft: #1fb9ff;
          }
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(31, 185, 255, 0.08), transparent 26%),
              linear-gradient(180deg, #122a61 0%, #11295f 48%, #10265b 100%);
            color: var(--text-main);
          }
          [data-testid="stHeader"] {
            background: transparent;
          }
          /* Hide Streamlit chrome (menu, deploy, toolbar) */
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }
          header { visibility: hidden; }
          [data-testid="stToolbar"] { display: none !important; }
          [data-testid="stDeployButton"] { display: none !important; }
          [data-testid="stDecoration"] { display: none !important; }
          [data-testid="stToolbar"] {
            right: 1rem;
          }
          [data-testid="stMainBlockContainer"] {
            padding-top: 1.4rem;
            padding-bottom: 1.8rem;
            max-width: 96rem;
          }
          .card {
            background: var(--bg-panel);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 18px 18px 14px 18px;
            backdrop-filter: blur(10px);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
          }
          .card h2, .card h3 { margin-top: 0; }
          .pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(255,255,255,0.10);
            margin-right: 6px;
            margin-bottom: 6px;
          }
          .muted { color: var(--text-muted); font-size: 13px; }
          .success { color: #34d399; font-weight: 700; }
          .danger { color: #f87171; font-weight: 700; }
          .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
          div[data-testid="stForm"] {
            background: var(--bg-panel);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 1rem 1rem 0.8rem 1rem;
          }
          .stSelectbox label, .stNumberInput label {
            color: var(--text-muted) !important;
            font-size: 0.78rem !important;
          }
          .stSelectbox div[data-baseweb="select"] > div,
          .stNumberInput div[data-baseweb="input"] > div {
            background: var(--bg-input) !important;
            border: 1px solid rgba(255,255,255,0.09) !important;
            border-radius: 9px !important;
            color: var(--text-main) !important;
            min-height: 40px;
          }
          .stSelectbox input, .stNumberInput input {
            color: var(--text-main) !important;
          }
          .stSelectbox svg {
            fill: rgba(255,255,255,0.75) !important;
          }
          .stFormSubmitButton > button,
          .btn-primary > button {
            background: linear-gradient(180deg, rgba(63, 69, 84, 0.9), rgba(48, 54, 69, 0.9)) !important;
            border: 1px solid rgba(255, 255, 255, 0.10) !important;
            color: white !important;
            border-radius: 10px !important;
            box-shadow: none !important;
          }
          .stFormSubmitButton > button:hover,
          .btn-primary > button:hover {
            border-color: rgba(49, 196, 255, 0.28) !important;
            color: white !important;
          }
          /* Sample buttons (right panel) */
          .sample-wrap .stButton > button,
          [data-testid="stButton"] > button[kind="secondary"] {
            width: 100%;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            background: linear-gradient(90deg, rgba(44, 124, 255, 0.95), rgba(27, 203, 255, 0.92)) !important;
            border: 1px solid rgba(82, 215, 255, 0.22) !important;
            color: white !important;
            font-weight: 600 !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
          }
          .sample-wrap .stButton > button:hover,
          [data-testid="stButton"] > button[kind="secondary"]:hover {
            filter: brightness(1.06);
            color: white !important;
          }
          .result-bar {
            display:flex;
            align-items:center;
            gap:10px;
            padding: 10px 14px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(2, 132, 199, 0.14);
          }
          .result-bar.ok {
            background: rgba(22, 163, 74, 0.14);
            border-color: rgba(52, 211, 153, 0.22);
          }
          .result-bar.bad {
            background: rgba(185, 28, 28, 0.16);
            border-color: rgba(248, 113, 113, 0.22);
          }
          .result-icon {
            width: 22px;
            height: 22px;
            border-radius: 6px;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight: 900;
            color: #0b1220;
          }
          .result-icon.ok { background: #34d399; }
          .result-icon.bad { background: #f87171; }
          .result-title { color: rgba(226,232,240,0.95); font-weight: 700; font-size: 14px; }
          .result-sub { color: rgba(226, 232, 240, 0.65); font-size: 12px; margin-top: 6px; }
          hr {
            border-color: rgba(102, 163, 255, 0.12) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Online Payment Fraud Detection", layout="wide")
    _render_css()

    st.markdown(
        """
        <div class="card">
          <div style="display:flex; align-items:center; gap:12px;">
            <div style="width:34px; height:34px; border-radius:10px; background: linear-gradient(135deg,#fbbf24,#f97316);"></div>
            <div>
              <div style="font-size:22px; font-weight:800; color:white;">Online Payment Fraud Detection</div>
              <div class="muted">Enter transaction details and get an instant model prediction using the trained fraud classifier.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    left, center, right = st.columns([1.05, 1.7, 1.25], gap="large")

    with left:
        st.markdown(
            """
            <div class="card">
              <h3 style="color:#22d3ee; margin-bottom: 8px;">Input Terms Guide</h3>
              <div class="muted" style="margin-bottom: 10px;">
                Transaction Category ("type")
              </div>
              <div>
                <span class="pill" style="background: rgba(16,185,129,0.12); color:#34d399;">PAYMENT</span>
                <span class="pill" style="background: rgba(59,130,246,0.12); color:#60a5fa;">TRANSFER</span>
                <span class="pill" style="background: rgba(249,115,22,0.12); color:#fb923c;">CASH_OUT</span>
                <span class="pill" style="background: rgba(168,85,247,0.12); color:#c084fc;">DEBIT</span>
              </div>
              <div class="muted" style="margin-top: 10px;">
                <div><b style="color:#34d399;">PAYMENT</b>: regular payment (bill, purchase, recharge)</div>
                <div><b style="color:#60a5fa;">TRANSFER</b>: money transfer from one account to another</div>
                <div><b style="color:#fb923c;">CASH_OUT</b>: withdrawal-like outgoing transfer</div>
                <div><b style="color:#c084fc;">DEBIT</b>: direct account debit transaction</div>
              </div>
              <hr style="border-color: rgba(255,255,255,0.08); margin: 14px 0;">
              <div class="muted"><b style="color: rgba(226,232,240,0.9);">Amount and Balances</b></div>
              <div class="muted" style="margin-top: 6px;">
                <div><span class="mono">amount</span>: money moved in this transaction</div>
                <div><span class="mono">oldbalanceOrg</span>: sender balance before transaction</div>
                <div><span class="mono">newbalanceOrig</span>: sender balance after transaction</div>
                <div><span class="mono">oldbalanceDest</span>: receiver balance before transaction</div>
                <div><span class="mono">newbalanceDest</span>: receiver balance after transaction</div>
              </div>
              <hr style="border-color: rgba(255,255,255,0.08); margin: 14px 0;">
              <div class="muted"><b style="color: rgba(226,232,240,0.9);">Quick Logic Check</b></div>
              <div class="muted" style="margin-top: 6px;">
                For outgoing transactions, sender balance usually decreases. Large transfers with unusual balance changes can indicate fraud risk.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    sample_inputs = [
        (
            "Sample 1 - High Transfer",
            {
                "type": "TRANSFER",
                "amount": 125000.50,
                "oldbalanceOrg": 300000.00,
                "newbalanceOrig": 174999.50,
                "oldbalanceDest": 50000.00,
                "newbalanceDest": 170000.00,
            },
        ),
        (
            "Sample 2 - Small Payment",
            {
                "type": "PAYMENT",
                "amount": 50.00,
                "oldbalanceOrg": 5000.00,
                "newbalanceOrig": 4950.00,
                "oldbalanceDest": 1000.00,
                "newbalanceDest": 1050.00,
            },
        ),
        (
            "Sample 3 - Cash Out",
            {
                "type": "CASH_OUT",
                "amount": 2500.00,
                "oldbalanceOrg": 5000.00,
                "newbalanceOrig": 2500.00,
                "oldbalanceDest": 0.00,
                "newbalanceDest": 0.00,
            },
        ),
        (
            "Sample 4 - Fraud Prone",
            {
                "type": "TRANSFER",
                "amount": 85000.00,
                "oldbalanceOrg": 90000.00,
                "newbalanceOrig": 5000.00,
                "oldbalanceDest": 10000.00,
                "newbalanceDest": 95000.00,
            },
        ),
    ]

    # State: selected sample
    if "selected_sample" not in st.session_state:
        st.session_state.selected_sample = 1
    if "form" not in st.session_state:
        st.session_state.form = sample_inputs[0][1].copy()

    with right:
        st.markdown(
            """
            <div class="card">
              <h3 style="color:#22d3ee; margin-bottom: 8px;">Sample Inputs</h3>
              <div class="muted" style="margin-bottom: 12px;">Choose one sample to auto-fill the form.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='sample-wrap'>", unsafe_allow_html=True)
        for idx, (name, payload) in enumerate(sample_inputs, start=1):
            clicked = st.button(name, key=f"sample_{idx}", use_container_width=True, type="secondary")
            if clicked:
                st.session_state.selected_sample = idx
                st.session_state.form = payload.copy()
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.selected_sample is not None:
            payload = sample_inputs[st.session_state.selected_sample - 1][1]
            lines = "\n".join([f"{k}: {v}" for k, v in payload.items()])
            st.markdown(f"<div class='muted mono'>{lines.replace(chr(10), '<br/>')}</div>", unsafe_allow_html=True)
            st.markdown("<div class='muted' style='margin-top:10px; font-style: italic;'>Tip: use different samples to compare model confidence.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>Pick a sample to preview inputs here.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with center:
        st.markdown(
            """
            <div class="card">
              <h3 style="color:white; margin-bottom: 6px;">Transaction Details</h3>
              <div class="muted" style="margin-bottom: 12px;">Edit the values and click Predict.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        form = st.session_state.form
        with st.form("txn_form", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                type_val = st.selectbox(
                    "Transaction type",
                    options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"],
                    index=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"].index(form.get("type", "TRANSFER")),
                )
            with c2:
                newbalanceOrig = st.number_input("New balance (Org)", value=float(form.get("newbalanceOrig", 0.0)))

            c3, c4 = st.columns(2)
            with c3:
                amount = st.number_input("Amount", value=float(form.get("amount", 0.0)))
            with c4:
                oldbalanceDest = st.number_input("Old balance (Dest)", value=float(form.get("oldbalanceDest", 0.0)))

            c5, c6 = st.columns(2)
            with c5:
                oldbalanceOrg = st.number_input("Old balance (Org)", value=float(form.get("oldbalanceOrg", 0.0)))
            with c6:
                newbalanceDest = st.number_input("New balance (Dest)", value=float(form.get("newbalanceDest", 0.0)))

            st.session_state.form = {
                "type": type_val,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
            }

            submitted = st.form_submit_button("Predict")

        st.write("")

        if submitted:
            try:
                model = _get_model()
                res = predict_transaction(
                    model,
                    type=type_val,
                    amount=float(amount),
                    oldbalanceOrg=float(oldbalanceOrg),
                    newbalanceOrig=float(newbalanceOrig),
                    oldbalanceDest=float(oldbalanceDest),
                    newbalanceDest=float(newbalanceDest),
                )

                threshold = 0.4
                confidence = max(res.probability, 1.0 - res.probability)

                if res.prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-bar bad">
                          <div class="result-icon bad">!</div>
                          <div class="result-title">Fraudulent Transaction</div>
                        </div>
                        <div class="result-sub">Threshold used: {threshold:.4f}</div>
                        <div class="muted" style="margin-top: 10px;">Fraud probability: <b>{res.probability:.6f}</b> ({res.probability*100:.2f}%)</div>
                        <div class="muted">Confidence: <b>{confidence:.4f}</b></div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-bar ok">
                          <div class="result-icon ok">✓</div>
                          <div class="result-title">Legitimate Transaction</div>
                        </div>
                        <div class="result-sub">Threshold used: {threshold:.4f}</div>
                        <div class="muted" style="margin-top: 10px;">Fraud probability: <b>{res.probability:.6f}</b> ({res.probability*100:.2f}%)</div>
                        <div class="muted">Confidence: <b>{confidence:.4f}</b></div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error("Prediction failed. Please make sure the model is trained and inputs are valid.")
                with st.expander("Error details"):
                    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))

        st.write("")
        st.markdown(
            """
            <div class="card">
              <div class="muted"><b style="color:#93c5fd;">Ready to predict?</b> Enter the transaction details and click Predict.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()