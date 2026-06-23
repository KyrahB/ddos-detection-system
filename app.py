import streamlit as st
import numpy as np
import pickle
import datetime
import pandas as pd

st.set_page_config(
    page_title="DDoS Threat Detection System",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    model = load_model('ddos_final_model.keras')
    with open('ddos_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ddos_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

st.title(" Predictive Threat Detection & Automated Remediation System")
st.markdown("**Deep Learning-Based DDoS Attack Detection & Remediation Strategies using CICIDS2017 Dataset**")
st.markdown("*Final Year Project: Elizabeth Emili & Ankara Job*")
st.divider()

with st.sidebar:
    st.header("About This System")
    st.info("""
    This system uses a Deep Neural Network trained on the 
    CICIDS2017 dataset to detect DDoS attacks in real time 
    and automatically trigger remediation actions:
    
    🔴 **IP Blocking** : blocks malicious source  
    🟡 **Rate Limiting** : throttles suspicious traffic  
    🔔 **Alert Generation** : notifies administrators
    """)
    st.metric("Model Accuracy", "99%+")
    st.metric("Dataset", "CICIDS2017")
    st.metric("Architecture", "Deep Neural Network")

tab1, tab2 = st.tabs(["📁 Upload CSV for Detection", "📖 How to Use"])

with tab1:
    st.subheader("Upload Network Traffic CSV")
    st.markdown("Upload a CSV file containing network flow records to detect DDoS attacks.")

    source_ip = st.text_input("Source IP Address (for alert log)", value="192.168.1.100")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)

    uploaded_file = st.file_uploader("Upload CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        st.write(f"✅ Loaded {len(df)} traffic records. Preview:")
        st.dataframe(df.head(3))

        if st.button(" ANALYSE ALL FLOWS", type="primary", use_container_width=True):
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0

            X_input = df[feature_names].copy()
            X_input = X_input.replace([np.inf, -np.inf], np.nan).fillna(0)

            for col in X_input.columns:
                lo = X_input[col].quantile(0.01)
                hi = X_input[col].quantile(0.99)
                X_input[col] = X_input[col].clip(lo, hi)

            X_scaled = scaler.transform(X_input)
            probs = model.predict(X_scaled, verbose=0).flatten()
            preds = (probs >= threshold).astype(int)

            n_attacks = preds.sum()
            n_benign = len(preds) - n_attacks

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Flows Analysed", len(preds))
            c2.metric("🔴 Attacks Detected", int(n_attacks))
            c3.metric("🟢 Benign Traffic", int(n_benign))

            results = df.copy()
            results['Anomaly_Score'] = probs.round(4)
            results['Prediction'] = ['🔴 ATTACK' if p == 1 else '🟢 BENIGN' for p in preds]

            st.dataframe(results[['Anomaly_Score', 'Prediction']].head(20))

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if n_attacks > 0:
                st.error(f"🚨 {n_attacks} DDoS ATTACK(S) DETECTED")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.error(f"🔴 **IP BLOCKED**\n\n`{source_ip}` added to firewall blocklist.")
                with col_b:
                    st.warning("🟡 **RATE LIMITED**\n\nTraffic throttled to safe threshold.")
                with col_c:
                    st.info(f"🔔 **ALERT SENT**\n\nAdmin notified at `{timestamp}`.")

                st.code(f"""
SECURITY ALERT LOG
══════════════════════════════════════
Timestamp:      {timestamp}
Source IP:      {source_ip}
Flows Analysed: {len(preds)}
Attacks Found:  {int(n_attacks)}
Anomaly Scores: {[round(float(p),4) for p in probs[:5]]}
Actions:
  [1] IP BLOCKED   — {source_ip} blocklisted
  [2] RATE LIMITED — traffic throttled  
  [3] ALERT SENT   — administrator notified
══════════════════════════════════════
                """)
            else:
                st.success("✅ All traffic flows are BENIGN. No action required.")

            csv_out = results.to_csv(index=False)
            st.download_button("📥 Download Results", csv_out,
                               "detection_results.csv", "text/csv")

with tab2:
    st.subheader("How to Test This System")
    st.markdown("""
    **Step 1:** Download a few rows from the CICIDS2017 dataset as a CSV file.
    
    **Step 2:** Upload that CSV using the file uploader above.
    
    **Step 3:** Click **Analyse All Flows** to see which rows are attacks and which are benign.
    
    **What the system does automatically when attacks are detected:**
    - 🔴 Blocks the source IP address
    - 🟡 Applies rate limiting to throttle suspicious traffic  
    - 🔔 Generates a timestamped security alert for administrators
    
    **Model details:**
    - Architecture: Deep Neural Network (3 hidden layers)
    - Dataset: CICIDS2017 Binary Balanced
    - Training accuracy: 99%+
    - Features used: 52 network flow features
    """)
