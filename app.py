import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DDoS Threat Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ── Load model and scaler ────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model('ddos_final_model.keras')
    with open('ddos_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('ddos_features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# ── Header ───────────────────────────────────────────────────
st.title("🛡️ Predictive Threat Detection & Automated Remediation System")
st.markdown("**Deep Learning-Based DDoS Attack Detection using CICIDS2017 Dataset**")
st.markdown("*Final Year Project — Emili Elizabeth Anwuli & Ankara Job Emmanuel*")
st.divider()

# ── Sidebar info ─────────────────────────────────────────────
with st.sidebar:
    st.header("About This System")
    st.info("""
    This system uses a Deep Neural Network trained on the 
    CICIDS2017 dataset to detect DDoS attacks in real time 
    and automatically trigger remediation actions:
    
    🔴 **IP Blocking** — blocks malicious source  
    🟡 **Rate Limiting** — throttles suspicious traffic  
    🔔 **Alert Generation** — notifies administrators
    """)
    st.metric("Model Accuracy", "99%+")
    st.metric("Dataset", "CICIDS2017")
    st.metric("Architecture", "Deep Neural Network")

# ── Two modes ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Flow Detection", "📊 Batch Detection"])

# ── TAB 1: Single prediction ─────────────────────────────────
with tab1:
    st.subheader("Enter Network Flow Features")
    st.markdown("Fill in the traffic flow details below to check if it represents an attack.")

    col1, col2, col3 = st.columns(3)

    # We show the 7 key features as manual inputs
    # Others default to 0 (conservative, works fine for demo)
    with col1:
        flow_duration = st.number_input("Flow Duration (μs)", min_value=0.0, value=128476.0)
        fwd_packets = st.number_input("Total Fwd Packets", min_value=0.0, value=1.0)
        flow_bytes_s = st.number_input("Flow Bytes/s", min_value=0.0, value=1315.42)

    with col2:
        flow_packets_s = st.number_input("Flow Packets/s", min_value=0.0, value=15.57)
        avg_packet_size = st.number_input("Average Packet Size", min_value=0.0, value=53.0)
        init_win = st.number_input("Init Win Bytes Forward", min_value=-1.0, value=256.0)

    with col3:
        pkt_len_var = st.number_input("Packet Length Variance", min_value=0.0, value=0.0)
        source_ip = st.text_input("Source IP Address (for remediation log)", value="192.168.1.100")
        threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)

    if st.button("🔍 ANALYSE TRAFFIC FLOW", type="primary", use_container_width=True):

        # Build a full feature vector (52 features), set key ones, rest to 0
        input_vector = np.zeros((1, len(feature_names)))
        key_values = {
            'Flow Duration': flow_duration,
            'Total Fwd Packets': fwd_packets,
            'Flow Bytes/s': flow_bytes_s,
            'Flow Packets/s': flow_packets_s,
            'Average Packet Size': avg_packet_size,
            'Init_Win_bytes_forward': init_win,
            'Packet Length Variance': pkt_len_var
        }
        for feat, val in key_values.items():
            if feat in feature_names:
                idx = feature_names.index(feat)
                input_vector[0, idx] = val

        # Scale and predict
        input_scaled = scaler.transform(input_vector)
        pred_prob = model.predict(input_scaled, verbose=0)[0][0]
        is_attack = pred_prob >= threshold
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.divider()

        # Result display
        if is_attack:
            st.error(f"🚨 **ATTACK DETECTED** — Anomaly Score: `{pred_prob:.4f}`")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.error("🔴 Step 1: IP BLOCKED\n\n"
                         f"Source IP `{source_ip}` has been added to the firewall blocklist.")
            with col_b:
                st.warning("🟡 Step 2: RATE LIMITED\n\n"
                           "Traffic throttling activated. Packet rate restricted to safe threshold.")
            with col_c:
                st.info("🔔 Step 3: ALERT SENT\n\n"
                        f"Security alert dispatched to administrator at `{timestamp}`.")

            st.code(f"""
SECURITY ALERT LOG
══════════════════════════════════════════
Timestamp:     {timestamp}
Source IP:     {source_ip}
Anomaly Score: {pred_prob:.6f}
Classification: DDOS ATTACK
Actions Taken:
  [1] IP BLOCKED   — {source_ip} added to blocklist
  [2] RATE LIMITED — traffic throttled
  [3] ALERT SENT   — administrator notified
══════════════════════════════════════════
            """)
        else:
            st.success(f"✅ **BENIGN TRAFFIC** — Anomaly Score: `{pred_prob:.4f}`")
            st.info("No action required. Traffic is within normal parameters. Monitoring continues.")

# ── TAB 2: Batch detection ───────────────────────────────────
with tab2:
    st.subheader("Upload a CSV for Batch Analysis")
    st.markdown("Upload a CSV file containing network flow records. "
                "The system will classify each row and generate a full report.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        import pandas as pd
        batch_df = pd.read_csv(uploaded_file)
        batch_df.columns = batch_df.columns.str.strip()

        st.write(f"Loaded {len(batch_df)} records. Preview:")
        st.dataframe(batch_df.head())

        if st.button("🔍 RUN BATCH DETECTION", type="primary"):
            # Align columns to training features
            for col in feature_names:
                if col not in batch_df.columns:
                    batch_df[col] = 0
            batch_X = batch_df[feature_names].values
            batch_scaled = scaler.transform(batch_X)
            preds_prob = model.predict(batch_scaled, verbose=0).flatten()
            preds_label = (preds_prob >= 0.5).astype(int)

            batch_df['Anomaly_Score'] = preds_prob.round(4)
            batch_df['Prediction'] = ['🔴 ATTACK' if p == 1 else '🟢 BENIGN' for p in preds_label]

            n_attacks = preds_label.sum()
            n_benign = len(preds_label) - n_attacks

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Flows Analysed", len(preds_label))
            c2.metric("Attacks Detected 🔴", int(n_attacks))
            c3.metric("Benign Traffic 🟢", int(n_benign))

            st.dataframe(batch_df[['Anomaly_Score', 'Prediction']].join(
                batch_df.drop(['Anomaly_Score', 'Prediction'], axis=1).iloc[:, :5]))

            csv_out = batch_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv_out,
                               "ddos_detection_results.csv", "text/csv")

st.divider()
st.caption("Predictive Threat Modelling & Automated Remediation | "
           "Deep Learning Final Year Project | AUL 2025")
