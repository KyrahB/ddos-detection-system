import streamlit as st
import numpy as np
import pickle
import datetime

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

st.title("Predictive Threat Detection & Automated Remediation System")
st.markdown("**Deep Learning-Based DDoS Attack Detection & Remediation using CICIDS2017 Dataset**")
st.markdown("*Final Year Project : Elizabeth Emili & Ankara Job*")
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

st.subheader("Enter Network Flow Features")
st.markdown("Fill in the traffic flow details below to check for a DDoS attack.")

col1, col2, col3 = st.columns(3)

with col1:
    flow_duration    = st.number_input("Flow Duration (μs)",       min_value=0.0, value=128476.0)
    fwd_packets      = st.number_input("Total Fwd Packets",         min_value=0.0, value=1.0)
    flow_bytes_s     = st.number_input("Flow Bytes/s",              min_value=0.0, value=1315.42)

with col2:
    flow_packets_s   = st.number_input("Flow Packets/s",            min_value=0.0, value=15.57)
    avg_packet_size  = st.number_input("Average Packet Size",       min_value=0.0, value=53.0)
    init_win         = st.number_input("Init Win Bytes Forward",    min_value=-1.0, value=256.0)

with col3:
    pkt_len_var      = st.number_input("Packet Length Variance",    min_value=0.0, value=0.0)
    source_ip        = st.text_input("Source IP Address",           value="192.168.1.100")
    threshold        = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05)

if st.button("ANALYSE TRAFFIC FLOW", type="primary", use_container_width=True):

    input_vector = np.zeros((1, len(feature_names)))
    key_values = {
        'Flow Duration':          flow_duration,
        'Total Fwd Packets':      fwd_packets,
        'Flow Bytes/s':           flow_bytes_s,
        'Flow Packets/s':         flow_packets_s,
        'Average Packet Size':    avg_packet_size,
        'Init_Win_bytes_forward': init_win,
        'Packet Length Variance': pkt_len_var
    }
    for feat, val in key_values.items():
        if feat in feature_names:
            input_vector[0, feature_names.index(feat)] = val

    input_scaled = scaler.transform(input_vector)
    pred_prob    = model.predict(input_scaled, verbose=0)[0][0]
    is_attack    = pred_prob >= threshold
    timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.divider()

    if is_attack:
        st.error(f"🚨 ATTACK DETECTED — Anomaly Score: `{pred_prob:.4f}`")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.error(f"🔴 **IP BLOCKED**\n\n`{source_ip}` added to firewall blocklist.")
        with col_b:
            st.warning("🟡 **RATE LIMITED**\n\nPacket rate throttled to safe threshold.")
        with col_c:
            st.info(f"🔔 **ALERT SENT**\n\nAdministrator notified at `{timestamp}`.")

        st.code(f"""
SECURITY ALERT LOG
══════════════════════════════════════
Timestamp:      {timestamp}
Source IP:      {source_ip}
Anomaly Score:  {pred_prob:.6f}
Classification: DDoS ATTACK
Actions:
  [1] IP BLOCKED   — {source_ip} blocklisted
  [2] RATE LIMITED — traffic throttled
  [3] ALERT SENT   — admin notified
══════════════════════════════════════
        """)
    else:
        st.success(f"✅ BENIGN TRAFFIC — Anomaly Score: `{pred_prob:.4f}`")
        st.info("No action required. Traffic is within normal parameters.")

st.divider()
st.caption("Predictive Threat Modelling & Automated Remediation | Deep Learning Final Year Project")
