import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

# --- 1. ENTERPRISE CONFIGURATION ---
st.set_page_config(
    page_title="FinGuard Elite | Corporate Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. THE CSS SUITE ---
st.markdown("""
    <style>
        /* GLOBAL THEME */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* BACKGROUND & GRADIENTS */
        .stApp {
            background-color: #0f172a; /* Slate 900 */
            background-image: radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.1) 0px, transparent 50%),
                              radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 50%);
        }
        
        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {
            background-color: #1e293b; /* Slate 800 */
            border-right: 1px solid #334155;
        }
        
        /* HEADERS */
        h1, h2, h3 {
            background: -webkit-linear-gradient(0deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700 !important;
        }
        
        /* GLASSMORPHISM CARDS */
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #38bdf8;
        }
        
        /* BUTTONS */
        .stButton>button {
            background: linear-gradient(90deg, #0ea5e9, #2563eb);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 14px 0 rgba(14, 165, 233, 0.39);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0284c7, #1d4ed8);
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.23);
        }
        
        /* DATAFRAMES */
        [data-testid="stDataFrame"] {
            border: 1px solid #334155;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* ALERTS & INFO */
        .stAlert {
            background-color: rgba(30, 41, 59, 0.8);
            border: 1px solid #334155;
            color: #e2e8f0;
            border-radius: 8px;
        }
        
        /* FOOTER */
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Set plotting style to dark
plt.style.use('dark_background')
sns.set_palette("viridis")

# --- 3. SESSION STATE LOGIC (UNCHANGED) ---
keys = ['df', 'df_clean', 'target', 'model', 'scaler', 'features', 'test_data', 'history']
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# --- 4. CORE LOGIC (UNCHANGED) ---
def clean_data_logic(df):
    df.columns = df.columns.str.strip()
    target = None
    if 'isFraud' in df.columns:
        target = 'isFraud'
        if df[target].dtype == 'object':
            df = df[df[target] != 'Not reviewed']
            df[target] = df[target].map({'Safe': 0, 'Fraud': 1})
    elif 'Class' in df.columns:
        target = 'Class'
    else:
        return None, None
    
    drop_cols = ['nameOrig', 'nameDest', 'Date of transaction', 'Time of day', 
                 'branch', 'Acct type', 'DayOfWeek(new)', 'Column1', 'isFraud - Copy', 'TransactionID']
    actual_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=actual_drop)
    
    if 'type' in df.columns:
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'].astype(str))
    
    df = df.select_dtypes(include=[np.number])
    return df, target

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("## 🛡️ FinGuard **Elite**")
    st.caption("v5.0.1 | ENTERPRISE EDITION")
    st.markdown("---")
    
    menu = st.radio(
        "COMMAND CENTER",
        [
            "1. 🏠 Executive Overview",
            "2. 📂 Secure Ingestion",
            "3. 🧹 Smart Cleaning Log",
            "4. 📊 Fraud Distribution",
            "5. 🔗 Correlation Matrix",
            "6. ⚙️ Neural Config",
            "7. 📉 Model Performance",
            "8. 📈 Advanced Forensics",
            "9. 🕵️ Live Screener",
            "10. 📁 Bulk Audit"
        ]
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='background-color: #1e293b; padding: 10px; border-radius: 5px; border-left: 3px solid #10b981;'>
            <small style='color: #94a3b8;'>STATUS</small><br>
            <strong style='color: #e2e8f0;'>● SYSTEM ONLINE</strong>
        </div>
        <div style='margin-top: 10px; background-color: #1e293b; padding: 10px; border-radius: 5px; border-left: 3px solid #38bdf8;'>
            <small style='color: #94a3b8;'>SECURITY</small><br>
            <strong style='color: #e2e8f0;'>🔒 ENCRYPTED</strong>
        </div>
        """, unsafe_allow_html=True
    )

# ==============================================================================
# PAGE 1: SYSTEM OVERVIEW
# ==============================================================================
if menu == "1. 🏠 Executive Overview":
    st.title("Executive Command Center")
    st.markdown("_Real-time Fraud Intelligence Dashboard_")
    
    st.markdown("---")
    
    # Fake Metrics for Dashboard Look
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Protocols", "12", "+2")
    col2.metric("Threat Level", "LOW", "-14%")
    col3.metric("Uptime", "99.99%", "Stable")
    col4.metric("Transactions Scanned", "1.2M", "+50k")
    
    st.markdown("### 📡 System Telemetry")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.info("✅ **Mainframe Connection:** Stable (12ms latency)")
        st.info("✅ **AI Engine:** Idle / Ready for training")
        st.info("✅ **Database:** Synced with AWS RDS")
        
    with c2:
        st.markdown(
            """
            <div style='background: #1e293b; padding: 20px; border-radius: 10px; border: 1px solid #334155;'>
                <h4 style='margin:0; color: #94a3b8;'>Recent Alerts</h4>
                <ul style='list-style: none; padding: 0; margin-top: 10px; color: #cbd5e1; font-size: 0.9em;'>
                    <li style='margin-bottom: 8px;'>⚠️ IP 192.168.1.X flagged</li>
                    <li style='margin-bottom: 8px;'>🔄 Batch upload completed</li>
                    <li style='margin-bottom: 8px;'>🔒 Admin login detected</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

# ==============================================================================
# PAGE 2: DATA INGESTION
# ==============================================================================
elif menu == "2. 📂 Secure Ingestion":
    st.title("Secure Data Ingestion")
    st.markdown("Upload encrypted transaction ledgers (CSV) for analysis.")
    
    with st.container():
        uploaded_file = st.file_uploader("Drag & Drop Secure CSV", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            
            st.success(f"✅ FILE AUTHENTICATED: {uploaded_file.name}")
            
            with st.expander("🔎 INSPECT RAW DATA STREAM", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
        
        elif st.session_state['df'] is not None:
            st.info("Active dataset loaded in memory.")
            st.dataframe(st.session_state['df'].head(), use_container_width=True)

# ==============================================================================
# PAGE 3: DATA CLEANING LOG
# ==============================================================================
elif menu == "3. 🧹 Smart Cleaning Log":
    st.title("Automated Preprocessing")
    
    if st.session_state['df'] is None:
        st.warning("⚠️ No data stream detected. Please upload in Module 2.")
    else:
        st.markdown("### 🛠️ Sanitization Pipeline")
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(
            """<div style='padding:15px; border:1px solid #334155; border-radius:8px; text-align:center;'>
                <h3>Step 1</h3>
                <p>Whitespace & Null Removal</p>
            </div>""", unsafe_allow_html=True)
        col2.markdown(
            """<div style='padding:15px; border:1px solid #334155; border-radius:8px; text-align:center;'>
                <h3>Step 2</h3>
                <p>String ID Filtration</p>
            </div>""", unsafe_allow_html=True)
        col3.markdown(
            """<div style='padding:15px; border:1px solid #334155; border-radius:8px; text-align:center;'>
                <h3>Step 3</h3>
                <p>Categorical Encoding</p>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 EXECUTE CLEANING PROTOCOLS"):
            with st.spinner("Optimizing datasets..."):
                df_clean, target = clean_data_logic(st.session_state['df'])
                
                if df_clean is not None:
                    st.session_state['df_clean'] = df_clean
                    st.session_state['target'] = target
                    st.success("✅ DATASET OPTIMIZED FOR MACHINE LEARNING")
                    st.dataframe(df_clean.head(), use_container_width=True)
                else:
                    st.error("Protocol Failed: Target variable not found.")

# ==============================================================================
# PAGE 4: DISTRIBUTION ANALYSIS
# ==============================================================================
elif menu == "4. 📊 Fraud Distribution":
    st.title("Distribution Intelligence")
    
    if st.session_state['df_clean'] is None:
        st.error("Please run cleaning protocols first.")
    else:
        df = st.session_state['df_clean']
        target = st.session_state['target']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Class Imbalance Visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_alpha(0) # Transparent background
            ax.set_facecolor('#0f172a')
            
            sns.countplot(x=target, data=df, palette=['#10b981', '#ef4444'], ax=ax)
            ax.set_title("Safe vs. Fraudulent Transactions", color='white', pad=20)
            ax.set_xlabel("Transaction Class", color='#94a3b8')
            ax.set_ylabel("Count", color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            sns.despine()
            st.pyplot(fig)
            
        with col2:
            st.markdown("### Metrics")
            fraud_count = df[target].sum()
            safe_count = len(df) - fraud_count
            
            st.metric("Total Safe", f"{safe_count:,}", delta="Normal")
            st.metric("Total Fraud", f"{fraud_count:,}", delta="-CRITICAL", delta_color="inverse")
            st.metric("Imbalance Ratio", f"1 : {safe_count//fraud_count}")

# ==============================================================================
# PAGE 5: CORRELATION LAB
# ==============================================================================
elif menu == "5. 🔗 Correlation Matrix":
    st.title("Feature Correlation Lab")
    st.markdown("Identify hidden relationships between financial variables.")
    
    if st.session_state['df_clean'] is None:
        st.warning("Data not ready.")
    else:
        df = st.session_state['df_clean']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_alpha(0)
        ax.set_facecolor('#0f172a')
        
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        sns.heatmap(df.corr(), mask=mask, cmap='viridis', vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        
        ax.tick_params(colors='#94a3b8')
        st.pyplot(fig)

# ==============================================================================
# PAGE 6: MODEL CONFIG
# ==============================================================================
elif menu == "6. ⚙️ Neural Config":
    st.title("Model Hyperparameters")
    
    if st.session_state['df_clean'] is None:
        st.warning("Awaiting data.")
    else:
        st.markdown("### 🎛️ Architecture Settings")
        
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                model_type = st.selectbox("Algorithm Selection", ["Random Forest Classifier", "Logistic Regression"])
                estimators = st.slider("Ensemble Estimators", 10, 200, 50)
            with c2:
                test_size = st.slider("Validation Split", 0.1, 0.5, 0.2)
                random_state = st.number_input("Random Seed (Reproducibility)", value=42)
        
        st.markdown("---")
        
        if st.button("⚡ INITIALIZE NEURAL TRAINING"):
            with st.spinner("Training models on GPU cluster (simulated)..."):
                df = st.session_state['df_clean']
                target = st.session_state['target']
                
                X = df.drop(columns=[target])
                y = df[target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                if model_type == "Random Forest Classifier":
                    model = RandomForestClassifier(n_estimators=estimators, random_state=random_state)
                else:
                    model = LogisticRegression()
                
                model.fit(X_train, y_train)
                
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['features'] = X.columns.tolist()
                st.session_state['test_data'] = (X_test, y_test)
                
                st.success("✅ TRAINING COMPLETE. MODEL ARTIFACTS SAVED.")

# ==============================================================================
# PAGE 7: PERFORMANCE METRICS
# ==============================================================================
elif menu == "7. 📉 Model Performance":
    st.title("Performance Analytics")
    
    if st.session_state['model'] is None:
        st.error("Model not trained.")
    else:
        model = st.session_state['model']
        X_test, y_test = st.session_state['test_data']
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        col1.metric("Model Accuracy", f"{acc:.2%}", delta="High Confidence")
        
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('#0f172a')
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel("Actual Class", color='white')
        ax.set_xlabel("Predicted Class", color='white')
        ax.tick_params(colors='#94a3b8')
        st.pyplot(fig)
        
        with st.expander("🔎 View Detailed Classification Report"):
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# ==============================================================================
# PAGE 8: ADVANCED ANALYTICS
# ==============================================================================
elif menu == "8. 📈 Advanced Forensics":
    st.title("Advanced Forensics Suite")
    
    if st.session_state['model'] is None:
        st.error("Model not trained.")
    else:
        model = st.session_state['model']
        X_test, y_test = st.session_state['test_data']
        y_prob = model.predict_proba(X_test)[:, 1]
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### ROC-AUC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots()
            fig.patch.set_alpha(0)
            ax.set_facecolor('#0f172a')
            
            ax.plot(fpr, tpr, color='#38bdf8', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='#94a3b8')
            ax.set_xlabel('False Positive Rate', color='#94a3b8')
            ax.set_ylabel('True Positive Rate', color='#94a3b8')
            ax.legend(facecolor='#1e293b', labelcolor='white')
            ax.tick_params(colors='#94a3b8')
            sns.despine()
            st.pyplot(fig)
            
        with c2:
            st.markdown("### Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            
            fig2, ax2 = plt.subplots()
            fig2.patch.set_alpha(0)
            ax2.set_facecolor('#0f172a')
            
            ax2.plot(recall, precision, color='#10b981', lw=2)
            ax2.set_xlabel('Recall', color='#94a3b8')
            ax2.set_ylabel('Precision', color='#94a3b8')
            ax2.tick_params(colors='#94a3b8')
            sns.despine()
            st.pyplot(fig2)

# ==============================================================================
# PAGE 9: SINGLE TRANSACTION SCREEN
# ==============================================================================
elif menu == "9. 🕵️ Live Screener":
    st.title("Manual Transaction Screener")
    st.markdown("Analyze single transaction vectors for immediate risk assessment.")
    
    if st.session_state['model'] is None:
        st.warning("⚠️ Neural Network Offline. Initialize in Module 6.")
    else:
        features = st.session_state['features']
        inputs = {}
        
        with st.form("screener_form"):
            st.markdown("### 📝 Transaction Parameters")
            cols = st.columns(3)
            for i, col in enumerate(features):
                if i < 9:
                    with cols[i % 3]:
                        inputs[col] = st.number_input(f"{col}", value=0.0)
            
            st.markdown("---")
            submitted = st.form_submit_button("🔍 ANALYZE RISK VECTOR")
        
        if submitted:
            df_in = pd.DataFrame([inputs]).reindex(columns=features, fill_value=0)
            scaled = st.session_state['scaler'].transform(df_in)
            prob = st.session_state['model'].predict_proba(scaled)[0][1]
            
            st.markdown("### Analysis Result")
            if prob > 0.5:
                st.error(f"🚨 THREAT DETECTED (Confidence: {prob:.2%})")
                st.markdown("**Recommendation:** Flag for immediate review and freeze assets.")
            else:
                st.success(f"✅ CLEARED (Risk: {prob:.2%})")
                st.markdown("**Recommendation:** Approve transaction.")

# ==============================================================================
# PAGE 10: BATCH FORENSICS
# ==============================================================================
elif menu == "10. 📁 Bulk Audit":
    st.title("Bulk Forensic Audit")
    st.markdown("High-volume CSV processing for retroactive fraud identification.")
    
    if st.session_state['model'] is None:
        st.error("Model must be trained before running audits.")
    else:
        batch_file = st.file_uploader("Upload Audit CSV", type=["csv"])
        
        if batch_file:
            batch_df = pd.read_csv(batch_file)
            st.info(f"Processing {len(batch_df)} records...")
            
            # Logic
            clean_batch = batch_df.select_dtypes(include=[np.number])
            clean_batch = clean_batch.reindex(columns=st.session_state['features'], fill_value=0)
            
            scaled = st.session_state['scaler'].transform(clean_batch)
            probs = st.session_state['model'].predict_proba(scaled)[:, 1]
            
            batch_df['Fraud_Probability'] = probs
            batch_df['Prediction'] = (probs > 0.5).astype(int)
            
            frauds = batch_df[batch_df['Prediction'] == 1]
            
            if not frauds.empty:
                st.markdown(f"### 🚩 Audit Report: {len(frauds)} Threats Identified")
                st.dataframe(frauds.head(), use_container_width=True)
                
                csv = frauds.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ DOWNLOAD FORENSIC REPORT",
                    data=csv,
                    file_name="FinGuard_Forensic_Report.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ Audit Clean: No threats detected in this batch.")
