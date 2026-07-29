import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
}

h1,h2,h3{
    color:white;
}

.metric-card{
    background:#1f2937;
    padding:18px;
    border-radius:15px;
    border:1px solid #374151;
    text-align:center;
}

.prediction-card{
    padding:25px;
    border-radius:18px;
    color:white;
    font-size:24px;
    text-align:center;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "notebook", "xgb_model.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Model file not found at `{MODEL_PATH}`. "
        "Update MODEL_PATH to point to your trained xgb_model.pkl."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
# 📊 AI Customer Churn Prediction Dashboard
### Predict whether a telecom customer is likely to leave the company using Machine Learning (XGBoost)

---
""")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=90
)

st.sidebar.title("⚙ Customer Information")

st.sidebar.markdown("""
Fill all customer details and click **Predict Churn**.
""")

# =====================================================
# CUSTOMER PROFILE
# =====================================================
st.sidebar.header("👤 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
)
Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

# =====================================================
# TELEPHONE SERVICES
# =====================================================
st.sidebar.header("📞 Telephone Services")

PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)

# =====================================================
# INTERNET SERVICES
# =====================================================
st.sidebar.header("🌐 Internet Services")

InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

# =====================================================
# STREAMING SERVICES
# =====================================================
st.sidebar.header("📺 Streaming")

StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# =====================================================
# BILLING
# =====================================================
st.sidebar.header("💳 Billing")

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Predict Churn", use_container_width=True)

if predict_button:

    with st.spinner("🤖 AI is analyzing customer data..."):

        # Binary encoding — kept in separate _enc variables so the
        # original Yes/No/Male/Female labels stay available for display.
        gender_enc = 1 if gender == "Male" else 0
        Partner_enc = 1 if Partner == "Yes" else 0
        Dependents_enc = 1 if Dependents == "Yes" else 0
        PhoneService_enc = 1 if PhoneService == "Yes" else 0
        PaperlessBilling_enc = 1 if PaperlessBilling == "Yes" else 0

        input_data = pd.DataFrame({
            "gender": [gender_enc],
            "SeniorCitizen": [SeniorCitizen],
            "Partner": [Partner_enc],
            "Dependents": [Dependents_enc],
            "tenure": [tenure],
            "PhoneService": [PhoneService_enc],
            "PaperlessBilling": [PaperlessBilling_enc],
            "MonthlyCharges": [MonthlyCharges],
            "TotalCharges": [TotalCharges],

            "MultipleLines_No phone service": [1 if MultipleLines == "No phone service" else 0],
            "MultipleLines_Yes": [1 if MultipleLines == "Yes" else 0],

            "InternetService_Fiber optic": [1 if InternetService == "Fiber optic" else 0],
            "InternetService_No": [1 if InternetService == "No" else 0],

            "OnlineSecurity_No internet service": [1 if OnlineSecurity == "No internet service" else 0],
            "OnlineSecurity_Yes": [1 if OnlineSecurity == "Yes" else 0],

            "OnlineBackup_No internet service": [1 if OnlineBackup == "No internet service" else 0],
            "OnlineBackup_Yes": [1 if OnlineBackup == "Yes" else 0],

            "DeviceProtection_No internet service": [1 if DeviceProtection == "No internet service" else 0],
            "DeviceProtection_Yes": [1 if DeviceProtection == "Yes" else 0],

            "TechSupport_No internet service": [1 if TechSupport == "No internet service" else 0],
            "TechSupport_Yes": [1 if TechSupport == "Yes" else 0],

            "StreamingTV_No internet service": [1 if StreamingTV == "No internet service" else 0],
            "StreamingTV_Yes": [1 if StreamingTV == "Yes" else 0],

            "StreamingMovies_No internet service": [1 if StreamingMovies == "No internet service" else 0],
            "StreamingMovies_Yes": [1 if StreamingMovies == "Yes" else 0],

            "Contract_One year": [1 if Contract == "One year" else 0],
            "Contract_Two year": [1 if Contract == "Two year" else 0],

            "PaymentMethod_Credit card (automatic)": [1 if PaymentMethod == "Credit card (automatic)" else 0],
            "PaymentMethod_Electronic check": [1 if PaymentMethod == "Electronic check" else 0],
            "PaymentMethod_Mailed check": [1 if PaymentMethod == "Mailed check" else 0],
        })

        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # =====================================================
    # DASHBOARD
    # =====================================================

    confidence = max(probability[0]) * 100
    churn_probability = probability[0][1] * 100
    stay_probability = probability[0][0] * 100

    if churn_probability < 30:
        risk = "🟢 LOW"
    elif churn_probability < 60:
        risk = "🟡 MEDIUM"
    else:
        risk = "🔴 HIGH"

    st.markdown("## 📊 Customer Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📅 Tenure", f"{tenure} Months")
    with col2:
        st.metric("💵 Monthly", f"${MonthlyCharges:.2f}")
    with col3:
        st.metric("📈 Churn Probability", f"{churn_probability:.2f}%")
    with col4:
        st.metric("🎯 Confidence", f"{confidence:.2f}%")

    st.markdown("---")

    if prediction[0] == 1:
        st.markdown("""
        <div style='
        background:#991b1b;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;'>
        <h2>⚠️ Customer is Likely to Churn</h2>
        Immediate retention action is recommended.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
        background:#065f46;
        padding:25px;
        border-radius:15px;
        text-align:center;
        color:white;'>
        <h2>✅ Customer is Likely to Stay</h2>
        Customer shows low churn risk.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚨 Risk Level")
    st.info(risk)

    st.markdown("## 👤 Customer Summary")

    left, right = st.columns(2)

    with left:
        st.write("### Personal")
        st.write(f"Gender : {gender}")
        st.write(f"Senior Citizen : {'Yes' if SeniorCitizen else 'No'}")
        st.write(f"Partner : {Partner}")
        st.write(f"Dependents : {Dependents}")
        st.write(f"Tenure : {tenure} Months")

    with right:
        st.write("### Services")
        st.write(f"Internet : {InternetService}")
        st.write(f"Contract : {Contract}")
        st.write(f"Payment : {PaymentMethod}")
        st.write(f"Monthly Charges : ${MonthlyCharges:.2f}")
        st.write(f"Total Charges : ${TotalCharges:.2f}")

    st.markdown("---")

    st.markdown("## 💡 AI Recommendation")

    if prediction[0] == 1:
        st.warning("""
Offer retention benefits like:

• Loyalty discount

• Upgrade internet speed

• Recommend yearly contract

• Contact customer personally

• Offer free technical support
""")
    else:
        st.success("""
Customer is likely to remain.

Recommended Actions:

• Continue quality service

• Send reward coupons

• Promote premium plans
""")

    # =====================================================
    # VISUALIZATIONS
    # =====================================================
    st.markdown("---")
    st.header("📈 Prediction Analytics")

    # Gauge Chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability,
        number={'suffix': "%"},
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2563EB"},
            'steps': [
                {'range': [0, 30], 'color': "#16A34A"},
                {'range': [30, 60], 'color': "#FACC15"},
                {'range': [60, 100], 'color': "#DC2626"}
            ]
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # Pie Chart
    pie = px.pie(
        names=["Stay", "Churn"],
        values=[stay_probability, churn_probability],
        title="Prediction Probability Distribution",
        hole=0.55
    )
    st.plotly_chart(pie, use_container_width=True)

    # Probability Comparison Bar
    bar_df = pd.DataFrame({
        "Prediction": ["Stay", "Churn"],
        "Probability": [stay_probability, churn_probability]
    })
    bar = px.bar(
        bar_df,
        x="Probability",
        y="Prediction",
        orientation="h",
        text="Probability",
        title="Stay vs Churn Probability"
    )
    bar.update_traces(texttemplate='%{text:.2f}%')
    st.plotly_chart(bar, use_container_width=True)

    st.markdown("---")
    st.header("📋 Customer Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Charges", f"${MonthlyCharges:.2f}")
    with col2:
        st.metric("Total Charges", f"${TotalCharges:.2f}")
    with col3:
        st.metric("Tenure", f"{tenure} Months")

    st.markdown("### 🚦 Risk Meter")
    if churn_probability < 30:
        st.success("🟢 Low Risk Customer")
    elif churn_probability < 60:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.error("🔴 High Risk Customer")

    # =====================================================
    # AI EXPLAINABILITY (SHAP)
    # =====================================================
    st.markdown("---")
    st.header("🧠 AI Explainability (SHAP)")

    st.write("""
    SHAP (SHapley Additive exPlanations) explains **why** the model made its prediction by showing
    which features increased or decreased the customer's churn risk.
    """)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Some SHAP/XGBoost version combos return a list (one array per
        # class) for binary classifiers instead of a single 2D array.
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
            base_value = explainer.expected_value[1] if isinstance(
                explainer.expected_value, (list, tuple)
            ) else explainer.expected_value
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value

        st.subheader("🌊 SHAP Waterfall Plot")

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=base_value,
                data=input_data.iloc[0],
                feature_names=input_data.columns
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.warning("SHAP visualization could not be generated.")
        st.write(e)

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    st.markdown("---")
    st.header("📊 Top Feature Importance")

    try:
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df.head(10), use_container_width=True)

        fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Most Important Features"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("Feature importance unavailable.")

    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================
    st.markdown("---")
    st.header("📋 Prediction Summary")

    summary = pd.DataFrame({
        "Feature": ["Prediction", "Risk Level", "Confidence", "Churn Probability", "Stay Probability"],
        "Value": [
            "Churn" if prediction[0] == 1 else "Stay",
            risk,
            f"{confidence:.2f}%",
            f"{churn_probability:.2f}%",
            f"{stay_probability:.2f}%"
        ]
    })
    st.table(summary)

    st.markdown("---")
    st.header("👤 Customer Snapshot")

    snapshot = pd.DataFrame({
        "Customer Detail": [
            "Gender", "Senior Citizen", "Partner", "Dependents",
            "Internet", "Contract", "Payment Method",
            "Monthly Charges", "Total Charges", "Tenure"
        ],
        "Value": [
            gender,
            "Yes" if SeniorCitizen else "No",
            Partner,
            Dependents,
            InternetService,
            Contract,
            PaymentMethod,
            f"${MonthlyCharges:.2f}",
            f"${TotalCharges:.2f}",
            tenure
        ]
    })
    st.dataframe(snapshot, use_container_width=True)

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================
    st.markdown("---")
    st.header("📥 Download Prediction Report")

    report = pd.DataFrame({
        "Prediction": ["Churn" if prediction[0] == 1 else "Stay"],
        "Risk Level": [risk],
        "Confidence": [f"{confidence:.2f}%"],
        "Churn Probability": [f"{churn_probability:.2f}%"],
        "Stay Probability": [f"{stay_probability:.2f}%"],
        "Contract": [Contract],
        "Internet": [InternetService],
        "Payment": [PaymentMethod]
    })

    csv = report.to_csv(index=False)
    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="Customer_Churn_Report.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.header("🤖 Model Information")

    model_info = pd.DataFrame({
        "Property": ["Model", "Algorithm", "Target", "Framework", "Language"],
        "Value": ["Customer Churn Predictor", "XGBoost Classifier", "Customer Churn", "Streamlit", "Python"]
    })
    st.table(model_info)

    st.markdown("---")
    st.header("📘 About this Project")

    st.info("""
This application predicts whether a telecom customer is likely to churn.

Machine Learning Model:
• XGBoost Classifier

Features Used:
• Customer Demographics
• Services
• Billing
• Internet Usage
• Contract Details

Explainability:
• SHAP AI Explainability

Visualization:
• Plotly Dashboard
""")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:gray;'>

    ### 🚀 Customer Churn Prediction Dashboard

    Developed using

    Python • Streamlit • XGBoost • Plotly • SHAP

    © 2026

    </div>
    """, unsafe_allow_html=True)