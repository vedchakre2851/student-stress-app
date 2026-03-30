import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Phone Addicition Predictor",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model/phone_addiction_prediction_model.pkl")
    return model

pipeline = load_model()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("📘 About")
st.sidebar.info(
    """
    This app predicts **student stress level**
    based on digital habits, study patterns, sleep, and lifestyle behavior.
    """
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("Phone Addiction Predictor")
st.markdown("Fill in the details below and click **Predict**.")

st.divider()

# -----------------------------------
# INPUT FORM
# -----------------------------------
st.subheader("📋 Enter Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)

    daily_screen_time_hours = st.number_input(
        "Daily Screen Time (hours)", min_value=0.0, max_value=24.0, value=6.0
    )

    social_media_hours = st.number_input(
        "Social Media Usage (hours)", min_value=0.0, max_value=24.0, value=3.0
    )

    gaming_hours = st.number_input(
        "Gaming Hours", min_value=0.0, max_value=24.0, value=1.0
    )

    work_study_hours = st.number_input(
        "Work / Study Hours", min_value=0.0, max_value=24.0, value=6.0
    )

with col2:
    sleep_hours = st.number_input(
        "Sleep Hours", min_value=0.0, max_value=24.0, value=7.0
    )

    notifications_per_day = st.number_input(
        "Notifications Per Day", min_value=0, max_value=1000, value=120
    )

    app_opens_per_day = st.number_input(
        "App Opens Per Day", min_value=0, max_value=1000, value=60
    )

    weekend_screen_time = st.number_input(
        "Weekend Screen Time (hours)", min_value=0.0, max_value=24.0, value=8.0
    )

    academic_work_impact = st.selectbox(
        "Academic / Work Impact",
        ["Yes", "No"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    stress_level = st.selectbox(
        "Current Self-Reported Stress Level",
        ["Low", "Medium", "High"]
    )

st.divider()

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🚀 Predict"):

    input_data = pd.DataFrame([{
        'age': age,
        'daily_screen_time_hours': daily_screen_time_hours,
        'social_media_hours': social_media_hours,
        'gaming_hours': gaming_hours,
        'work_study_hours': work_study_hours,
        'sleep_hours': sleep_hours,
        'notifications_per_day': notifications_per_day,
        'app_opens_per_day': app_opens_per_day,
        'weekend_screen_time': weekend_screen_time,
        'academic_work_impact': academic_work_impact,
        'gender': gender,
        'stress_level': stress_level
    }])

    prediction = pipeline.predict(input_data)[0]

    st.subheader("📌 Prediction Result")
    st.success(f"Predicted Output: **{prediction}**")

    # Show probabilities if classifier supports it
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(input_data)[0]
        classes = pipeline.classes_

        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.subheader("📊 Prediction Confidence")
        st.dataframe(prob_df, use_container_width=True)