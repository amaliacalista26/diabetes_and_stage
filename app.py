import streamlit as st
import pandas as pd
import joblib

# Load Models
model = joblib.load("rf_diabetes_task1.pkl")
scaler = joblib.load("scaler_task1.pkl")

model_task2 = joblib.load("rf_diabetes_task2.pkl")
scaler_task2 = joblib.load("scaler_task2.pkl")

# Page Config
st.set_page_config(
    page_title="Diabetes Detection System",
    layout="centered"
)

st.title("Diabetes Detection System")
st.write(
    "A machine learning–based system to predict whether a person is diagnosed "
    "with diabetes based on clinical and demographic data."
)

st.markdown("---")

# Input Section
st.subheader("Clinical Information")

col1, col2 = st.columns(2)

with col1:
    hba1c = st.number_input("HbA1c (%)", 4.0, 12.0, step=0.1)
    glucose_fasting = st.number_input("Fasting Glucose (mg/dL)", 60, 200)
    insulin = st.number_input("Insulin Level", 2.0, 40.0, step=0.1)

with col2:
    glucose_post = st.number_input("Postprandial Glucose (mg/dL)", 70, 300)
    triglycerides = st.number_input("Triglycerides (mg/dL)", 30, 400)

st.subheader("Demographic Information")

col3, col4 = st.columns(2)

with col3:
    family_history = st.selectbox(
        "Family History of Diabetes",
        ["No", "Yes"]
    )

    hypertension = st.selectbox(
        "History of Hypertension",
        ["No", "Yes"]
    )

age = st.number_input("Age (years)", 18, 100)
bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0)

st.markdown("---")

# ======================
# Prepare Input Data
# ======================
input_df = pd.DataFrame([{
    'hba1c': hba1c,
    'glucose_fasting': glucose_fasting,
    'glucose_postprandial': glucose_post,
    'insulin_level': insulin,
    'triglycerides': triglycerides,
    'family_history_diabetes': 1 if family_history == "Yes" else 0,
    'hypertension_history': 1 if hypertension == "Yes" else 0,
    'age': age,
    'bmi': bmi
}])

num_cols = [
    'hba1c',
    'glucose_fasting',
    'glucose_postprandial',
    'insulin_level',
    'triglycerides',
    'age',
    'bmi'
]

input_df[num_cols] = scaler.transform(input_df[num_cols])

# ======================
# Prediction
# ======================
if st.button("Predict Diabetes Status"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("### Prediction Result")

    if pred == 1:
        st.error(
            f"**Diabetes Detected**\n\n"
            f"Probability of diabetes: **{prob:.2%}**"
        )

        # ======================
        # Task 2 Prediction
        # ======================
        st.markdown("### Diabetes Type Prediction")

        input_task2 = pd.DataFrame([{
            'hba1c': hba1c,
            'glucose_fasting': glucose_fasting,
            'glucose_postprandial': glucose_post,
            'insulin_level': insulin,
            'bmi': bmi,
            'age': age
        }])

        num_cols_task2 = [
            'hba1c',
            'glucose_fasting',
            'glucose_postprandial',
            'insulin_level',
            'bmi',
            'age'
        ]

        input_task2[num_cols_task2] = scaler_task2.transform(
            input_task2[num_cols_task2]
        )

        stage_pred = model_task2.predict(input_task2)[0]

        stage_mapping = {
            0: "Gestational Diabetes",
            3: "Type 1 Diabetes",
            4: "Type 2 Diabetes"
        }

        st.success(
            f"Predicted Diabetes Type: "
            f"**{stage_mapping.get(stage_pred, 'Unknown')}**"
        )

    else:
        st.success(
            f"**No Diabetes Detected**\n\n"
            f"Probability of no diabetes: **{1 - prob:.2%}**"
        )

# ======================
# Footer
# ======================
st.markdown("---")
st.caption(
    "Model: Random Forest – Binary Classification\n"
    "This system is intended for educational purposes only."
)