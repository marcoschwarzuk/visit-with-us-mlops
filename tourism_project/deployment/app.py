import json

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "Marco8000/visit-with-us-model"
MODEL_FILE = "best_tourism_model_v1.joblib"


@st.cache_resource
def load_model_and_threshold():
    """Download the registered model and its serving threshold once."""
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILE, repo_type="model")
    classification_threshold = 0.50

    try:
        metrics_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="metrics.json", repo_type="model")
        with open(metrics_path, "r", encoding="utf-8") as file:
            classification_threshold = float(json.load(file).get("classification_threshold", classification_threshold))
    except Exception:
        # Models registered before threshold metadata remain usable at the default threshold.
        pass

    return joblib.load(model_path), classification_threshold


model, classification_threshold = load_model_and_threshold()

st.title("Visit with Us - Wellness Tourism Package Predictor")
st.write("""
This application predicts whether a customer is likely to **purchase the new Wellness Tourism Package**
before the sales team contacts them. Enter the customer profile and the interaction details below.
""")

st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=23000, step=500)
with col2:
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    number_of_trips = st.number_input("Average number of trips per year", min_value=0, max_value=30, value=3, step=1)
    persons_visiting = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3, step=1)
    children_visiting = st.number_input("Number of children (below 5) visiting", min_value=0, max_value=5, value=1, step=1)
    property_star = st.selectbox("Preferred property star rating", [3, 4, 5])

st.subheader("Interaction details")
col3, col4 = st.columns(2)
with col3:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=180, value=15, step=1)
with col4:
    number_of_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4, step=1)
    pitch_satisfaction = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)

# Assemble the inputs into a DataFrame with the same column names as the training data
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    probability = float(model.predict_proba(input_data)[0][1])
    prediction = int(probability >= classification_threshold)
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The customer is **likely to purchase** the Wellness Tourism Package (probability {probability:.1%}).")
    else:
        st.info(f"The customer is **unlikely to purchase** the Wellness Tourism Package (probability {probability:.1%}).")
    with st.expander("Show the input sent to the model"):
        st.dataframe(input_data)
