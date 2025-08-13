import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -------------------
# Load Model & Scaler
# -------------------
model = pickle.load(open("pipeline_model.pkl", "rb"))     
scaler = pickle.load(open("pipeline_scaler.pkl", "rb"))   

# Dynamically get feature names from scaler
feature_names = list(scaler.feature_names_in_)

# Risk Labels
condition_mapping = {0: 'Normal', 1: 'Moderate', 2: 'Critical'}

# Prediction Function (reads correct column names from scaler)
def predict_condition(model, scaler, features):
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    return condition_mapping[prediction[0]]

# Maintenance Strategies
maintenance_strategies = {
    'Normal': [
        "Maintain a 12-month inspection cycle using inline inspection (ILI) or ultrasonic testing (UT) to verify wall thickness trends.",
        "Requalify external protective coatings on a 3–5 year schedule to ensure long-term barrier performance.",
        "Continuously monitor cathodic protection (CP) system potentials; confirm compliance with NACE SP0169 criteria.",
        "Verify that operating pressure remains within design limits to reduce cyclic stress and corrosion fatigue."
    ],
    'Moderate': [
        "Accelerate inspection frequency to every 6 months using high-resolution ILI or targeted direct assessments.",
        "Deploy advanced wall-thickness mapping tools and install composite sleeves or steel reinforcement over high-loss areas.",
        "Upgrade coating systems in degraded sections to arrest external corrosion progression.",
        "Perform close-interval potential surveys (CIPS) to detect and correct CP shielding or underprotection.",
        "Implement transient pressure monitoring to control corrosion-fatigue interaction."
    ],
    'Critical': [
        "Immediately isolate and depressurize the compromised pipeline segment to prevent catastrophic failure.",
        "Cut out and replace pipe sections showing >50% wall loss, active leaks, or cracking indications.",
        "Execute hydrostatic pressure testing post-repair to validate the integrity of remaining segments.",
        "Perform a full-scope pipeline integrity assessment in accordance with API 1163 and ASME B31.8S standards.",
        "Deploy emergency mitigation such as anodic protection or temporary chemical inhibitors until permanent remediation is complete."
    ]
}


# -------------------
# Page Config & Title
# -------------------
st.set_page_config(page_title="BOST Pipeline Condition Predictor", page_icon="🛠️", layout="wide")

st.markdown("""
    <div style='text-align: center; padding: 10px 0; position: relative;'>
        <h1 style='color: #004f7c; margin-bottom: 0;'>🛠️ BOST Pipeline Risk Prediction System</h1>
        <h4 style='color: #555;'>Predict pipeline corrosion risk and get instant maintenance advice</h4>
    </div>
""", unsafe_allow_html=True)

# -------------------
# Input Layout
# -------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📏 Physical Properties")
    pipe_size = st.number_input("Pipe Size (mm)", min_value=50, max_value=1000, value=300)
    thickness = st.number_input("Original Wall Thickness (mm)", min_value=1.0, max_value=50.0, value=8.0)
    material = st.selectbox("Material", ['Carbon Steel', 'HDPE', 'PVC', 'Fiberglass', 'Stainless Steel'])
    grade = st.selectbox("Material Grade", ['A', 'B', 'C', 'X42', 'X52'])
    thickness_loss = st.number_input("Measured Thickness Loss (mm)", min_value=0.0, max_value=20.0, value=2.5)

with col2:
    st.subheader("⚙️ Operating Conditions")
    max_pressure = st.number_input("Max Pressure (psi)", 100, 2000, 800)
    temperature = st.slider("Operating Temperature (°C)", -10, 150, 25)
    corrosion_impact = st.slider("Corrosion Impact (%)", 0.0, 100.0, 12.5)
    material_loss = st.slider("Material Loss (%)", 0.0, 100.0, 5.0)
    time_years = st.slider("Time in Service (Years)", 0, 50, 10)

# Encode categorical variables
material_encoded = {'Carbon Steel': 0, 'HDPE': 1, 'PVC': 2, 'Fiberglass': 3, 'Stainless Steel': 4}[material]
grade_encoded = {'A': 0, 'B': 1, 'C': 2, 'X42': 3, 'X52': 4}[grade]

# Match order to scaler.feature_names_in_
input_data = {
    'Pipe_Size_mm': pipe_size,
    'Thickness_mm': thickness,
    'Material': material_encoded,
    'Grade': grade_encoded,
    'Max_Pressure_psi': max_pressure,
    'Temperature_C': temperature,
    'Corrosion_Impact_Percent': corrosion_impact,
    'Material_Loss_Percent': material_loss,
    'Time_Years': time_years,
    'Thickness_Loss_mm': thickness_loss
}

# Create list in correct order
input_features = [input_data[col] for col in feature_names]

# -------------------
# Prediction Button
# -------------------
st.markdown("---")
if st.button("🔍 Predict Risk Level", use_container_width=True):
    prediction = predict_condition(model, scaler, input_features)
    
    st.markdown("## 📊 Prediction Result")
    if prediction == 'Critical':
        st.error(f"**{prediction}** – ⚠️ Immediate Maintenance Required!")
    elif prediction == 'Moderate':
        st.warning(f"**{prediction}** – Schedule Inspection in 6 Months.")
    else:
        st.success(f"**{prediction}** – Continue Routine Monitoring.")
    
    # Create a popover
    with st.popover("Open Pop-Up"):
    # Collapsible section for maintenance strategies
        with st.expander("🛠 View Recommended Maintenance Actions"):
            for action in maintenance_strategies[prediction]:
                st.markdown(f"- {action}")

# -------------------
# Styling
# -------------------
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stButton>button {background-color: #004f7c; color: white; font-size: 18px; border-radius: 8px;}
    .stButton>button:hover {background-color: #006ea1; color: white;}
</style>
""", unsafe_allow_html=True)
