import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Exoplanet Habitability Predictor",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #6c63ff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #5146d9;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    h1, h2, h3 {
        color: #a29bfe !important;
    }
    .prediction-habitable {
        background: rgba(46, 204, 113, 0.2);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #2ecc71;
        text-align: center;
    }
    .prediction-non-habitable {
        background: rgba(231, 76, 60, 0.2);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #e74c3c;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load data and models
@st.cache_resource
def load_resources():
    # Paths relative to the root of the repository
    model_path = Path("outputs/models/production/simplified_8feature_model.pkl")
    feature_path = Path("outputs/models/production/simplified_features.json")
    presets_path = Path("outputs/planet_presets.json")
    
    model = joblib.load(model_path)
    
    with open(feature_path, 'r') as f:
        feature_info = json.load(f)
        required_features = feature_info['features']
        
    with open(presets_path, 'r') as f:
        planet_presets = json.load(f)
        
    return model, required_features, planet_presets

try:
    model, required_features, planet_presets = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Sidebar - Planet Selection
st.sidebar.title("🪐 Planet Explorer")
st.sidebar.markdown("Select a planet to pre-fill the data or enter your own values.")

# Dropdown options
preset_options = ["Custom Input", "Earth"] + [p["name"] for p in planet_presets["kepler"]] + [p["name"] for p in planet_presets["test_samples"]]
selected_preset = st.sidebar.selectbox("Choose a Planet", preset_options)

# Get selected data
current_data = {}
if selected_preset == "Earth":
    current_data = planet_presets["earth"]["data"]
elif selected_preset != "Custom Input":
    # Find in kepler or test_samples
    all_presets = planet_presets["kepler"] + planet_presets["test_samples"]
    match = next((p for p in all_presets if p["name"] == selected_preset), None)
    if match:
        current_data = match["data"]

# Main Title
st.title("Exoplanet Habitability Predictor")
st.markdown("### Determining the potential for life across the stars 🚀")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### Exoplanet Parameters")
    with st.container():
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            p_mass = st.number_input("Planet Mass (Earth=1.0)", value=float(current_data.get("P_MASS_EST", 1.0)), step=0.01, format="%.2f")
            p_radius = st.number_input("Planet Radius (Earth=1.0)", value=float(current_data.get("P_RADIUS_EST", 1.0)), step=0.01, format="%.2f")
            p_temp = st.number_input("Equilibrium Temp (K)", value=float(current_data.get("P_TEMP_EQUIL", 288.0)), step=0.1, format="%.1f")
            p_period = st.number_input("Orbital Period (days)", value=float(current_data.get("P_PERIOD", 365.25)), step=0.1, format="%.1f")
            
        with subcol2:
            p_flux = st.number_input("Stellar Flux (Earth=1.0)", value=float(current_data.get("P_FLUX", 1.0)), step=0.01, format="%.2f")
            s_mass = st.number_input("Star Mass (Sun=1.0)", value=float(current_data.get("S_MASS", 1.0)), step=0.01, format="%.2f")
            s_radius = st.number_input("Star Radius (Sun=1.0)", value=float(current_data.get("S_RADIUS", 1.0)), step=0.01, format="%.2f")
            s_temp = st.number_input("Star Temp (K)", value=float(current_data.get("S_TEMP", 5778.0)), step=1.0, format="%.0f")

    if st.button("Analyze Habitability"):
        # Prepare input
        input_dict = {
            "P_MASS_EST": p_mass,
            "P_RADIUS_EST": p_radius,
            "P_TEMP_EQUIL": p_temp,
            "P_PERIOD": p_period,
            "P_FLUX": p_flux,
            "S_MASS": s_mass,
            "S_RADIUS": s_radius,
            "S_TEMPERATURE": s_temp  # Model expects S_TEMPERATURE
        }
        
        # Create DataFrame in correct order
        X = pd.DataFrame([input_dict])[required_features]
        
        # Prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Binary classification mapping: 0 -> Non-Habitable, (1, 2) -> Habitable
        prob_non_habitable = probabilities[0]
        prob_habitable = probabilities[1] + (probabilities[2] if len(probabilities) > 2 else 0)
        
        is_habitable = prob_habitable > prob_non_habitable
        
        with col2:
            st.markdown("#### Prediction Result")
            if is_habitable:
                st.markdown(f"""
                <div class="prediction-habitable">
                    <h2 style='color: #2ecc71 !important; margin: 0;'>HABITABLE</h2>
                    <p style='margin: 0;'>Confidence: {prob_habitable:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="prediction-non-habitable">
                    <h2 style='color: #e74c3c !important; margin: 0;'>NON-HABITABLE</h2>
                    <p style='margin: 0;'>Confidence: {prob_non_habitable:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Confidence Breakdown")
            st.progress(prob_habitable)
            st.text(f"Habitability Probability: {prob_habitable:.1%}")

with col2:
    if not st.session_state.get('analyzed', False):
        st.info("👈 Enter planet parameters and click 'Analyze Habitability' to see results.")
    
    st.markdown("---")
    st.markdown("#### Model Info")
    st.write("Using Linear SVM with SMOTE")
    st.write("Accuracy: 99.63%")
    st.write("Features: 8 key parameters")

# Footer
st.markdown("---")
st.markdown(
    "Created by Aditya Jatling | [Infosys Springboard](https://springboard.infosysapps.com/)"
)
