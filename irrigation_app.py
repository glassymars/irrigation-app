import streamlit as st
import pickle
import torch
import numpy as np
import requests

st.markdown(
    """
    <style>

        body, .stApp {
            background-color: #e8f4ff !important;
        }

        header {
            margin-bottom: 30px !important;
        }

        .stApp {
            padding-top: 40px !important;
        }

        .block-container {
            padding-top: 20px !important;
            padding-bottom: 10px !important;
        }

        .block-container {
            background-color: transparent !important;
            box-shadow: none !important;
            padding: 1rem !important;
            max-width: 900px !important;
        }

        .main-container {
            background-color: white !important;
            border-radius: 15px !important;
            padding: 30px !important;
            margin-bottom: 30px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
            border: 1px solid #e0e7ff !important;
        }

        .main-title {
            color: #1a237e !important;
            text-align: center !important;
            margin-bottom: 10px !important;
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            padding-top: 20px !important;
        }

        .sub-title {
            color: #3949ab !important;
            text-align: center !important;
            margin-bottom: 30px !important;
            font-size: 1.2rem !important;
            font-weight: 400 !important;
        }

        .instruction-box {
            background-color: #f0f9ff !important;
            border-left: 5px solid #2196f3 !important;
            border-radius: 10px !important;
            padding: 20px !important;
            margin-bottom: 25px !important;
            margin-top: 20px !important;
            box-shadow: 0 2px 10px rgba(33, 150, 243, 0.1) !important;
        }

        /* Section headers */
        .section-header {
            color: #1565c0 !important;
            border-bottom: 2px solid #bbdefb !important;
            padding-bottom: 10px !important;
            margin-top: 25px !important;
            margin-bottom: 20px !important;
        }

        input, textarea, select {
            background-color: #ffffff !important;
            color: black !important;
            border-radius: 8px !important;
            border: 1px solid #bdbdbd !important;
        }

        /* Streamlit-specific wrappers */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stDateInput > div > div input,
        .stTimeInput > div > div input,
        .stTextArea textarea,
        .stSlider > div > div > div {
            background-color: #ffffff !important;
            border-radius: 8px !important;
            color: black !important;
            border: 1px solid #bdbdbd !important;
        }

        /* + and - buttons blue */
        .stNumberInput button {
            background-color: #31578f !important;
            color: white !important;
            border-radius: 6px !important;
            border: none !important;
        }

        /* Dropdown menu background white */
        div[role="listbox"] {
            background-color: #ffffff !important;
            border-radius: 8px !important;
        }

        /* Button styling */
        .stButton>button {
            border-radius: 10px !important;
            background-color: #1976d2 !important;
            color: white !important;
            font-weight: bold !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            border: none !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            margin-top: 20px !important;
        }

        .stButton>button:hover {
            background-color: #1565c0 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3) !important;
        }

        /* Alerts */
        .stAlert {
            border-radius: 10px !important;
            border-left: 5px solid !important;
        }

        /* Card containers for features */
        .feature-card {
            background: linear-gradient(135deg, #f5f7ff 0%, #e8f0fe 100%) !important;
            border-radius: 10px !important;
            padding: 15px !important;
            margin-bottom: 15px !important;
            border: 1px solid #e0e7ff !important;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .main-container {
                padding: 20px !important;
            }
            .main-title {
                font-size: 2rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# MAIN TITLE FIRST (Fixed order)
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)  # Simple line breaks
st.markdown('<h1 class="main-title">🌾 Smart Irrigation Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Get accurate water requirements for your crops with AI-powered insights</p>', unsafe_allow_html=True)

# -----------------------------
# INSTRUCTIONS AFTER TITLE
# -----------------------------
st.markdown(
    """
    <div class="instruction-box">
        <h3 style="color: #1976d2; margin-top: 0;">How to Use This Tool</h3>
        <p style="margin-bottom: 5px;"><strong>Step 1:</strong> Select your crop, soil type, season, and water source from the dropdown menus.</p>
        <p style="margin-bottom: 5px;"><strong>Step 2:</strong> Enter your specific field measurements (soil pH, temperature, nutrients, etc.)</p>
        <p style="margin-bottom: 5px;"><strong>Step 3:</strong> Choose your preferred language for the explanation.</p>
        <p style="margin-bottom: 0;"><strong>Step 4:</strong> Click "Predict Water Requirement" to get your customized irrigation plan.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# DeepSeek/OpenRouter API Key
CHAT_API_KEY = st.secrets["CHAT_API_KEY"]  # replace with your key

# Load Model Bundle
with open("water_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model_state = bundle["model_state"]
scaler = bundle["scaler"]

# PyTorch model definition
class ImprovedWaterModel(torch.nn.Module):
    def __init__(self, input_size=11, hidden_size=128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.BatchNorm1d(hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, x):
        return self.network(x)

model = ImprovedWaterModel(input_size=11)
model.load_state_dict(model_state)
model.eval()

# Load Global SHAP Values
with open("global_shap_values.pkl", "rb") as f:
    shap_data = pickle.load(f)

global_shap_values = shap_data["shap_values"]
X_sample = shap_data["X_sample"]
feature_names = shap_data["feature_names"]

# Encoders
replacement_encoders = {
    "CROPS": {0: "Pearl millet", 1: "bitter gourd", 2: "blackgram", 3: "bottle gourd",
              4: "capsicum", 5: "cauliflower", 6: "chowchow", 7: "cowpea", 8: "jute",
              9: "kudiraivali", 10: "panivaragu", 11: "peas", 12: "pumpkin",
              13: "redgram", 14: "ribbed gourd", 15: "samai", 16: "soyabean",
              17: "sugarcane", 18: "tomato", 19: "watermelon"},
    "SOIL": {0: "Clay soil", 1: "Laterite soil", 2: "Loamy soil", 3: "Sandy soil",
             4: "Sandy soil", 5: "Sandy soil", 6: "black cotton soil", 7: "brown Loamy soil",
             8: "clay Loamy soil", 9: "deep soil", 10: "loamy soil", 11: "red Loamy soil",
             12: "sandy Loamy soil", 13: "sandy loamy soil", 14: "well-drained loamy soil",
             15: "well-drained soil", 16: "cotton soil"},
    "SEASON": {0: "Zaid", 1: "kharif", 2: "rabi"},
    "WATER_SOURCE": {0: "irrigated", 1: "rainfed"}
}
reverse_encoders = {col: {v: k for k, v in mapping.items()} for col, mapping in replacement_encoders.items()}

# DeepSeek API
def generate_farmer_explanation(prompt_text):
    try:
        headers = {
            'Authorization': f'Bearer {CHAT_API_KEY}',
            'Content-Type': 'application/json',
        }
        data = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You explain crop water needs in simple language for farmers."},
                {"role": "user", "content": prompt_text}
            ]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"API error: {e}"

# Start Main Content Container

st.markdown('<h2 class="section-header">📊 Enter Your Field Details</h2>', unsafe_allow_html=True)

# Streamlit UI- Crop Details
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    crop = st.selectbox("**CROP TYPE**", list(reverse_encoders["CROPS"].keys()))
    soil = st.selectbox("**SOIL TYPE**", list(reverse_encoders["SOIL"].keys()))
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    season = st.selectbox("**SEASON**", list(reverse_encoders["SEASON"].keys()))
    water_source = st.selectbox("**WATER SOURCE**", list(reverse_encoders["WATER_SOURCE"].keys()))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<h2 class="section-header">📈 Field Measurements</h2>', unsafe_allow_html=True)

# Environmental conditions
col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    soi_ph = st.number_input("**SOIL PH**", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    crop_duration = st.number_input("**CROP DURATION (days)**", min_value=1.0, max_value=365.0, value=90.0, step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    temp = st.number_input("**TEMPERATURE (°C)**", min_value=-10.0, max_value=50.0, value=30.0, step=0.5)
    rel_humidity = st.number_input("**RELATIVE HUMIDITY (%)**", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Nutrient inputs
st.markdown('<h2 class="section-header">🌿 Soil Nutrients</h2>', unsafe_allow_html=True)

col5, col6, col7 = st.columns(3)
with col5:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    n = st.number_input("**NITROGEN (N)**", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col6:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    p = st.number_input("**PHOSPHORUS (P)**", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
    st.markdown('</div>', unsafe_allow_html=True)

with col7:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    k = st.number_input("**POTASSIUM (K)**", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Language selection
st.markdown('<h2 class="section-header">🗣️ Explanation Settings</h2>', unsafe_allow_html=True)
st.markdown('<div class="feature-card">', unsafe_allow_html=True)

language_options = [
    "English",
    "Hindi",
    "Marathi",
    "Tamil",
    "Telugu",
    "Kannada",
    "Gujarati",
    "Bengali",
    "Punjabi",
    "Malayalam",
    "Odia"
]

selected_language = st.selectbox("**🌐 SELECT EXPLANATION LANGUAGE**", language_options)
st.markdown('</div>', unsafe_allow_html=True)

# Function to build prompt
def build_comprehensive_prompt(y_pred, x_input, feature_names):
    crop_name = replacement_encoders["CROPS"][int(x_input[0,0])]
    soil_name = replacement_encoders["SOIL"][int(x_input[0,1])]
    season_name = replacement_encoders["SEASON"][int(x_input[0,2])]
    water_source_name = replacement_encoders["WATER_SOURCE"][int(x_input[0,3])]

    feature_dict = {
        "CROPS": crop_name,
        "SOIL": soil_name,
        "SEASON": season_name,
        "WATER_SOURCE": water_source_name,
        "SOIL_PH": x_input[0,4],
        "CROPDURATION": x_input[0,5],
        "TEMP": x_input[0,6],
        "REL_HUMIDITY": x_input[0,7],
        "N": x_input[0,8],
        "P": x_input[0,9],
        "K": x_input[0,10],
    }

    shap_dict = dict(zip(feature_names, np.mean(np.abs(global_shap_values), axis=0)))
    shap_sorted = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    top_features_text = ", ".join([f"{f[0]} ({f[1]:.2f})" for f in shap_sorted])

    prompt = (
        f"You are an agricultural assistant. Explain irrigation water needs to a farmer in simple, clear language.\n\n"
        f"The predicted water requirement is {y_pred:.2f} liters.\n\n"
        f"Crop details:\n"
        f"- Crop: {crop_name}\n"
        f"- Season: {season_name}\n"
        f"- Soil type: {soil_name}\n"
        f"- Water source: {water_source_name}\n\n"
        f"Input conditions:\n"
        f"- Soil pH: {feature_dict['SOIL_PH']}\n"
        f"- Crop duration: {feature_dict['CROPDURATION']} days\n"
        f"- Temperature: {feature_dict['TEMP']}°C\n"
        f"- Relative humidity: {feature_dict['REL_HUMIDITY']}%\n"
        f"- Nitrogen (N): {feature_dict['N']}\n"
        f"- Phosphorus (P): {feature_dict['P']}\n"
        f"- Potassium (K): {feature_dict['K']}\n\n"
        f"Feature influences based on SHAP values:\n"
        f"{top_features_text}\n\n"
        f"Your task:\n"
        f"1. Explain in plain language why this amount of water is needed.\n"
        f"2. For each input (soil type, season, temperature, humidity, N, P, K, soil pH, crop duration), explain how it increases or decreases water requirement.\n"    
        f"3. Keep the explanation practical and easy for a farmer to understand.\n"
        f"4. Use short sentences, bullet points, and simple vocabulary and do not start with filler words.\n"
        f"5. Do not repeat all numbers unless needed for clarity.\n"
        f"6. Avoid phrases like 'the model says' or 'based on the input'.\n"
        f"7. Make the tone friendly and supportive.\n"
        f"8. Format the answer clearly so it is easy to read on a phone.\n"
        f"9. Write the explanation in {selected_language}. If this language is not supported, use simple English.\n"
    )
    return prompt

# Prediction + Explanation
if st.button("🚀 PREDICT WATER REQUIREMENT"):
    # Build input array
    x_input = np.array([[reverse_encoders["CROPS"][crop],
                         reverse_encoders["SOIL"][soil],
                         reverse_encoders["SEASON"][season],
                         reverse_encoders["WATER_SOURCE"][water_source],
                         soi_ph, crop_duration, temp,
                         rel_humidity, n, p, k]], dtype=np.float32)

    x_scaled = scaler.transform(x_input)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(x_tensor).item()
    
    # Success message with icon
    st.markdown(f"""
        <div style="text-align: center; background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%); 
                    border-radius: 15px; padding: 20px; margin-bottom: 30px; border-left: 5px solid #4caf50;">
            <h2 style="color: #2e7d32; margin: 0;">💧 Prediction Complete!</h2>
            <h1 style="color: #1b5e20; font-size: 2.5rem; margin: 10px 0;">{y_pred:.2f} Liters</h1>
            <p style="color: #388e3c; margin: 0;">Total water requirement for your {crop}</p>
        </div>
    """, unsafe_allow_html=True)

    # Generate explanation
    st.markdown('<h2 class="section-header">📝 Farmer-Friendly Explanation</h2>', unsafe_allow_html=True)
    
    with st.spinner(f"Generating explanation in {selected_language}..."):
        prompt_text = build_comprehensive_prompt(y_pred, x_input, feature_names)
        explanation = generate_farmer_explanation(prompt_text)
        
        st.markdown(explanation)

    # SHAP values in an expander
    with st.expander("🔍 View Feature Importance Analysis"):
        st.markdown('<h3 class="section-header">📊 Global Feature Importances (SHAP)</h3>', unsafe_allow_html=True)
        
        shap_vals_mean = np.mean(np.abs(global_shap_values), axis=0)
        shap_pairs = list(zip(feature_names, shap_vals_mean))
        shap_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, value) in enumerate(shap_pairs):
            color = "#1976d2" if i == 0 else "#2196f3" if i == 1 else "#64b5f6" if i == 2 else "#90caf9"
            st.markdown(f"""
                <div style="background: linear-gradient(90deg, {color}20 0%, {color}10 100%); 
                            border-radius: 8px; padding: 12px 15px; margin-bottom: 8px;">
                    <span style="font-weight: bold;">{name}</span>
                    <span style="float: right; font-weight: bold; color: #1565c0;">{value:.4f}</span>
                </div>
            """, unsafe_allow_html=True)
    
# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; color: #546e7a;">
        <hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, #90a4ae, transparent);">
        <p style="font-size: 0.9rem;">🌾 Smart Irrigation Predictor • Made for Farmers • AI-Powered Insights</p>
    </div>
""", unsafe_allow_html=True)