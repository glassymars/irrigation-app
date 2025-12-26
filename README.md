# 🌾 Personalized Irrigation Predictor

**The Smart Irrigation Predictor** web app is built with Streamlit to help farmers estimate irrigation water requirements for crops. Using machine learning models and SHAP values, the app predicts water needs and provides clear, farmer-friendly explanations in multiple languages using a chatbot API.

## Key Features

- Predict crop-specific water requirements based on field data.
- Understand which factors influence water needs using global SHAP feature importance.
- Get practical explanations in multiple Indian regional languages using the OpenRouter DeepSeek API.
- A user-friendly and responsive interface.

## Input

### Crop Details
- Crop type, soil type, season, and water source.

### Field Measurements
- Soil pH, crop duration, temperature, and relative humidity.

### Soil Nutrients
- Levels of Nitrogen (N), Phosphorus (P), and Potassium (K).

### Explanation Settings
- Choose the preferred language for chatbot-generated explanations.
