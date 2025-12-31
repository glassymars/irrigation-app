# 🌾 Personalized Irrigation Predictor

**The Smart Irrigation Predictor** web app is built with Streamlit to help farmers estimate irrigation water requirements for crops. It uses a machine learning regression model implemented in PyTorch, trained with a simulated federated learning (FedAvg) setup, along with SHAP values to predict water needs and provide clear, farmer-friendly explanations in multiple languages using the DeepSeek API.

## Key Features

- Machine learning–based irrigation prediction using a PyTorch regression model. 
- Simulated federated learning with FedAvg for privacy-aware training.
- Predict crop-specific water requirements based on field data.
- Understand which factors influence water needs using global SHAP feature importance.
- Get practical explanations in multiple Indian regional languages using the OpenRouter DeepSeek API.
- A user-friendly and responsive interface.

## Streamlit

You can access the app live on [Irrigation Predictor](https://irrigation-app-temjbxgncgnqnscftcqe2v.streamlit.app/).  
Enter your field details to get water predictions and explanations.

## Input

### Crop Details
- Crop type, soil type, season, and water source.

### Field Measurements
- Soil pH, crop duration, temperature, and relative humidity.

### Soil Nutrients
- Levels of Nitrogen (N), Phosphorus (P), and Potassium (K).

### Explanation Settings
- Choose the preferred language for chatbot-generated explanations.
