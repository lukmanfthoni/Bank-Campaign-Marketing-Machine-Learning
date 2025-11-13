import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt
import joblib


# Page configuration
st.set_page_config(page_title="XGBoost Model Prediction", layout="wide")

# Load model and explainer
@st.cache_resource
def load_model():
    with open('Streamlit/bestmodel.pkl', 'rb') as f:
        model = joblib.load(f)
    return model

@st.cache_resource
def load_explainer():
    with open('Streamlit/lime_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return explainer

# Initialize
try:
    pipeline = load_model()
    lime_explainer = load_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or explainer: {e}")
    model_loaded = False

# Title
st.title("🎯 Deposit Model Prediction System")
st.markdown("---")

# Create input form in the middle of the screen
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.header("📋 Input Features")
    
    # Group 1: Personal Information
    st.subheader("👤 Personal Information")
    age = st.number_input("Age", min_value=18, max_value=95, value=41, step=1)
    job = st.selectbox("Job", ['admin.', 'self-employed', 'services', 'housemaid', 
                                'technician', 'management', 'student', 'blue-collar', 
                                'entrepreneur', 'retired', 'unemployed', 'unknown'])
    
    # Group 2: Financial Information
    st.subheader("💰 Financial Information")
    balance = st.number_input("Balance", min_value=-6847, max_value=66653, value=1512, step=100)
    housing = st.selectbox("Housing Loan", ['no', 'yes'])
    loan = st.selectbox("Personal Loan", ['no', 'yes'])
    
    # Group 3: Campaign Information
    st.subheader("📞 Campaign Information")
    contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])
    month = st.slider("Month", min_value=1, max_value=12, value=6, step=1)
    campaign = st.number_input("Campaign (number of contacts)", min_value=1, max_value=63, value=2, step=1)
    pdays = st.number_input("Days since last contact (-1 = never contacted)", 
                            min_value=-1, max_value=854, value=51, step=1)
    
    st.markdown("---")
    
    # Predict button
    predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# Prediction and LIME explanation
if predict_button and model_loaded:
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'campaign': [campaign],
        'pdays': [pdays]
    })
    
    try:
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)[0]
        
        # Display prediction
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("Prediction", "Positive" if prediction == 1 else "Negative")
        
        with result_col2:
            st.metric("Confidence (Class 0)", f"{prediction_proba[0]:.2%}")
        
        with result_col3:
            st.metric("Confidence (Class 1)", f"{prediction_proba[1]:.2%}")
        
        # LIME Explanation
        st.markdown("---")
        st.header("🔍 LIME Explanation")
        st.write("Understanding what features influenced this prediction:")
        
        # Transform input data using preprocessing step
        preprocessed_data = pipeline.named_steps['preprocessing'].transform(input_data)
        
        # Get LIME explanation
        # Create prediction function for LIME that takes preprocessed data
        def predict_fn(data):
            return pipeline.named_steps['model'].predict_proba(data)
        
        # Generate explanation
        exp = lime_explainer.explain_instance(
            preprocessed_data.values[0], 
            predict_fn,
            num_features=10
        )
        
        # Display LIME plot
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)
        plt.close()
        
        # Additional explanation details
        with st.expander("📝 Feature Importance Details"):
            explanation_list = exp.as_list()
            for feature, weight in explanation_list:
                st.write(f"**{feature}**: {weight:.4f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Please check your input values and try again.")

elif predict_button and not model_loaded:
    st.error("Model or explainer not loaded. Please check the files.")

# Footer
st.markdown("---")

st.markdown("*Built with Streamlit and XGBoost*")

