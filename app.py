import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scalers
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('fraud_detection_model.pkl')
        scaler_amount = joblib.load('scaler_amount.pkl')
        scaler_time = joblib.load('scaler_time.pkl')
        return model, scaler_amount, scaler_time
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

model, scaler_amount, scaler_time = load_assets()

# --- Sidebar ---
st.sidebar.title("🛡️ Project Details")
st.sidebar.write("### Credit Card Fraud Detection System")
st.sidebar.write("This application uses Machine Learning to identify fraudulent credit card transactions.")

st.sidebar.subheader("ℹ️ Instructions")
st.sidebar.info(
    """
    1. Enter the transaction details in the main panel.
    2. Click the 'Detect Fraud' button.
    3. View the prediction result.
    """
)

st.sidebar.subheader("📊 Model Details")
st.sidebar.write("**Algorithm:** Random Forest Classifier")
st.sidebar.write("**Dataset:** Credit Card Fraud Detection Dataset (Kaggle)")
st.sidebar.write("**Features:** Time, Amount, V1-V28 (PCA Components)")

st.sidebar.divider()
st.sidebar.write("Developed by **ML Developer**")

# --- Main Page ---
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    """
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='margin: 0; color: #333;'>
            This web app uses a Machine Learning model to detect fraudulent credit card transactions.
            Enter the transaction features below to check for potential fraud.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Section 1: Transaction Input ---
st.subheader("📝 Transaction Features")

with st.form("prediction_form"):
    st.write("Please enter the transaction details:")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    # Time and Amount are specific, so we place them prominently
    with col1:
        time_val = st.number_input("Time (Seconds elapsed)", min_value=0.0, value=0.0, step=1.0, help="Number of seconds elapsed between this transaction and the first transaction in the dataset")
    with col2:
        amount_val = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=0.01, help="Transaction Amount")
    
    # Dictionary to store V features
    v_features = {}
    
    # Generate inputs for V1 to V28
    # We'll distribute them across the 3 columns
    cols = [col1, col2, col3]
    
    for i in range(1, 29):
        feature_name = f"V{i}"
        col_index = (i + 1) % 3 # Shift by 2 because we used 2 spots already? No, let's just flow them naturally.
        # Actually, let's just create a new set of columns for V features to be neat
    
    st.markdown("---")
    st.write("**PCA Components (V1 - V28)**")
    
    # Create a grid for V features (4 columns)
    v_cols = st.columns(4)
    for i in range(1, 29):
        feature_name = f"V{i}"
        with v_cols[(i-1) % 4]:
            v_features[feature_name] = st.number_input(f"{feature_name}", value=0.0, step=0.01)

    # --- Section 2: Prediction Button ---
    st.markdown("---")
    submit_button = st.form_submit_button(label="🔍 Detect Fraud", use_container_width=True)

# --- Section 3: Prediction Output ---
if submit_button:
    if model and scaler_amount and scaler_time:
        try:
            # Prepare input data
            # 1. Scale Time and Amount
            time_scaled = scaler_time.transform(np.array([[time_val]]))
            amount_scaled = scaler_amount.transform(np.array([[amount_val]]))
            
            # 2. Combine all features in the correct order: Time, V1...V28, Amount?
            # We need to check the order used in training.
            # In training: df = pd.read_csv('creditcard.csv') -> X = df.drop('Class', axis=1)
            # The columns in CSV are: Time, V1...V28, Amount.
            # So the order is: Time, V1, V2, ..., V28, Amount.
            
            input_data = [time_scaled[0][0]]
            for i in range(1, 29):
                input_data.append(v_features[f"V{i}"])
            input_data.append(amount_scaled[0][0])
            
            # Convert to numpy array and reshape
            final_input = np.array([input_data])
            
            # Make prediction
            prediction = model.predict(final_input)[0]
            prediction_proba = model.predict_proba(final_input)[0]
            
            # Display Result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                # Fraud
                st.error("🚨 **WARNING: FRAUDULENT TRANSACTION DETECTED!**")
                st.write(f"Confidence Score: **{prediction_proba[1]*100:.2f}%**")
                st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDdtY2J5dG5wY2J5dG5wY2J5dG5wY2J5dG5wY2J5dG5wY2J5dG5w/3o7aD2saalBwwftBIY/giphy.gif", width=200, caption="Fraud Alert") # Optional decorative gif
            else:
                # Normal
                st.success("✅ **Transaction is Normal.**")
                st.write(f"Confidence Score: **{prediction_proba[0]*100:.2f}%**")
                
            # Metrics / Chart (Simple visualization of probability)
            st.write("### Confidence Levels")
            prob_df = pd.DataFrame({
                "Class": ["Normal", "Fraud"],
                "Probability": prediction_proba
            })
            st.bar_chart(prob_df.set_index("Class"))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model assets not loaded properly. Please check the setup.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>ML Project | Credit Card Fraud Detection | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
