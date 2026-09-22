import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# 1. Define the exact column names the model expects
columns = ['engine_id', 'cycle', 'set_1', 'set_2', 'set_3'] + ['sensor_' + str(i) for i in range(1, 22)]

# 2. Set up the page
st.set_page_config(page_title="NASA Jet Engine RUL", page_icon="✈️")
st.title("✈️ Predictive Maintenance Dashboard")
st.write("Upload engine sensor data to predict Remaining Useful Life (RUL).")

# 3. Download and Load the saved model using joblib
@st.cache_resource
def get_model():
    model_path = 'nasa_rul_model.pkl
    model_url = "https://github.com/mehdifr24/predictive_maintenance_pipeline/releases/download/v1.0/nasa_rul_model.pkl"  
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading the AI model... Please wait (195 MB)"):
            urllib.request.urlretrieve(model_url, model_path)
            
    # Load the model directly using joblib
    model_pipeline = joblib.load(model_path)
    return model_pipeline
    
model = get_model()

# 4. Create a file uploader
uploaded_file = st.file_uploader("Upload Sensor Data", type=["csv", "txt"])

if uploaded_file is not None:
    # Read the uploaded data AND apply the column names
    df = pd.read_csv(uploaded_file, sep='\s+', header=None, names=columns)
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # 5. Make predictions
    if st.button("Predict Engine Life"):
        # Make predictions directly with the loaded pipeline
        predictions = model.predict(df)
        
        st.success("Prediction Complete!")
        st.write("Predicted RUL (Remaining Cycles):")
        
        # Combine engine_id and cycle with the prediction for better readability
        result_df = df[['engine_id', 'cycle']].copy()
        result_df['Predicted RUL'] = predictions
        st.dataframe(result_df)
