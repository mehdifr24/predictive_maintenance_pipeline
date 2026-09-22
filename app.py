import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# 1. Define the exact column names the NASA dataset expects
columns = ['engine_id', 'cycle', 'set_1', 'set_2', 'set_3'] + ['sensor_' + str(i) for i in range(1, 22)]

# 2. Configure Streamlit page settings
st.set_page_config(page_title="NASA Jet Engine RUL", page_icon="✈️")
st.title("✈️ Predictive Maintenance Dashboard")
st.write("Upload engine sensor data to predict Remaining Useful Life (RUL).")

# 3. Download the model from GitHub Releases and load it using joblib
@st.cache_resource
def get_model():
    model_path = 'nasa_rul_model.pkl'
    model_url = "https://github.com/mehdifr24/predictive_maintenance_pipeline/releases/download/v1.0/nasa_rul_model.pkl" 

    if not os.path.exists(model_path):
        with st.spinner("Downloading the AI model... Please wait (190 MB)"):
            urllib.request.urlretrieve(model_url, model_path)

    model_pipeline = joblib.load(model_path)
    return model_pipeline

model = get_model()

# 4. Create an upload widget for sensor data files
uploaded_file = st.file_uploader("Upload Sensor Data", type=["csv", "txt"])

if uploaded_file is not None:
    # Read the uploaded file and apply the predefined column names
    df = pd.read_csv(uploaded_file, sep='\s+', header=None)

    # Ensure dataframe matches expected columns count
    if df.shape[1] >= len(columns):
        df = df.iloc[:, :len(columns)]
        df.columns = columns
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    st.write("Data Preview:")
    st.dataframe(df.head())

# 5. Predict the Remaining Useful Life (RUL)
    if st.button("Predict Engine Life"):
        try:
            # Drop engine_id and cycle because the model was trained only on sensor features
            feature_cols = [col for col in df.columns if col not in ['engine_id', 'cycle']]
            features = df[feature_cols]

            # Pass .values to bypass scikit-learn feature names mismatch
            predictions = model.predict(features.values)

            st.success("Prediction Complete!")
            st.write("Predicted RUL (Remaining Cycles):")

            result_df = df[['engine_id', 'cycle']].copy()
            result_df['Predicted RUL'] = predictions
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
