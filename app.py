import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# Define standard raw columns for NASA FD001
columns = ['engine_id', 'cycle', 'set_1', 'set_2', 'set_3'] + ['sensor_' + str(i) for i in range(1, 22)]

st.set_page_config(page_title="NASA Jet Engine RUL", page_icon="✈️")
st.title("✈️ Predictive Maintenance Dashboard")
st.write("Upload engine sensor data to predict Remaining Useful Life (RUL).")

@st.cache_resource
def get_model_bundle():
    model_path = 'nasa_rul_model.pkl'
    model_url = "https://github.com/mehdifr24/predictive_maintenance_pipeline/releases/download/v1.0/nasa_rul_model.pkl" 

    if not os.path.exists(model_path):
        with st.spinner("Downloading the AI model... Please wait (190 MB)"):
            urllib.request.urlretrieve(model_url, model_path)

    # Load the bundle containing both model and expected features
    bundle = joblib.load(model_path)
    return bundle

bundle = get_model_bundle()
model = bundle['model']
expected_features = bundle['features']

uploaded_file = st.file_uploader("Upload Sensor Data", type=["csv", "txt"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep='\s+', header=None)

    if df.shape[1] >= len(columns):
        df = df.iloc[:, :len(columns)]
        df.columns = columns
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    st.write("Data Preview:")
    st.dataframe(df.head())

    if st.button("Predict Engine Life"):
        try:
            # Automatically filter and align columns to match EXACTLY what the model expects
            X_inference = df[expected_features]

            predictions = model.predict(X_inference)

            st.success("Prediction Complete!")
            st.write("Predicted RUL (Remaining Cycles):")

            result_df = df[['engine_id', 'cycle']].copy() if 'engine_id' in df.columns and 'cycle' in df.columns else pd.DataFrame()
            result_df['Predicted RUL'] = predictions
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Prediction Error: {e}")
