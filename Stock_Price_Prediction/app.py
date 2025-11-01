# app.py - Streamlit app to upload CSV, show EDA, run model predictions
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("Long-Term Stock Price Prediction")

uploaded = st.file_uploader("Upload stock CSV (Date,Open,High,Low,Close,Volume,...)", type=['csv'])
if not uploaded:
    st.info("Upload a CSV to begin. Example: historic OHLCV data.")
else:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]
    date_col = [c for c in df.columns if 'date' in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    st.subheader("Preview")
    st.dataframe(df.head())

    st.subheader("Price chart")
    st.line_chart(df.set_index(date_col)['Close'])

    # Option to run training locally (warning)
    if st.checkbox("Train baseline models now (may take time)"):
        st.write("Training baseline RandomForest... (this runs in browser/server)")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        # simple train target: next-day close
        df2 = df.copy()
        df2['target'] = df2['Close'].shift(-30)  # predict 30 days ahead
        df2 = df2.dropna()
        features = [c for c in df2.select_dtypes(include=[np.number]).columns if c not in ['target','Close']]
        X = df2[features]; y = df2['target']
        split = int(len(X)*0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        st.write("RF RMSE:", mean_squared_error(y_test, y_pred, squared=False))
        joblib.dump(rf, 'rf_local.joblib')
        st.success("RandomForest trained & saved to rf_local.joblib")

    # Load saved LSTM model/scalers if available
    if os.path.exists('lstm_stock_model.h5') and st.checkbox("Load saved LSTM and predict future"):
        model = load_model('lstm_stock_model.h5')
        if os.path.exists('scaler_minmax.joblib'):
            mm = joblib.load('scaler_minmax.joblib')
        else:
            mm = None
        st.write("Model loaded. Use forecast function in code to predict future prices.")

    st.markdown("**Note:** This app is a template. For production deployment, secure model files and add input validation.")
