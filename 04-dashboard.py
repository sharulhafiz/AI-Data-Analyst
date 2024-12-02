import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(page_title="Score Prediction Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/00_processed.csv')
    df['YEAR'] = pd.to_datetime(df['YEAR'], format='%Y')
    return df

# Load model and get top features
@st.cache_resource
def prepare_model(df):
    features = [col for col in df.columns if col not in ['YEAR', 'SCORE_AR']]
    X = df[features]
    y = df['SCORE_AR']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get top 10 features
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_features = feature_importance['feature'].head(10).tolist()
    
    return model, top_features, features  # Return all features

# Helper function for ARIMA predictions
def get_arima_forecast(data, steps=6):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Load data and prepare model once at the start
df = load_data()
model, top_features, all_features = prepare_model(df)  # Get all features

# Sidebar for page selection
page = st.sidebar.radio("Select Page", ["Target Score Prediction", "Feature-based Prediction"])

if page == "Target Score Prediction":
    st.title("Target Score Prediction")

    # Target score slider
    target_score = st.slider("Select Target Score", 
                           float(0.00), 
                           float(100.00), 
                           float(df['SCORE_AR'].mean()))

    # Calculate suggested features
    feature_importance = pd.DataFrame({
        'feature': top_features,
        'importance': model.feature_importances_[:len(top_features)]
    }).sort_values('importance', ascending=False)

    st.subheader("Suggested Feature Values")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Top Features:")
        st.dataframe(feature_importance)

if page == "Feature-based Prediction":
    st.title("Feature-based Prediction")

    # Create sliders for top features
    feature_values = {}

    # Initialize all features with mean values
    for feature in all_features:
        feature_values[feature] = df[feature].mean()
    
    # Create sliders only for top features
    for feature in top_features:
        feature_values[feature] = st.slider(
            f"Select {feature}",
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )

    # Make prediction using all features
    input_data = pd.DataFrame([feature_values])
    predicted_score = model.predict(input_data)[0]
    
    st.subheader(f"Predicted Score: {predicted_score:.2f}")
    
    # Historical and forecast visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(x=df['YEAR'], y=df['SCORE_AR'],
                            mode='lines+markers',
                            name='Historical Data'))
    
    # Forecast
    forecast = get_arima_forecast(df.set_index('YEAR')['SCORE_AR'])
    future_dates = pd.date_range(start=df['YEAR'].max(), periods=len(forecast)+1, freq='Y')[1:]
    
    fig.add_trace(go.Scatter(x=future_dates, y=forecast,
                            mode='lines+markers',
                            line=dict(dash='dash'),
                            name='Forecast'))
    
    fig.update_layout(title='Historical and Predicted Scores',
                     xaxis_title='Year',
                     yaxis_title='Score')
    
    st.plotly_chart(fig)