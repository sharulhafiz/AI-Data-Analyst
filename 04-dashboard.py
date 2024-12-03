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
    # df = pd.read_csv('data/00_processed.csv')
    df = pd.read_csv('data/00_combined_raw.csv')
    df = prepare_data(df)
    return df

def prepare_data(df):
    # Make all column names uppercase
    df.columns = df.columns.str.upper()

    # Replace special characters in column names with '_'
    df.columns = df.columns.str.replace(r'\W', '_', regex=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace 0 with forward fill and backward fill
    df = df.replace(0, np.nan).ffill().bfill()

    # Remove column where all values are 0 or NaN or missing values
    df = df.dropna(axis=1, how='all')

    # Convert all columns to numeric (round to 0 decimal places)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.round(2)

    # Move SCORE_AR to the last column
    cols = list(df.columns)
    cols.remove('SCORE_AR')
    cols.append('SCORE_AR')
    df = df[cols]

    # Convert YEAR to datetime
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

# Add helper function to get closest score and features
def get_closest_score_features(df, target_score, top_features):
    # Find the row with closest score
    closest_idx = (df['SCORE_AR'] - target_score).abs().idxmin()
    closest_row = df.iloc[closest_idx]

    # Create feature value DataFrame
    feature_values = pd.DataFrame({
        'Feature': top_features,
        'Value': [closest_row[feature] for feature in top_features],
        'Current Score': closest_row['SCORE_AR']
    })
    
    return feature_values

# An augmented dataset with the forecasted values up to SCORE_AR = 100
# Add caching to get_augmented_data
@st.cache_data
def get_augmented_data(df, model):
    # Get the last date and score
    last_date = df['YEAR'].max()
    last_score = df.loc[df['YEAR'] == last_date, 'SCORE_AR'].iloc[0]

    # Forecast score for next 100 years using cached ARIMA
    forecast = get_arima_forecast(df.set_index('YEAR')['SCORE_AR'], steps=100)
    future_dates = pd.date_range(start=last_date, periods=len(forecast)+1, freq='Y')

    # Create initial forecast DataFrame
    forecast_df = pd.DataFrame({
        'YEAR': future_dates,
        'SCORE_AR': np.concatenate([[last_score], forecast])
    })

    # Cache feature predictions
    for feature in [col for col in df.columns if col not in ['YEAR', 'SCORE_AR']]:
        feature_forecast = get_arima_forecast(df.set_index('YEAR')[feature], steps=100)
        last_feature_value = df.loc[df['YEAR'] == last_date, feature].iloc[0]
        forecast_df[feature] = np.concatenate([[last_feature_value], feature_forecast])

    return pd.concat([df, forecast_df], ignore_index=True)

# Helper function for ARIMA predictions
# Cache ARIMA model fits
@st.cache_resource
def get_arima_forecast(data, steps=6):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Load data and prepare model once at the start
df = load_data()
model, top_features, all_features = prepare_model(df)  # Get all features

# Sidebar for page selection
page = st.sidebar.radio("Select Page", ["Dataset Info", "Correlation Matrix", "Target Score Prediction", "Feature-based Prediction"])

if page == "Dataset Info":
    st.title("Dataset Info")
    st.write("This dataset contains historical data of scores for different features.")
    st.write("The goal is to predict the score based on the selected features.")
    st.write("The dataset contains the following columns/features:")
    st.write(all_features)

    # Show dataset
    st.subheader("Dataset")
    st.write(df)

if page == "Correlation Matrix":
    st.title("Correlation Matrix")

    # Get top 20 features based on importance
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    top_20_features = feature_importance['feature'].head(20).tolist()
    top_20_features.append('SCORE_AR')  # Add target variable

    # Create correlation matrix for top 20 features
    correlation_matrix = df[top_20_features].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'
    ))

    # Update layout for better readability
    fig.update_layout(
        height=800,
        width=800,
        title='Correlation Matrix (Top 20 Features)',
    )

    st.plotly_chart(fig, use_container_width=True)

if page == "Target Score Prediction":
    # Initialize aug_df only when needed
    aug_df = get_augmented_data(df, model)

    st.title("Target Score Prediction")
    
    target_score = st.slider("Select Target Score", 
                           float(0), 
                           float(100), 
                           float(df['SCORE_AR'].mean()))
    
    # Get and display closest actual values
    closest_features = get_closest_score_features(aug_df, target_score, top_features)
    
    st.subheader(f"Actual Feature Values for Score closest to {target_score:.2f}")
    st.write(f"Closest actual score: {closest_features['Current Score'].iloc[0]:.2f}")
    
    # Display feature values
    st.dataframe(closest_features[['Feature', 'Value']], hide_index=True)

if page == "Feature-based Prediction":
    st.title("Feature-based Prediction")

    # Create two columns
    left_col, right_col = st.columns([1, 2])  # 1:2 ratio for better visualization

    with left_col:
        st.subheader("Feature Inputs")
        # Create sliders for top features
        feature_values = {}

        # Initialize all features with mean values
        for feature in all_features:
            feature_values[feature] = df[feature].mean()
        
        # Create sliders only for top features
        for feature in top_features:
            feature_values[feature] = st.slider(
                f"{feature}",  # Shortened label
                float(df[feature].min()),
                float(df[feature].max()),
                float(df[feature].mean())
            )

    with right_col:
        # Make prediction using all features
        input_data = pd.DataFrame([feature_values])
        predicted_score = model.predict(input_data)[0]

        st.subheader(f"Predicted Score: {predicted_score:.2f}")

        # Historical and forecast visualization
        fig = go.Figure()

        # Historical data with year only
        fig.add_trace(go.Scatter(x=df['YEAR'].dt.strftime('%Y'),  # Convert to year only
                                y=df['SCORE_AR'],
                                mode='lines+markers',
                                name='Historical Data'))

        # Forecast
        forecast = get_arima_forecast(df.set_index('YEAR')['SCORE_AR'])
        last_date = df['YEAR'].max()
        last_score = df.loc[df['YEAR'] == last_date, 'SCORE_AR'].iloc[0]

        # Combine last historical point with forecast
        forecast_values = np.concatenate([[last_score], forecast])
        future_dates = pd.date_range(start=last_date, 
                                periods=len(forecast)+1, 
                                freq='Y')
        future_years = future_dates.strftime('%Y')

        # Add the forecast trace
        fig.add_trace(go.Scatter(x=future_years,
                                y=forecast_values,
                                mode='lines+markers',
                                line=dict(dash='dash'),
                                name='Forecast'))

        fig.update_layout(title='Historical and Predicted Scores',
                        xaxis_title='Year',
                        yaxis_title='Score',
                        height=500)

        st.plotly_chart(fig, use_container_width=True)