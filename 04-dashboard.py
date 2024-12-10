import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Score Prediction Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/00_processed.csv')
    # df = pd.read_csv('data/00_combined_raw.csv')
    df = prepare_data_improved(df)
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

def prepare_data_improved(df):
    # Make all column names uppercase
    df.columns = df.columns.str.upper()

    # Replace special characters in column names with '_'
    df.columns = df.columns.str.replace(r'\W', '_', regex=True)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Replace 0 with NaN to identify missing or noisy data
    df = df.replace(0, np.nan)

    # Interpolate missing values
    df = df.interpolate(method='linear', limit_direction='both')

    # Forward fill and backward fill as fallback methods
    df = df.ffill().bfill()

    # In each column, if any value variation is less than a threshold, replace with interpolated values
    threshold = 5  # Adjust this value as needed
    for col in df.columns:
        if df[col].max() - df[col].min() < threshold:
            df[col] = df[col].interpolate(method='linear', limit_direction='both').ffill().bfill()

    # Remove columns where all values are NaN or missing
    df = df.dropna(axis=1, how='all')

    # Convert all columns to numeric (round to 2 decimal places)
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
# Add caching to get_augmented_data with ignored model parameter
@st.cache_data
def get_augmented_data(df, _model):
    # Get the last date and score
    last_date = df['YEAR'].max()
    last_score = df.loc[df['YEAR'] == last_date, 'SCORE_AR'].iloc[0]

    # Set consistent forecast steps
    forecast_steps = 6  # Adjust this number as needed

    # Forecast score using cached ARIMA with fixed steps
    forecast = get_arima_forecast(df.set_index('YEAR')['SCORE_AR'], steps=forecast_steps)
    future_dates = pd.date_range(start=last_date, periods=len(forecast)+1, freq='Y')[1:]

    # Create initial forecast DataFrame
    forecast_df = pd.DataFrame({
        'YEAR': future_dates,
        'SCORE_AR': forecast
    })

    # Cache feature predictions with same steps
    for feature in [col for col in df.columns if col not in ['YEAR', 'SCORE_AR']]:
        feature_forecast = get_arima_forecast(df.set_index('YEAR')[feature], steps=forecast_steps)
        forecast_df[feature] = feature_forecast

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

# Allow user to upload a new dataset
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = prepare_data_improved(df)
    model, top_features, all_features = prepare_model(df)  # Get all features



# Sidebar for page selection
page = st.sidebar.radio("Select Page", ["Dashboard (Looker)", "Dataset Correlation", "Correlation Analysis (Features vs SCORE_AR)", "Correlation Analysis (Features vs Year)", "Correlation Matrix", "Target Score Prediction", "Feature-based Prediction"])

if page == "Dashboard (Looker)":
    st.title("Dashboard (Looker)")
    # st.markdown("This is a placeholder for the Looker dashboard.")

    # Embed looker studio link https://lookerstudio.google.com/embed/reporting/b94a32a2-7470-42c2-b0c8-763f8de26526/page/p_tiw8b1sild
    st.markdown('<iframe src="https://lookerstudio.google.com/embed/reporting/b94a32a2-7470-42c2-b0c8-763f8de26526/page/p_tiw8b1sild" width="100%" height="800"></iframe>', unsafe_allow_html=True)

if page == "Dataset Correlation":
    st.title("Dataset Correlation")
    st.write("The dataset Correlation of columns/features with with SCORE_AR (sorted):")

    # Show dataset
    # Calculate correlation with SCORE_AR and sort by highest correlation, excluding YEAR column
    correlation_with_score = df.drop(columns=['YEAR']).corr()['SCORE_AR'].sort_values(ascending=False)
    
    # Convert to DataFrame and hide SCORE_AR from display
    correlation_df = correlation_with_score.drop('SCORE_AR').reset_index()
    correlation_df.columns = ['Feature', 'Correlation']
    correlation_df.index = correlation_df.index + 1  # Start index from 1

    # Display the correlation table in full width and height
    st.dataframe(correlation_df, use_container_width=True)

if page == "Correlation Analysis (Features vs SCORE_AR)":
    # This page will loop all features and calculate correlation with SCORE_AR
    # Each loop will create a scatter plot and display the correlation value
    st.title("Correlation Analysis")

    # Loop through all features
    for feature in all_features:
        # Skip YEAR and SCORE_AR
        if feature in ['YEAR', 'SCORE_AR']:
            continue

        # Create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[feature], y=df['SCORE_AR'], mode='markers', name='Data'))

        # Calculate correlation
        correlation = df[feature].corr(df['SCORE_AR'])

        # Fit linear regression model
        X = df[feature].values.reshape(-1, 1)
        y = df['SCORE_AR'].values
        model = LinearRegression()
        model.fit(X, y)
        trendline = model.predict(X)

        # Add trend line to the plot
        fig.add_trace(go.Scatter(x=df[feature], y=trendline, mode='lines', name='Trend Line'))

        # Update layout for better readability
        fig.update_layout(
            title=f"{feature} vs. SCORE_AR (Correlation: {correlation:.2f})",
            xaxis_title=feature,
            yaxis_title='SCORE_AR',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

if page == "Correlation Analysis (Features vs Year)":
    # This page will loop all features and calculate correlation with SCORE_AR
    # Each loop will create a scatter plot and display the correlation value
    st.title("Correlation Analysis")
    
    # Loop through all features
    for feature in all_features:
        # Skip YEAR and SCORE_AR
        if feature in ['YEAR', 'SCORE_AR']:
            continue

        # Create scatter plot
        fig = go.Figure()

        # Add feature values to the primary y-axis
        fig.add_trace(go.Scatter(x=df['YEAR'].dt.year, y=df[feature], mode='lines+markers', name=feature, yaxis='y1'))

        # Add SCORE_AR values to the secondary y-axis
        fig.add_trace(go.Scatter(x=df['YEAR'].dt.year, y=df['SCORE_AR'], mode='lines+markers', name='SCORE_AR', yaxis='y2'))

        # Calculate correlation
        correlation = df[feature].corr(df['SCORE_AR'])

        # Fit linear regression model for the feature
        X = df['YEAR'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # Convert YEAR to ordinal for regression
        y = df[feature].values
        model = LinearRegression()
        model.fit(X, y)
        trendline = model.predict(X)

        # Add trend line to the plot
        fig.add_trace(go.Scatter(x=df['YEAR'].dt.year, y=trendline, mode='lines', name=f'{feature} Trend Line', yaxis='y1'))

        # Update layout for better readability
        fig.update_layout(
            title=f"{feature} vs. SCORE_AR (Correlation: {correlation:.2f})",
            xaxis_title='YEAR',
            yaxis=dict(title=feature, side='left'),
            yaxis2=dict(title='SCORE_AR', side='right', overlaying='y'),
            height=500,
            xaxis=dict(tickmode='linear', tick0=df['YEAR'].dt.year.min(), dtick=1)  # Ensure x-axis shows only whole years
        )

        st.plotly_chart(fig, use_container_width=True)

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