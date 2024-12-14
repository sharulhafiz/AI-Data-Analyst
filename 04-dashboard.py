import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore
import os

# Page configuration
st.set_page_config(page_title="Score Prediction Dashboard", layout="wide")

## ============================================ ##
## ============= Helper Functions ============= ##
## ============================================ ##

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/uploaded_dataset.csv')
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
    top_features = feature_importance['feature'].tolist()

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

# Helper function for ARIMA predictions
# Cache ARIMA model fits
@st.cache_resource
def get_arima_forecast(data, steps=6):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

@st.cache_data
# this function will generate a forecast dataset until SCORE_AR = 100
def get_forecast_dataset(df, _model):
    # Get the last date and score
    last_date = df['YEAR'].max()
    last_score = df.loc[df['YEAR'] == last_date, 'SCORE_AR'].iloc[0]

    # Set consistent forecast steps
    forecast_steps = 26  # Adjust this number as needed

    # Forecast score using ARIMA with fixed steps
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
        # Ensure non-negative predictions
        feature_forecast = np.clip(feature_forecast, 0, None)
        forecast_df[feature] = feature_forecast

        # Fit linear regression model
        X = forecast_df['YEAR'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # Convert YEAR to ordinal for regression
        y = forecast_df[feature].values
        model = LinearRegression()
        model.fit(X, y)
        trendline = model.predict(X)

        # Calculate Z-scores
        z_scores = zscore(forecast_df[feature])

        # Update only outlier values to match the trend
        outliers = np.abs(z_scores) > 3  # Define outliers as values with Z-score > 3
        forecast_df.loc[outliers, feature] = trendline[outliers]

    # For ALUMNI column, the value should be random between year 2020 to 2024 as it should not increase more that that range
    min_alumni = df['ALUMNI'].min() # min ALUMNI in df
    max_alumni = df['ALUMNI'].max() # max ALUMNI in df
    forecast_df['ALUMNI'] = np.random.randint(min_alumni, max_alumni, size=len(forecast_df))

    # Append forecast to the original dataset
    forecast_df = pd.concat([df, forecast_df], ignore_index=True)

    # Round values to 0 decimal places
    forecast_df = forecast_df.round()

    return forecast_df

# Load data and prepare model once at the start
df = load_data()
model, top_features, all_features = prepare_model(df)  # Get all features

# # Allow user to upload a new dataset
# uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
# if uploaded_file is not None:
#     # Save the uploaded file to a permanent location
#     permanent_file_path = os.path.join('data', 'uploaded_dataset.csv')
#     with open(permanent_file_path, 'wb') as f:
#         f.write(uploaded_file.getbuffer())

#     # Load the saved file
#     df = pd.read_csv(permanent_file_path)
#     df = prepare_data_improved(df)
#     model, top_features, all_features = prepare_model(df)  # Get all features

# Sidebar for years selection from main dataset using checkbox
st.sidebar.title("Filter Data")
years = st.sidebar.multiselect("Select Years", df['YEAR'].dt.year.unique(), df['YEAR'].dt.year.unique())

# Filter data based on selected years
filtered_df = df[df['YEAR'].dt.year.isin(years)]

# Sidebar for page selection
page = st.sidebar.radio("Select Page", ["Slide", "Dataset", "Dashboard (Looker)", "Correlation", "Prediction", "Knime"])

if page == "Slide":
    st.title("Slide")

    st.markdown('<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTjW8JjufEEUv8bH8dSXVvludQ4EtUK_CdYMFCIq1H3DwGOZwtEDue1hW9KX9MS4FddOI81bNbG4X1c/embed?start=false&loop=false&delayms=3000" frameborder="0" width="100%" height="600" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>', unsafe_allow_html=True)

if page == "Dashboard (Looker)":
    st.title("Dashboard (Looker)")

    st.markdown('<iframe src="https://lookerstudio.google.com/embed/reporting/b94a32a2-7470-42c2-b0c8-763f8de26526/page/p_tiw8b1sild" width="100%" height="800"></iframe>', unsafe_allow_html=True)

if page == "Dataset":
    st.title("Dataset")

    # Create tabs
    tab1, tab2 = st.tabs(["Dataset Correlation", "Forecast Dataset"])

    with tab1:
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
        st.dataframe(correlation_df, use_container_width=True, height=800)

        # Display raw data for reference
        st.write("Raw Data:")
        st.dataframe(df, height=800)

    with tab2:
        st.title("Forecast Dataset")
        
        # Get forecast dataset
        forecast_df = get_forecast_dataset(df, model)
        
        # Display forecast dataset
        st.write("Forecast Dataset:")
        st.dataframe(forecast_df, height=800)

if page == "Correlation":
    st.title("Correlation Analysis")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Features vs SCORE_AR", "Features vs Year", "Correlation Matrix", "Correlation Table"])

    with tab1:
        st.subheader("Features vs SCORE_AR")

        # Create two columns
        col1, col2 = st.columns(2)

        # Loop through all features
        for i, feature in enumerate(all_features):
            # Skip YEAR and SCORE_AR
            if feature in ['YEAR', 'SCORE_AR']:
                continue

            # Create scatter plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df[feature], y=filtered_df['SCORE_AR'], mode='markers', name='Data'))

            # Calculate correlation
            correlation = filtered_df[feature].corr(filtered_df['SCORE_AR'])

            # Fit linear regression model
            X = filtered_df[feature].values.reshape(-1, 1)
            y = filtered_df['SCORE_AR'].values
            model = LinearRegression()
            model.fit(X, y)
            trendline = model.predict(X)

            # Add trend line to the plot
            fig.add_trace(go.Scatter(x=filtered_df[feature], y=trendline, mode='lines', name='Trend Line'))

            # Update layout for better readability
            fig.update_layout(
                title=f"{feature} vs. SCORE_AR (Correlation: {correlation:.2f})",
                xaxis_title=feature,
                yaxis_title='SCORE_AR',
                height=500
            )

            # Display plots in two columns
            if i % 2 == 0:
                col1.plotly_chart(fig, use_container_width=True)
            else:
                col2.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Features vs Year")

        # Loop through all features
        for feature in all_features:
            # Skip YEAR and SCORE_AR
            if feature in ['YEAR', 'SCORE_AR']:
                continue

            # Create scatter plot
            fig = go.Figure()

            # Add feature values to the primary y-axis
            fig.add_trace(go.Scatter(x=filtered_df['YEAR'].dt.year, y=filtered_df[feature], mode='lines+markers', name=feature, yaxis='y1'))

            # Add SCORE_AR values to the secondary y-axis
            fig.add_trace(go.Scatter(x=filtered_df['YEAR'].dt.year, y=filtered_df['SCORE_AR'], mode='lines+markers', name='SCORE_AR', yaxis='y2'))

            # Calculate correlation
            correlation = filtered_df[feature].corr(filtered_df['SCORE_AR'])

            # Fit linear regression model for the feature
            X = filtered_df['YEAR'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # Convert YEAR to ordinal for regression
            y = filtered_df[feature].values
            model = LinearRegression()
            model.fit(X, y)
            trendline = model.predict(X)

            # Add trend line to the plot
            fig.add_trace(go.Scatter(x=filtered_df['YEAR'].dt.year, y=trendline, mode='lines', name=f'{feature} Trend Line', yaxis='y1'))

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

    with tab3:
        st.title("Correlation Matrix")

        # Generate RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(filtered_df[all_features], filtered_df['SCORE_AR'])

        # Get all features and sort based on importance
        feature_importance = pd.DataFrame({
            'feature': all_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Select top 20 features
        top_20_features = feature_importance['feature'].head(20).tolist()
        top_20_features.append('SCORE_AR')  # Add target variable

        # Create correlation matrix for top 20 features
        correlation_matrix = filtered_df[top_20_features].corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='Viridis'
        ))

        # Update layout for better readability
        fig.update_layout(
            height=800,
            width=800,
            title='Correlation Matrix by Features (Sorted by Importance)',
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.title("Correlation Table")

        # Calculate correlation of all features with SCORE_AR
        correlation_df = filtered_df.corr()['SCORE_AR'].sort_values(ascending=False)

        # Drop YEAR and SCORE_AR columns
        correlation_df = correlation_df.drop(['YEAR', 'SCORE_AR']).reset_index()

        # Sort by values of SCORE_AR column
        correlation_df.columns = ['Feature', 'Correlation']
        correlation_df.index = correlation_df.index + 1

        # Apply color scale to the Correlation column
        def color_scale(val):
            color = 'green' if val > 0 else 'red'
            return f'color: {color}'

        # Apply the color scale to the Correlation column
        styled_correlation_df = correlation_df.style.applymap(color_scale, subset=['Correlation'])

        # Display the styled correlation table
        st.dataframe(styled_correlation_df, use_container_width=True, height=800)

if page == "Prediction":
    st.title("Prediction Dashboard")

    # Create tabs
    tab1, tab2 = st.tabs(["Target Score Prediction", "Feature-based Prediction"])

    with tab1:
        # Initialize aug_df only when needed
        aug_df = get_forecast_dataset(df, model)

        st.subheader("Target Score Prediction")
        
        target_score = st.slider("Select Target Score", 
                               float(aug_df['SCORE_AR'].min()), 
                               float(aug_df['SCORE_AR'].max()), 
                               float(aug_df['SCORE_AR'].mean()))
        
        # Get and display closest actual values
        closest_features = get_closest_score_features(aug_df, target_score, top_features)
        
        st.subheader(f"Actual Feature Values for Score closest to {target_score:.2f}")
        st.write(f"Closest actual score: {closest_features['Current Score'].iloc[0]:.2f}")
        
        # Display feature values
        st.dataframe(closest_features[['Feature', 'Value']], hide_index=True, height=800)

    with tab2:
        st.subheader("Feature-based Prediction")

        # Initialize aug_df only when needed
        aug_df = get_forecast_dataset(df, model)

        # Retrain the model with augmented dataset
        aug_model, aug_top_features, aug_all_features, = prepare_model(aug_df)

        # Create two columns
        left_col, right_col = st.columns([1, 2])  # 1:2 ratio for better visualization

        with left_col:
            st.subheader("Feature Inputs")
            # Create sliders for top features
            feature_values = {}

            # Initialize all features with mean values
            for feature in aug_all_features:
                feature_values[feature] = df[feature].mean()
            
            # Create sliders only for top features
            for feature in top_features:
                feature_values[feature] = st.slider(
                    f"{feature}",  # Shortened label
                    float(aug_df[feature].min()),
                    float(aug_df[feature].max()),
                    float(aug_df[feature].mean())
                )

        with right_col:
            # Make prediction using all features
            input_data = pd.DataFrame([feature_values])
            predicted_score = aug_model.predict(input_data)[0]

            st.subheader(f"Predicted Score: {predicted_score:.2f}")

            # Historical and forecast visualization
            fig = go.Figure()

            # Historical data with year only
            fig.add_trace(go.Scatter(x=df['YEAR'].dt.strftime('%Y'),  # Convert to year only
                                    y=df['SCORE_AR'],
                                    mode='lines+markers',
                                    name='Historical Data'))

            # Forecast
            forecast = get_arima_forecast(df.set_index('YEAR')['SCORE_AR'],26)
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

if page == "Knime":
    st.title("Knime")

    tab1, tab2 = st.tabs(["Knime Workflow", "Knime Hub"])

    with tab1:
        st.image("https://s3.eu-central-1.amazonaws.com/knime-hubprod-catalog-service-eu-central-1/ff37bce5-ffef-4adc-864b-fb82722fd588?response-content-disposition=inline&response-content-encoding=gzip&response-content-type=image%2Fsvg%2Bxml&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241214T093545Z&X-Amz-SignedHeaders=host&X-Amz-Expires=600&X-Amz-Credential=AKIAXLA4CVAR6UW4FREN%2F20241214%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=b69cb5380e3ea6b4a88df0b8c175db9a5a37f6dca995bf9c026381a06f8f7291", width=800)

    with tab2:
        st.image("https://s3.eu-central-1.amazonaws.com/knime-hubprod-catalog-service-eu-central-1/ab32abd6-f5ea-43ac-b23a-f1d331336694?response-content-disposition=inline&response-content-encoding=gzip&response-content-type=image%2Fsvg%2Bxml&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241214T093619Z&X-Amz-SignedHeaders=host&X-Amz-Expires=600&X-Amz-Credential=AKIAXLA4CVAR6UW4FREN%2F20241214%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Signature=af1fff0c8992bc17a49b46a19a9f0f9cd8bc29e1878b8d60e4c044607bd7b3bc", width=800)