import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.stats import zscore
import os

# Page configuration
st.set_page_config(page_title="Score Prediction Dashboard", layout="wide")

## ============================================ ##
## ============= Helper Functions ============= ##
## ============================================ ##

def prepare_data(df):
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

# Load data
@st.cache_data
def load_data():
    # Load from gsheet https://docs.google.com/spreadsheets/d/1ngBoEOTTE-XH0dwRSYl5c6lwdDhdIch9APEuuGwqYL0/edit?gid=0#gid=0
    url = 'https://docs.google.com/spreadsheets/d/1ngBoEOTTE-XH0dwRSYl5c6lwdDhdIch9APEuuGwqYL0/gviz/tq?tqx=out:csv'
    df = pd.read_csv(url)
    df = prepare_data(df)
    return df

# Load model and sort features
@st.cache_resource
def train_model(df):
    features = [col for col in df.columns if col not in ['YEAR', 'SCORE_AR']]
    X = df[features]
    y = df['SCORE_AR']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Sort features
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    features_sorted = feature_importance['feature'].tolist()

    return model, features, features_sorted  # Return all features

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

# Helper function for forecasting

# Function to add variance to predictions
def add_variance(predictions, variance_factor=0.05):
    noise = np.random.normal(0, variance_factor * predictions.std(), predictions.shape)
    return predictions + noise

# Cache ARIMA model fits
@st.cache_resource
def get_arima_forecast(data, steps=6):
    # Prepare data for ARIMA
    data.index = pd.date_range(start=f'{data.index[0].year}-01-01', periods=len(data), freq='Y')

    model = ARIMA(data, order=order)
    model.initialize_approximate_diffuse()
    model_fit = model.fit(method='lbfgs')
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Function to evaluate models
def evaluate_models(df, features, target, order):
    X = df[features]
    y = df[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Evaluate models using cross-validation
    scores = {}
    for name, model in models.items():
        # R2 score
        r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        # RMSE score (using neg_root_mean_squared_error and converting to positive)
        rmse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
        
        scores[name] = {
            'R2': r2_scores.mean(),
            'RMSE': rmse_scores.mean()
        }

    # ARIMA model
    # Note: ARIMA requires time series data, so we'll use the entire series
    model_arima = ARIMA(y, order=order)
    model_fit = model_arima.fit()
    arima_forecast = model_fit.forecast(steps=len(y_test))
    scores['ARIMA'] = {
        'R2': r2_score(y_test, arima_forecast),
        'RMSE': np.sqrt(mean_squared_error(y_test, arima_forecast)),
        'AIC': model_fit.aic
    }

    return scores

@st.cache_data
# this function will generate a forecast dataset until SCORE_AR = 100
def get_forecast_dataset(df):
    # Get the last date and score
    last_date = df['YEAR'].max()
    last_score = df.loc[df['YEAR'] == last_date, 'SCORE_AR'].iloc[0]

    # Set target year and forecast steps
    target_year = 2040  # Adjust this year as needed
    forecast_steps = target_year - last_date.year

    # Create future dates
    future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='YE')[1:]

    # Create initial forecast DataFrame
    forecast_df = pd.DataFrame({'YEAR': future_dates})


    # Cache feature predictions with same steps
    for feature in [col for col in df.columns if col not in ['YEAR', 'SCORE_AR']]:
        # Fit linear regression model
        X = df['YEAR'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # Convert YEAR to ordinal for regression
        y = df[feature].values
        model = LinearRegression()
        model.fit(X, y)

        # Generate predictions
        future_X = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        predictions = model.predict(future_X)

        # Add variance to predictions
        predictions_with_variance = add_variance(predictions)

        # Ensure non-negative predictions
        predictions_with_variance = np.clip(predictions_with_variance, 0, None)
        forecast_df[feature] = predictions_with_variance

        # Calculate Z-scores
        z_scores = zscore(forecast_df[feature])

        # Update only outlier values to match the trend
        trendline = model.predict(future_X)
        outliers = np.abs(z_scores) > 3  # Define outliers as values with Z-score > 3
        forecast_df.loc[outliers, feature] = trendline[outliers]

    # Fit linear regression model for SCORE_AR
    X = df['YEAR'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # Convert YEAR to ordinal for regression
    y = df['SCORE_AR'].values
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions for SCORE_AR
    future_X = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    predictions = model.predict(future_X)

    # Add variance to predictions for SCORE_AR
    predictions_with_variance = add_variance(predictions)

    # Ensure non-negative predictions for SCORE_AR
    predictions_with_variance = np.clip(predictions_with_variance, 0, None)
    forecast_df['SCORE_AR'] = predictions_with_variance

    # For ALUMNI column, the value should be random between year 2020 to 2024 as it should not increase more that that range
    min_alumni = df['ALUMNI'].min() # min ALUMNI in df
    max_alumni = df['ALUMNI'].max() # max ALUMNI in df
    forecast_df['ALUMNI'] = np.random.randint(min_alumni, max_alumni, size=len(forecast_df))

    # For MENTION_POSITIVE column, the value should be random between year 2020 to 2024 as it should not increase more that that range
    min_mention = df['MENTION_POSITIVE'].min() # min MENTION_POSITIVE in df
    max_mention = df['MENTION_POSITIVE'].max() # max MENTION_POSITIVE in df
    forecast_df['MENTION_POSITIVE'] = np.random.randint(min_mention, max_mention, size=len(forecast_df))
    forecast_df['MENTION_POSITIVE'] = add_variance(forecast_df['MENTION_POSITIVE']) # add variance to MENTION_POSITIVE

    # Append forecast to the original dataset
    all_df = pd.concat([df, forecast_df], ignore_index=True)

    # Round values to 0 decimal places
    all_df = forecast_df.round()

    return forecast_df, all_df

# Load data and prepare model once at the start
df = load_data()
model_rf, features, features_sorted = train_model(df)  # Get all features

# Sidebar for years selection from main dataset using checkbox
st.sidebar.title("Filter Data")
years = st.sidebar.multiselect("Select Years", df['YEAR'].dt.year.unique(), df['YEAR'].dt.year.unique())

# Filter data based on selected years
filtered_df = df[df['YEAR'].dt.year.isin(years)]

# Sidebar for page selection
page = st.sidebar.radio("Select Page", ["Slide", "Dataset", "Dashboard (Looker)", "Correlation", "Models", "Prediction", "Knime"])

# Create sliders for each ARIMA order parameter
p = st.sidebar.slider("ARIMA Order (p)", 0, 10, 1)
d = st.sidebar.slider("ARIMA Order (d)", 0, 2, 1)
q = st.sidebar.slider("ARIMA Order (q)", 0, 10, 1)

# Combine the parameters into a tuple
order = (p, d, q)

# Display the selected order
st.sidebar.write(f"Selected ARIMA Order: {order}")

if page == "Slide":
    st.title("Slide")

    st.markdown('<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vTjW8JjufEEUv8bH8dSXVvludQ4EtUK_CdYMFCIq1H3DwGOZwtEDue1hW9KX9MS4FddOI81bNbG4X1c/embed?start=false&loop=false&delayms=3000" frameborder="0" width="100%" height="600" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>', unsafe_allow_html=True)

if page == "Dashboard (Looker)":
    st.title("Dashboard (Looker)")

    st.markdown('<iframe src="https://lookerstudio.google.com/embed/reporting/b94a32a2-7470-42c2-b0c8-763f8de26526/page/p_tiw8b1sild" width="100%" height="800"></iframe>', unsafe_allow_html=True)

if page == "Dataset":
    st.title("Dataset")

    # Create tabs
    tab1, tab2 = st.tabs(["Dataset", "Forecast Dataset"])

    with tab1:
        st.title("Raw Dataset")

        # Dataset summary
        st.write("Dataset Summary:")
        st.write(df.describe())

        # Display raw data for reference
        st.write("Raw Data:")
        st.dataframe(df)

    with tab2:
        st.title("Forecast Dataset")
        
        # Get forecast dataset
        all_df, forecast_df = get_forecast_dataset(df)
        
        # Display forecast dataset
        st.write("Forecast Dataset:")
        st.dataframe(all_df, height=800)

if page == "Correlation":
    st.title("Correlation Analysis")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Features vs SCORE_AR", "Features vs Year", "Correlation Matrix", "Correlation Table"])

    with tab1:
        st.subheader("Features vs SCORE_AR")

        # Create two columns
        col1, col2 = st.columns(2)

        # Loop through all features
        for i, feature in enumerate(features):
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
        for feature in features:
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
        model.fit(filtered_df[features], filtered_df['SCORE_AR'])

        # Get all features and sort based on importance
        feature_importance = pd.DataFrame({
            'feature': features,
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

if page == "Models":
    st.title("Models Evaluation")

    # Get forecast dataset
    forecast_df, all_df = get_forecast_dataset(df)

    # Define features and target
    features = [col for col in df.columns if col not in ['YEAR', 'SCORE_AR']]
    target = 'SCORE_AR'
    order = (5, 1, 0)  # Example ARIMA order, adjust as needed

    # Evaluate models on df
    scores_df = evaluate_models(df, features, target, order)

    # Evaluate models on all_df
    scores_all_df = evaluate_models(all_df, features, target, order)

    # Evaluate models on forecast_df
    scores_forecast_df = evaluate_models(forecast_df, features, target, order)

    # Display scores
    st.write("Model Scores on df:")
    st.write(pd.DataFrame(scores_df).T)

    st.write("Model Scores on all_df:")
    st.write(pd.DataFrame(scores_all_df).T)

    st.write("Model Scores on forecast_df:")
    st.write(pd.DataFrame(scores_forecast_df).T)


if page == "Prediction":
    st.title("Prediction Dashboard")

    # Create tabs
    tab1, tab2 = st.tabs(["Target Score Prediction", "Feature-based Prediction"])

    with tab1:
        # Initialize aug_df only when needed
        aug_df, forecast_df = get_forecast_dataset(df)

        st.subheader("Target Score Prediction")
        
        last_value = float(df['SCORE_AR'].iloc[-1])
        target_score = st.slider("Select Target Score", 
                               float(df['SCORE_AR'].min()), 
                               float(aug_df['SCORE_AR'].max()), 
                               last_value)
        
        # Get and display closest actual values
        closest_features = get_closest_score_features(aug_df, target_score, features)
        
        st.subheader(f"Actual Feature Values for Score closest to {target_score:.2f}")
        st.write(f"Closest actual score: {closest_features['Current Score'].iloc[0]:.2f}")
        
        # Display feature values
        st.dataframe(closest_features[['Feature', 'Value']], hide_index=True, height=800)

    with tab2:
        st.subheader("Feature-based Prediction")

        # Initialize aug_df only when needed
        aug_df, forecast_df = get_forecast_dataset(df)

        # Retrain the model with augmented dataset
        aug_model, aug_features, features_sorted = train_model(aug_df)

        # Create two columns
        left_col, right_col = st.columns([1, 2])  # 1:2 ratio for better visualization

        with left_col:
            st.subheader("Feature Inputs")
            # Create sliders for top features
            feature_values = {}

            # Initialize all features with mean values
            for feature in aug_features:
                feature_values[feature] = df[feature].mean()
            
            # Create sliders only for top features
            for feature in features:
                last_value = float(df[feature].iloc[-1])  # Get the last value in df for the feature
                feature_values[feature] = st.slider(
                    f"{feature}",  # Shortened label
                    float(df[feature].min()),
                    float(aug_df[feature].max()),
                    last_value  # Set the initial value to the last value in df
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
            all_df, forecast = get_forecast_dataset(df)
            last_date = df['YEAR'].max()
            last_score = df.loc[df['YEAR'] == last_date, 'SCORE_AR'].iloc[0]

            # Combine last historical point with forecast
            forecast_values = np.concatenate([[last_score], forecast['SCORE_AR'].values])
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

    tab1, tab2 = st.tabs(["Knime PreProcess", "Knime Prediction"])

    with tab1:
        st.image("https://api.hub.knime.com/repository/*HNTlN8HT6SaohePM:image?version=current-state&timestamp=1734210510000", use_container_width=True)

    with tab2:
        st.image("https://api.hub.knime.com/repository/*O6uvMJZT57yb6AnQ:image?version=current-state&timestamp=1734210550000", use_container_width=True)
