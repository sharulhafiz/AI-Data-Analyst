import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Function to safely calculate percentage change
def safe_percentage_change(new_value, old_value):
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
    return ((new_value - old_value) / abs(old_value)) * 100

# Update the feature value calculation section
def get_current_values(data, features):
    current_values = {}
    for feature in features:
        try:
            # Calculate mean with error handling
            current_val = data[feature].mean()
            if pd.isna(current_val):
                current_val = 0
            current_values[feature] = current_val
        except KeyError:
            current_values[feature] = 0
    return current_values

# Update standardization function
def safe_standardize(data, means, stds):
    standardized = data.copy()
    for col in data.columns:
        if stds[col] != 0:
            standardized[col] = (data[col] - means[col]) / stds[col]
        else:
            standardized[col] = 0
    return standardized

# Modified objective function
def weighted_objective_function(feature_values, target_score, current_values, importance_weights):
    try:
        input_data = pd.DataFrame([feature_values], columns=top_features)
        for feature in X.columns:
            if feature not in top_features:
                input_data[feature] = data[feature].mean()
        
        input_data = input_data[X.columns]
        
        # Safe standardization
        means = features.mean()
        stds = features.std()
        input_data_standardized = input_data.copy()
        for col in input_data.columns:
            if stds[col] != 0:
                input_data_standardized[col] = (input_data[col] - means[col]) / stds[col]
            else:
                input_data_standardized[col] = 0
        
        predicted_score = model.predict(input_data_standardized)[0]
        score_diff_penalty = abs(predicted_score - target_score)
        change_penalty = np.sum(importance_weights * np.abs(feature_values - current_values))
        
        return score_diff_penalty + 0.1 * change_penalty
    except Exception as e:
        st.error(f"Error in objective function: {str(e)}")
        return float('inf')

# Load and prepare data
try:
    data = pd.read_csv('data.csv')
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(data.mean(), inplace=True)
    
    features = data.drop(columns=['SCORE_AR'])
    features_standardized = features.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
    data_standardized = pd.concat([features_standardized, data['SCORE_AR']], axis=1)
    
    df_bootstrapped = data_standardized.sample(n=1000, replace=True, random_state=42).reset_index(drop=True)
    
    X = df_bootstrapped.drop(columns=['SCORE_AR'])
    y = df_bootstrapped['SCORE_AR']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Select top 15 features
    top_features = X.columns[indices]
    top_importances = importances[indices]
    
    sorted_indices = np.argsort(top_importances)[::-1]
    top_features = top_features[sorted_indices]
    top_importances = top_importances[sorted_indices]

    
    # Create Streamlit app
    st.title('Score Optimization and Effort Analysis')
    
    # Sidebar inputs
    st.sidebar.header('Score Configuration')
    current_score = st.sidebar.number_input('Current Score', 
                                          min_value=float(max(0, y.min())), 
                                          max_value=float(min(100, y.max())), 
                                          value=float(y.mean()))
    
    desired_score = st.sidebar.number_input('Desired Score', 
                                          min_value=float(max(0, y.min())), 
                                          max_value=float(100), 
                                          value=float(y.mean()*1.2))
    
    # Display score gap
    score_gap = desired_score - current_score
    st.write(f"Score Gap to Close: {score_gap:.2f}")
    
    # Calculate bounds and initial values
    bounds = []
    current_values = []
    for feature in top_features:
        mean = features[feature].mean()
        std = features[feature].std()
        
        # Safe standardization for current value
        current_val = data[feature].mean()
        if std != 0:
            current_val = (current_val - mean) / std
        current_values.append(current_val)
        
        # Safe bounds calculation
        if std == 0:
            bounds.append((mean - 1e-5, mean + 1e-5))
        else:
            bounds.append((data[feature].min(), data[feature].max()))
    
    # Perform optimization
    importance_weights = top_importances / np.sum(top_importances)
    result = minimize(
        weighted_objective_function,
        x0=current_values,
        args=(desired_score, current_values, importance_weights),
        bounds=bounds,
        method='SLSQP'
    )
    
    optimized_values = result.x
    
    # Calculate feature impacts with safe operations
    feature_impacts = pd.DataFrame({
        'Feature': top_features,
        'Current Value': current_values,
        'Recommended Value': optimized_values,
        'Change Required': optimized_values - current_values,
        'Feature Importance': top_importances,
    })
    
    # Safe calculation of Effort Impact Score
    feature_impacts['Effort Impact Score'] = abs(feature_impacts['Change Required']) * feature_impacts['Feature Importance']
    feature_impacts = feature_impacts.sort_values('Effort Impact Score', ascending=False)
    
    # Display recommendations
    st.header('Effort Recommendations')
    st.write('Here are the most impactful efforts to achieve your desired score:')
    
    # Create recommendation categories
    high_impact = feature_impacts.head(5)
    medium_impact = feature_impacts.iloc[5:10]
    low_impact = feature_impacts.iloc[10:]
    
    # Display high impact recommendations
    st.subheader('🔥 High Impact Efforts')
    for _, row in high_impact.iterrows():
        try:
            change_pct = safe_percentage_change(row['Recommended Value'], row['Current Value'])
            
            if np.isinf(change_pct):
                if change_pct > 0:
                    direction = "increase significantly"
                else:
                    direction = "decrease significantly"
                change_txt = "(from zero)"
            else:
                direction = "increase" if change_pct > 0 else "decrease"
                change_txt = f"by {abs(change_pct):.1f}%"
            
            st.write(f"**{row['Feature']}**")
            st.write(f"- Current value: {row['Current Value']:.2f}")
            st.write(f"- Recommended value: {row['Recommended Value']:.2f}")
            st.write(f"- Recommendation: {direction} effort {change_txt}")
            st.write(f"- Impact weight: {row['Feature Importance']:.3f}")
            st.write("---")
        except Exception as e:
            st.error(f"Error processing feature {row['Feature']}: {str(e)}")
    
    # Display medium impact recommendations
    st.subheader('⚡ Medium Impact Efforts')
    medium_impact_df = medium_impact[['Feature', 'Change Required', 'Effort Impact Score']]
    st.dataframe(medium_impact_df)
    
    # Display low impact recommendations
    st.subheader('📊 Low Impact Efforts')
    low_impact_df = low_impact[['Feature', 'Change Required', 'Effort Impact Score']]
    st.dataframe(low_impact_df)
    
    # Display correlation with target score
    st.header('Feature-Score Correlations')
    correlations = pd.DataFrame({
        'Feature': top_features,
        'Correlation': data[top_features].corrwith(data['SCORE_AR']).abs()
    }).sort_values('Correlation', ascending=False)
    st.dataframe(correlations)

    # Display correlation matrix graph
    st.header('Correlation Matrix')
    corr_matrix = data[top_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Display model metrics
    st.header('Model Confidence Metrics')
    test_predictions = model.predict(X_test)
    mse = np.mean((test_predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    st.write(f"Root Mean Square Error: {rmse:.2f}")
    st.write(f"Model R² Score: {model.score(X_test, y_test):.3f}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your data format and try again.")