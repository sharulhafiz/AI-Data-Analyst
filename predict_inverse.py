import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def map_to_full_features(partial_values, top_features, all_features, original_data):
    """Map partial feature values back to full feature vector"""
    full_values = original_data.mean().values  # Get mean values for all features
    top_indices = [list(all_features).index(f) for f in top_features]
    full_values[top_indices] = partial_values
    return full_values

def incremental_adjustment(current_features, target_score, model, importance_weights, 
                         top_features, all_features, original_data, steps=100):
    for step in range(steps):
        # Map to full feature vector for prediction
        full_features = map_to_full_features(current_features, top_features, 
                                           all_features, original_data)
        prediction = model.predict([full_features])[0]
        
        if abs(prediction - target_score) < 0.1:
            break
            
        # Adjust only top features
        for i, importance in enumerate(importance_weights):
            if prediction < target_score:
                current_features[i] += importance * 0.1
            else:
                current_features[i] -= importance * 0.1
                
    return current_features

# Load and prepare data
data = pd.read_csv('data/01_augmented.csv')
data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())
# Sample data to 1000 rows
data = data.sample(n=1000, random_state=42, replace=True)

X = data.drop(columns=['SCORE_AR'])
y = data['SCORE_AR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Get feature importance and top features
feature_importances = model.feature_importances_
top_features = X.columns[np.argsort(-feature_importances)][:15]
importance_weights = feature_importances[np.argsort(-feature_importances)][:15]

# UI inputs
current_score = st.sidebar.number_input('Current Score', min_value=0.0, max_value=100.0, value=42.0)
desired_score = st.sidebar.number_input('Desired Score', min_value=0.0, max_value=100.0, value=55.0)
current_values = X[top_features].mean().values

# Optimize
recommended_values = incremental_adjustment(
    current_values.copy(), 
    desired_score,
    model,
    importance_weights,
    top_features,
    X.columns,
    X
)

# Get full feature prediction
full_recommended = map_to_full_features(recommended_values, top_features, X.columns, X)
predicted_score = model.predict([full_recommended])[0]

# Display results
st.title("Score Optimization Insights")
st.write(f"Desired Score: {desired_score}")
st.write(f"Achieved Score: {predicted_score:.2f}")

recommendations = pd.DataFrame({
    'Feature': top_features,
    'Current Value': current_values,
    'Recommended Value': recommended_values,
    'Change Required': recommended_values - current_values
})
st.dataframe(recommendations)