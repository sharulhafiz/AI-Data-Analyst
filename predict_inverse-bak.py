import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions ---
def safe_percentage_change(new_value, old_value):
    if not isinstance(new_value, (int, float)) or not isinstance(old_value, (int, float)):
        raise ValueError("Inputs must be numeric")
    if old_value == 0:
        return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
    return ((new_value - old_value) / abs(old_value)) * 100

def standardize_data(data, means=None, stds=None):
    if means is None: means = data.mean()
    if stds is None: stds = data.std()
    standardized = data.copy()
    for col in data.columns:
        if stds[col] != 0:
            standardized[col] = (data[col] - means[col]) / stds[col]
        else:
            standardized[col] = 0
    return standardized

def get_original_value(standardized_val, feature, data):
    mean, std = data[feature].mean(), data[feature].std()
    return (standardized_val * std) + mean if std != 0 else mean

def format_change_message(current, recommended):
    try:
        if abs(current) < 1e-6 and abs(recommended) < 1e-6:
            return "no change needed"
        pct_change = safe_percentage_change(recommended, current)
        if np.isinf(pct_change):
            return "increase" if recommended > 0 else "decrease"
        direction = "increase" if pct_change > 0 else "decrease"
        return f"{direction} by {abs(pct_change):.1f}%"
    except Exception as e:
        return f"Error: {str(e)}"

def optimize_for_target(model, current_features, target_score, bounds):
    def objective(x):
        return (model.predict([x])[0] - target_score) ** 2
    
    result = minimize(
        objective,
        x0=current_features,
        bounds=bounds,
        method='L-BFGS-B'
    )
    return result.x

def plot_optimization_path(current_values, recommended_values, feature_names):
    plt.figure(figsize=(10, 6))
    x = range(len(feature_names))
    
    plt.bar(x, current_values, width=0.35, label='Current', alpha=0.6)
    plt.bar([i + 0.35 for i in x], recommended_values, width=0.35, label='Recommended', alpha=0.6)
    
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Current vs Recommended Values')
    plt.xticks([i + 0.175 for i in x], feature_names, rotation=45, ha='right')
    plt.legend()
    return plt

def predict_with_constraints(model, features, target_score, bounds=None):
    if bounds is None:
        bounds = [(0, None) for _ in range(len(features))]
    
    optimized = optimize_for_target(model, features, target_score, bounds)
    predicted_score = model.predict([optimized])[0]
    
    return optimized, predicted_score

# --- Model Functions ---
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42, objective='reg:squarederror')
    model.fit(X_train, y_train)
    return model, X_test, y_test

def weighted_objective_function(feature_values, target_score, current_values, importance_weights):
    try:
        input_data = pd.DataFrame([feature_values], columns=top_features)
        means, stds = features.mean(), features.std()
        input_data_standardized = standardize_data(input_data, means, stds)
        
        predicted_score = model.predict(input_data_standardized)[0]
        score_diff_penalty = abs(predicted_score - target_score)
        change_penalty = np.sum(importance_weights * np.abs(feature_values - current_values))
        
        return score_diff_penalty + 0.1 * change_penalty
    except Exception:
        return float('inf')

# --- Visualization Functions ---
def plot_correlation_matrix(data, features):
    corr_matrix = data[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    return fig

def display_recommendations(feature_impacts, data):
    for _, row in feature_impacts.iterrows():
        try:
            feature = row['Feature']
            current = row['Current Value']
            recommended = row['Recommended Value']
            change_message = format_change_message(current, recommended)
            
            st.write(f"**{feature}**")
            st.write(f"- Current value: {current:.2f}")
            st.write(f"- Recommended value: {recommended:.2f}")
            st.write(f"- Recommendation: {change_message}")
            st.write(f"- Impact weight: {row['Feature Importance']:.3f}")
            st.write("---")
        except Exception as e:
            st.error(f"Error processing feature {feature}: {str(e)}")

# --- Main Application ---
try:
    # Load and prepare data
    data = pd.read_csv('data.csv')
    data = data.replace([np.inf, -np.inf], np.nan).fillna(data.mean())
    
    features = data.drop(columns=['SCORE_AR'])
    features_standardized = standardize_data(features)
    df_bootstrapped = pd.concat([features_standardized, data['SCORE_AR']], axis=1)
    df_bootstrapped = df_bootstrapped.sample(n=1000, replace=True, random_state=42).reset_index(drop=True)
    
    X = df_bootstrapped.drop(columns=['SCORE_AR'])
    y = df_bootstrapped['SCORE_AR']
    
    # Train model and get feature importance
    model, X_test, y_test = train_model(X, y)
    feature_importance_dict = dict(zip(X.columns, model.feature_importances_))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:15]]
    top_importances = np.array([f[1] for f in sorted_features[:15]])
    normalized_importances = top_importances / np.sum(top_importances)
    
    # Streamlit UI
    st.title('Score Optimization and Effort Analysis')
    
    # User inputs
    st.sidebar.header('Score Configuration')
    current_score = st.sidebar.number_input('Current Score', 
                                          min_value=float(max(0, y.min())), 
                                          max_value=float(min(100, y.max())), 
                                          value=float(y.mean()))
    desired_score = st.sidebar.number_input('Desired Score', 
                                          min_value=float(max(0, y.min())), 
                                          max_value=float(100), 
                                          value=float(y.mean()*1.2))
    
    # Optimization
    bounds = [(max(0, data[f].min()), data[f].max() * 1.2) for f in top_features]
    current_values = [data[f].mean() for f in top_features]
    importance_weights = normalized_importances
    
    result = minimize(weighted_objective_function,
                     x0=current_values,
                     args=(desired_score, current_values, importance_weights),
                     bounds=bounds,
                     method='SLSQP')
    
    # Calculate impact scores
    feature_impacts = pd.DataFrame({
        'Feature': top_features,
        'Current Value': current_values,
        'Recommended Value': result.x,
        'Change Required': abs(result.x - current_values),
        'Feature Importance': normalized_importances,
        'Effort Impact Score': abs(result.x - current_values) * normalized_importances
    }).sort_values('Effort Impact Score', ascending=False)
    
    # Display results
    st.write(f"Score Gap to Close: {desired_score - current_score:.2f}")
    
    st.header('Effort Recommendations')
    st.subheader('🔥 High Impact Initiatives')
    display_recommendations(feature_impacts.head(5), data)
    
    st.subheader('⚡ Medium Impact Initiatives')
    st.dataframe(feature_impacts.iloc[5:10][['Feature', 'Change Required', 'Effort Impact Score']])
    
    st.subheader('📊 Low Impact Initiatives')
    st.dataframe(feature_impacts.iloc[10:][['Feature', 'Change Required', 'Effort Impact Score']])
    
    # Display metrics
    st.header('Model Confidence Metrics')
    test_predictions = model.predict(X_test)
    rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
    r2_score = model.score(X_test, y_test)
    st.write(f"Root Mean Square Error: {rmse:.2f}")
    st.write(f"Model R² Score: {r2_score:.3f}")
    
    # Display correlations
    st.header('Feature-Score Correlations')
    correlations = pd.DataFrame({
        'Feature': top_features,
        'Correlation': data[top_features].corrwith(data['SCORE_AR']).abs()
    }).sort_values('Correlation', ascending=False)
    st.dataframe(correlations)
    
    st.header('Correlation Matrix')
    st.pyplot(plot_correlation_matrix(data, top_features))

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your data format and try again.")