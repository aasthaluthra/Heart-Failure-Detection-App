import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Failure Prediction App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
    }
    .low-risk {
        background-color: #e8f5e8;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load the heart failure dataset"""
    try:
        df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'heart_failure_clinical_records_dataset.csv' is in the same directory.")
        return None

# Train models and cache results
@st.cache_resource
def train_models(df):
    """Train multiple models and return the best one"""
    # Prepare data
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    
    for name, model in models.items():
        if name in ['SVM', 'Logistic Regression']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        model.fit(X_train_model, y_train)
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    return results, best_model, best_model_name, scaler, X_test, y_test, X.columns

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Age of the patient (years)',
    'anaemia': 'Decrease of red blood cells or hemoglobin (0: No, 1: Yes)',
    'creatinine_phosphokinase': 'Level of CPK enzyme in blood (mcg/L)',
    'diabetes': 'If patient has diabetes (0: No, 1: Yes)',
    'ejection_fraction': 'Percentage of blood leaving heart at each contraction (%)',
    'high_blood_pressure': 'If patient has hypertension (0: No, 1: Yes)',
    'platelets': 'Platelets in blood (kiloplatelets/mL)',
    'serum_creatinine': 'Level of serum creatinine in blood (mg/dL)',
    'serum_sodium': 'Level of serum sodium in blood (mEq/L)',
    'sex': 'Gender of patient (0: Female, 1: Male)',
    'smoking': 'If patient smokes (0: No, 1: Yes)',
    'time': 'Follow-up period (days)'
}

def main():
    st.markdown('<h1 class="main-header">❤️ Heart Failure Prediction App</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["📊 Data Overview", "🤖 Model Training", "🔮 Make Prediction", "📈 Model Analysis"])
    
    if page == "📊 Data Overview":
        show_data_overview(df)
    elif page == "🤖 Model Training":
        show_model_training(df)
    elif page == "🔮 Make Prediction":
        show_prediction_interface(df)
    elif page == "📈 Model Analysis":
        show_model_analysis(df)

def show_data_overview(df):
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Deaths", df['DEATH_EVENT'].sum())
    with col4:
        st.metric("Survival Rate", f"{(1 - df['DEATH_EVENT'].mean()) * 100:.1f}%")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    # Distribution plots
    st.subheader("Feature Distributions")
    
    # Create two columns for plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution
        fig_target = px.histogram(df, x='DEATH_EVENT', 
                                title='Target Distribution',
                                labels={'DEATH_EVENT': 'Death Event', 'count': 'Count'},
                                color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        # Age distribution by outcome
        fig_age = px.histogram(df, x='age', color='DEATH_EVENT',
                             title='Age Distribution by Outcome',
                             labels={'age': 'Age', 'count': 'Count'},
                             nbins=20)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    correlation_matrix = df.corr()
    fig_heatmap = px.imshow(correlation_matrix, 
                           text_auto=True,
                           title='Feature Correlation Matrix',
                           color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_heatmap, use_container_width=True)

def show_model_training(df):
    st.header("🤖 Model Training Results")
    
    # Train models
    with st.spinner("Training models..."):
        results, best_model, best_model_name, scaler, X_test, y_test, feature_names = train_models(df)
    
    st.success(f"✅ Models trained successfully! Best model: {best_model_name}")
    
    # Display model performance
    st.subheader("Model Performance Comparison")
    
    # Create performance dataframe
    performance_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'ROC AUC': [results[model]['roc_auc'] for model in results.keys()]
    })
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = px.bar(performance_df, x='Model', y='Accuracy',
                        title='Model Accuracy Comparison',
                        color='Accuracy',
                        color_continuous_scale='viridis')
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        fig_auc = px.bar(performance_df, x='Model', y='ROC AUC',
                        title='Model ROC AUC Comparison',
                        color='ROC AUC',
                        color_continuous_scale='viridis')
        st.plotly_chart(fig_auc, use_container_width=True)
    
    # ROC Curves
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, 
                                   name=f'{name} (AUC = {result["roc_auc"]:.3f})',
                                   mode='lines'))
    
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                               mode='lines',
                               name='Random Classifier',
                               line=dict(dash='dash')))
    
    fig_roc.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in results:
        st.subheader("Feature Importance (Random Forest)")
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(feature_importance, 
                               x='Importance', 
                               y='Feature',
                               orientation='h',
                               title='Feature Importance',
                               color='Importance',
                               color_continuous_scale='viridis')
        st.plotly_chart(fig_importance, use_container_width=True)

def show_prediction_interface(df):
    st.header("🔮 Make Prediction")
    
    # Train models if not already done
    with st.spinner("Loading model..."):
        results, best_model, best_model_name, scaler, X_test, y_test, feature_names = train_models(df)
    
    st.info(f"Using {best_model_name} model for predictions")
    
    # Create input form
    st.subheader("Patient Information")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=65, help=FEATURE_DESCRIPTIONS['age'])
        anaemia = st.selectbox("Anaemia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=FEATURE_DESCRIPTIONS['anaemia'])
        creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0, value=200, help=FEATURE_DESCRIPTIONS['creatinine_phosphokinase'])
        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=FEATURE_DESCRIPTIONS['diabetes'])
        ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=35, help=FEATURE_DESCRIPTIONS['ejection_fraction'])
        high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=FEATURE_DESCRIPTIONS['high_blood_pressure'])
    
    with col2:
        platelets = st.number_input("Platelets", min_value=0, value=250000, help=FEATURE_DESCRIPTIONS['platelets'])
        serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, value=1.0, step=0.1, help=FEATURE_DESCRIPTIONS['serum_creatinine'])
        serum_sodium = st.number_input("Serum Sodium", min_value=0, value=135, help=FEATURE_DESCRIPTIONS['serum_sodium'])
        sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help=FEATURE_DESCRIPTIONS['sex'])
        smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", help=FEATURE_DESCRIPTIONS['smoking'])
        time = st.number_input("Follow-up Time (days)", min_value=0, value=100, help=FEATURE_DESCRIPTIONS['time'])
    
    # Make prediction
    if st.button("🔮 Predict Risk", type="primary"):
        # Prepare input data
        input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                               high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                               sex, smoking, time]])
        
        # Scale input if needed
        if best_model_name in ['SVM', 'Logistic Regression']:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = best_model.predict(input_data)[0]
        probability = best_model.predict_proba(input_data)[0]
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.markdown(f'<div class="prediction-result high-risk">⚠️ HIGH RISK - Death Event Predicted</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prediction-result low-risk">✅ LOW RISK - Survival Predicted</div>', unsafe_allow_html=True)
        
        # Display probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Survival Probability", f"{probability[0]:.2%}")
        with col2:
            st.metric("Risk Probability", f"{probability[1]:.2%}")
        
        # Probability visualization
        fig_prob = go.Figure(data=[
            go.Bar(name='Probability', x=['Survival', 'Death Event'], y=[probability[0], probability[1]],
                   marker_color=['green', 'red'])
        ])
        fig_prob.update_layout(
            title='Prediction Probabilities',
            yaxis_title='Probability',
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)

def show_model_analysis(df):
    st.header("📈 Model Analysis")
    
    # Train models
    with st.spinner("Loading analysis..."):
        results, best_model, best_model_name, scaler, X_test, y_test, feature_names = train_models(df)
    
    # Model performance details
    st.subheader("Detailed Model Performance")
    
    for name, result in results.items():
        with st.expander(f"{name} Model Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{result['accuracy']:.4f}")
                st.metric("ROC AUC", f"{result['roc_auc']:.4f}")
            
            with col2:
                # Confusion matrix
                cm = confusion_matrix(y_test, result['y_pred'])
                fig_cm = px.imshow(cm, 
                                  text_auto=True,
                                  title=f'Confusion Matrix - {name}',
                                  labels=dict(x="Predicted", y="Actual"),
                                  x=['Survival', 'Death'],
                                  y=['Survival', 'Death'])
                st.plotly_chart(fig_cm, use_container_width=True)
    
    # Dataset insights
    st.subheader("Dataset Insights")
    
    # Risk factors analysis
    risk_factors = df.groupby('DEATH_EVENT').agg({
        'age': 'mean',
        'ejection_fraction': 'mean',
        'serum_creatinine': 'mean',
        'serum_sodium': 'mean',
        'time': 'mean'
    }).round(2)
    
    st.write("**Average values by outcome:**")
    st.dataframe(risk_factors)
    
    # Survival analysis by key factors
    st.subheader("Survival Analysis by Key Factors")
    
    # Age groups
    df_copy = df.copy()
    df_copy['age_group'] = pd.cut(df_copy['age'], bins=[0, 50, 60, 70, 80, 100], 
                                 labels=['<50', '50-60', '60-70', '70-80', '80+'])
    
    age_survival = df_copy.groupby(['age_group', 'DEATH_EVENT']).size().unstack(fill_value=0)
    age_survival['survival_rate'] = age_survival[0] / (age_survival[0] + age_survival[1])
    
    fig_age_survival = px.bar(
        x=age_survival.index,
        y=age_survival['survival_rate'],
        title='Survival Rate by Age Group',
        labels={'x': 'Age Group', 'y': 'Survival Rate'}
    )
    st.plotly_chart(fig_age_survival, use_container_width=True)

if __name__ == "__main__":
    main()