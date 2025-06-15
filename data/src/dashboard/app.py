import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import joblib

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train import JobFraudDetector
from utils.data_utils import generate_sample_data, validate_csv_structure, clean_job_data

# Set page config
st.set_page_config(
    page_title="Job Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Job Fraud Detection Dashboard")
st.markdown("""
This dashboard helps identify potentially fraudulent job postings using machine learning.
Upload a CSV file containing job postings to analyze them for potential fraud.
""")

# Initialize or load model
MODEL_PATH = 'model.joblib'

def initialize_model():
    """Initialize the model with sample data if no trained model exists."""
    if os.path.exists(MODEL_PATH):
        return JobFraudDetector.load_model(MODEL_PATH)
    else:
        # Generate sample data with known fraud patterns
        sample_data = generate_sample_data(n_samples=200)
        
        # Add some known fraud patterns
        fraud_patterns = [
            "work from home",
            "no experience needed",
            "immediate hiring",
            "earn money fast",
            "no investment required",
            "quick money",
            "work from anywhere",
            "no skills required",
            "get rich quick",
            "easy money"
        ]
        
        # Create synthetic fraud labels based on patterns
        sample_data['is_fraud'] = sample_data['description'].str.lower().apply(
            lambda x: any(pattern in x.lower() for pattern in fraud_patterns)
        )
        
        # Train the model
        detector = JobFraudDetector()
        detector.train(sample_data)
        detector.save_model(MODEL_PATH)
        return detector

# Initialize model in session state
if 'model' not in st.session_state:
    with st.spinner('Initializing model...'):
        st.session_state.model = initialize_model()
        st.success('Model initialized successfully!')

# File upload
uploaded_file = st.file_uploader("Upload your job postings CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate and clean the data
        if not validate_csv_structure(df):
            st.error("Invalid CSV structure. Please ensure the file contains 'title' and 'description' columns.")
        else:
            df = clean_job_data(df)
            
            # Make predictions
            results_df = st.session_state.model.predict(df)
            
            # Create a copy for display with formatted probabilities
            display_df = results_df.copy()
            display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.2%}")
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["Results Table", "Visualizations", "Top Suspicious"])
            
            with tab1:
                st.subheader("Analysis Results")
                st.dataframe(display_df)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fraud probability distribution
                    fig_hist = px.histogram(
                        results_df,
                        x='fraud_probability',
                        nbins=20,
                        title='Distribution of Fraud Probabilities'
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Pie chart of fraud vs genuine
                    fraud_counts = results_df['predicted_fraud'].value_counts()
                    labels = ['Genuine' if x == 0 else 'Fraudulent' for x in fraud_counts.index]
                    fig_pie = px.pie(
                        values=fraud_counts.values,
                        names=labels,
                        title='Distribution of Job Postings'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab3:
                st.subheader("Top 10 Most Suspicious Listings")
                # Sort by fraud probability and display top 10
                suspicious = results_df.nlargest(10, 'fraud_probability')
                for idx, row in suspicious.iterrows():
                    with st.expander(f"Job Title: {row['title']} (Fraud Probability: {row['fraud_probability']:.2%})"):
                        st.write(f"**Description:** {row['description']}")
                        if 'company' in row:
                            st.write(f"**Company:** {row['company']}")
                        if 'location' in row:
                            st.write(f"**Location:** {row['location']}")
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your CSV file contains the required columns: 'title', 'description', and optionally 'company' and 'location'.")
else:
    st.info("👆 Please upload a CSV file to begin analysis.") 