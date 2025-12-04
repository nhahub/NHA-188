import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import shutil
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import streamlit.components.v1 as components
from io import StringIO
import tempfile

# --- IMPORT OPTIONAL LIBRARIES ---
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    pass

try:
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.functions import col, when, count, sum, max, min
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import RandomForestClassificationModel
    HAS_PYSPARK = True
except Exception as e:
    HAS_PYSPARK = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# --- 1. GLOBAL PAGE CONFIGURATION ---
# This must be the first Streamlit command
st.set_page_config(
    page_title="Unified Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. NAVIGATION STATE ---
if 'current_app' not in st.session_state:
    st.session_state['current_app'] = 'Home'

def go_home():
    st.session_state['current_app'] = 'Home'
    st.rerun()

# ==============================================================================
# APP 1: CLICKSTREAM ANALYTICS (Full Logic from final.py)
# ==============================================================================
def run_clickstream_app():
    # --- HELPER FUNCTIONS FROM final.py ---
    # Renamed to avoid conflict with app.py
    @st.cache_resource
    def load_clickstream_resources():
        try:
            # Initialize Spark Session
            spark = SparkSession.builder \
                .appName("Clickstream-Analytics") \
                .config("spark.ui.showConsoleProgress", "false") \
                .master("local[*]") \
                .getOrCreate()

            # Load the NEW pre-trained model from local path
            # Using the path from your latest file attempt, but falling back to original if needed
            model_path_1 = "/home/ezz/Desktop/Handout depi/models/new_churn_model"
            model_path_2 = "/mnt/c/Users/Abdo/Desktop/clickstream/model/new_churn_model"
            
            if os.path.exists(model_path_1):
                model = PipelineModel.load(model_path_1)
            elif os.path.exists(model_path_2):
                model = PipelineModel.load(model_path_2)
            else:
                # model = PipelineModel.load("models/new_churn_model") # Generic fallback
                model = None

            return spark, model

        except Exception as e:
            st.error(f"Model load failed: {e}")
            return None, None

    # Load resources locally within the function
    spark, model = load_clickstream_resources()

    # --- HELPER: PREDICTION LOGIC ---
    def predict_churn(df):
        """Predict churn using the new pre-trained model."""
        predictions = []
        probs = []
        
        if not model:
            st.warning("Model not loaded. Please run training script first.")
            # Return dummy data for simulation if model fails
            return [1.0] * len(df), [85.5] * len(df)

        try:
            # Convert to Spark DataFrame
            sdf = spark.createDataFrame(df)
            
            # Apply the complete pipeline model
            pred_df = model.transform(sdf)

            # Collect results
            for row in pred_df.select("prediction", "probability").collect():
                predictions.append(float(row['prediction']))
                prob_vector = row['probability']
                
                if hasattr(prob_vector, 'toArray'):
                    prob_array = prob_vector.toArray()
                    churn_prob = float(prob_array[1]) * 100 if len(prob_array) > 1 else 50.0
                else:
                    churn_prob = 50.0
                    
                probs.append(churn_prob)

            return predictions, probs

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return [], []

    # --- GENERATE SAMPLE DATA FOR DASHBOARD ---
    def generate_sample_data():
        """Generate sample data for dashboard visualization"""
        np.random.seed(42)
        
        # User segments
        segments = ['High_Value_Buyer', 'Buyer', 'Active_Browser', 'Browser', 'Other']
        
        data = {
            'user_id': [f'user_{i}' for i in range(100)],
            'segment': np.random.choice(segments, 100, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
            'session_duration_seconds': np.random.randint(60, 3600, 100),
            'views': np.random.randint(1, 100, 100),
            'purchases': np.random.randint(0, 20, 100),
            'total_sales_amount': np.random.uniform(0, 2000, 100),
            'conversion_rate': np.random.uniform(0, 0.5, 100),
            'return_rate': np.random.uniform(0, 0.3, 100),
            'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], 100),
            'churn_risk': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.6, 0.3, 0.1])
        }
        
        return pd.DataFrame(data)

    # --- POWER BI DASHBOARD HELPER ---
    def display_powerbi_screenshot():
        """Display Power BI dashboard screenshot or placeholder"""
        st.info("📊 Power BI Dashboard Preview")
        # Placeholder image since local file "image.png" might not exist
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3>Power BI Dashboard Placeholder</h3>
            <p>(Actual image.png not found in upload context)</p>
        </div>
        """, unsafe_allow_html=True)

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        if st.button("⬅️ Back to Main Menu", key="back_btn_1", use_container_width=True):
            go_home()
            
        st.title("🔍 Navigation")
        
        page = st.radio(
            "Go to:",
            [
                "Project Overview", 
                "Data Engineering (ETL)", 
                "Feature Engineering", 
                "Model Training", 
                "Live Prediction Demo",
                "Visualization Dashboard",
                "Power BI Dashboard"
            ]
        )
        
        st.markdown("---")
        
        with st.expander("📊 System Status", expanded=False):
            if model:
                st.success("✅ Model: Loaded")
                st.info("🎯 Performance: AUC 0.988 | Accuracy 0.928")
            else:
                st.warning("⚠️ Model: Not Loaded")
            
            st.write("**Pipeline Components:**")
            st.write("• ✅ Data Ingestion (GitHub → GCS)")
            st.write("• ✅ Real-time Processing (Kafka → Spark)")
            st.write("• ✅ Feature Engineering")
            st.write("• ✅ ML Model Training")
            st.write("• ✅ Live Predictions")
        
        with st.expander("👨‍💻 Project Info", expanded=False):
            st.write("""
            **End-to-End Clickstream Analytics**
            
            **Tech Stack:**
            - Kafka (Real-time Streaming)
            - PySpark (Distributed Processing)  
            - Airflow (Orchestration)
            - Random Forest (ML Model)
            - Streamlit (Dashboard)
            
            **Data Sources:**
            - E-commerce Website Logs
            - Customer Transactions
            - User Behavior Data
            """)

    # --- PAGE 1: PROJECT OVERVIEW ---
    if page == "Project Overview":
        st.title("🚀 Clickstream Analytics Platform")
        
        st.markdown("""
        ### End-to-End Real-time Customer Behavior Analytics
        
        This platform processes **clickstream data** from e-commerce websites to predict customer churn 
        and provide actionable insights for business optimization.
        """)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Sources", "4", "GitHub + Real-time")
        with col2:
            st.metric("Processing Speed", "Real-time", "Kafka + Spark")
        with col3:
            st.metric("Model Performance", "98.8%", "AUC Score")
        with col4:
            st.metric("Features Engineered", "10", "Behavioral + Demographic")
        
        st.markdown("---")
        
        # Architecture Overview
        st.subheader("🏗️ System Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Data Pipeline:**
            1. **Ingestion** - Raw data from GitHub to GCS
            2. **Streaming** - Kafka for real-time processing  
            3. **Processing** - Spark for distributed analytics
            4. **Enrichment** - Feature engineering & sessionization
            5. **ML Modeling** - Random Forest for churn prediction
            6. **Visualization** - Streamlit dashboard
            """)
        
        with col2:
            st.markdown("""
            **Key Features:**
            - 📊 **Real-time Analytics** - Live customer behavior tracking
            - 🤖 **ML Predictions** - Churn risk scoring
            - 📈 **Session Analytics** - User engagement metrics
            - 🎯 **Customer Segmentation** - Behavior-based grouping
            - 🔄 **Automated Pipelines** - Airflow orchestration
            """)
        
        if model:
            st.success("🎯 **Current Status**: Model trained and ready for predictions!")
        else:
            st.warning("⚠️ **Current Status**: Please train the model first using the training script.")

    # --- PAGE 2: DATA ENGINEERING (ETL) ---
    elif page == "Data Engineering (ETL)":
        st.title("🔄 Data Engineering Pipeline")
        
        st.markdown("""
        ### Multi-source Data Ingestion & Processing
        
        Automated pipelines for collecting, processing, and storing data from various sources.
        """)
        
        # Pipeline Visualization
        st.subheader("📊 Pipeline Architecture")
        
        # Create a pipeline flow diagram
        pipeline_steps = [
            {"step": "1. Data Sources", "desc": "GitHub repositories with raw data files", "status": "✅"},
            {"step": "2. Ingestion", "desc": "Airflow DAGs download to GCS Raw Zone", "status": "✅"},
            {"step": "3. Real-time Stream", "desc": "Kafka producers stream clickstream data", "status": "✅"},
            {"step": "4. Spark Processing", "desc": "Distributed processing & parsing", "status": "✅"},
            {"step": "5. GCS Storage", "desc": "Curated data stored in data lake", "status": "✅"},
            {"step": "6. Feature Engineering", "desc": "Sessionization & metric calculation", "status": "✅"}
        ]
        
        for step in pipeline_steps:
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.success(step["step"])
            with col2:
                st.write(step["desc"])
            with col3:
                st.success(step["status"])
        
        st.markdown("---")
        
        # DAG Details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Ingestion DAG")
            st.code("""
            dag_id: data_ingestion_github_to_gcs
            Schedule: @daily
            Tasks:
            • ingest_transactions (Cust-churn.csv)
            • ingest_clickstream (E-commerce Logs.csv)  
            • ingest_complaints (consumer_complaints.rar)
            • ingest_reviews (reviews.rar)
            """, language="python")
        
        with col2:
            st.subheader("⚡ Real-time DAG")
            st.code("""
            dag_id: clickstream_pipeline_final
            Schedule: Trigger-based
            Tasks:
            • Kafka Producer (producer.py)
            • Spark Processing (spark_job.py)
            • Output Validation
            """, language="python")
        
        # Sample Data Preview
        st.subheader("📋 Sample Clickstream Data Structure")
        
        sample_data = {
            'accessed_date': ['2024-01-15 10:30:00', '2024-01-15 10:35:00'],
            'duration_secs': [120, 85],
            'ip_address': ['192.168.1.100', '192.168.1.101'],
            'bytes': [2048, 1024],
            'browser': ['Chrome', 'Firefox'],
            'age': [35, 28],
            'gender': ['Male', 'Female'],
            'country': ['US', 'UK'],
            'sales_amount': [150.0, 75.5],
            'returned': ['No', 'Yes']
        }
        
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

    # --- PAGE 3: FEATURE ENGINEERING ---
    elif page == "Feature Engineering":
        st.title("🔧 Feature Engineering")
        
        st.markdown("""
        ### Transforming Raw Clickstream Data into Predictive Features
        
        Advanced feature engineering techniques to extract meaningful patterns from user behavior data.
        """)
        
        # Feature Categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Behavioral Features")
            st.markdown("""
            - Session Duration
            - Page Views Count  
            - Purchase Frequency
            - Conversion Rate
            - Return Rate
            - Time-based Patterns
            """)
        
        with col2:
            st.subheader("👤 Demographic Features")
            st.markdown("""
            - Age Group
            - Gender
            - Geographic Location
            - Membership Type
            - Preferred Language
            """)
        
        with col3:
            st.subheader("💰 Transactional Features")
            st.markdown("""
            - Total Sales Amount
            - Average Order Value
            - Return History
            - Payment Method
            - Sales Frequency
            """)
        
        st.markdown("---")
        
        # Sessionization Process
        st.subheader("🔄 Sessionization Process")
        
        st.markdown("""
        **Raw Clickstream → User Sessions:**
        
        ```python
        # Session creation from clickstream
        sessions_df = clickstream_df.groupBy("user_id", "session_id").agg(
            (max("event_time") - min("event_time")).alias("session_duration_seconds"),
            count(when(col("event_type") == "view", True)).alias("views"),
            count(when(col("event_type") == "purchase", True)).alias("purchases"),
            sum("sales_amount").alias("total_sales_amount"),
            count(when(col("returned") == "Yes", True)).alias("returns")
        )
        ```
        """)
        
        # Feature Importance
        if model:
            st.subheader("🎯 Feature Importance (Trained Model)")
            
            importance_data = {
                'Feature': ['Return History', 'Purchase Count', 'Session Duration', 'Age', 'Sales Amount',
                        'Page Views', 'Country', 'Membership', 'Gender', 'Payment Method'],
                'Importance': [0.2696, 0.1278, 0.1274, 0.1151, 0.1068, 
                            0.0852, 0.0631, 0.0523, 0.0418, 0.0309]
            }
            
            importance_df = pd.DataFrame(importance_data)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        title='Feature Importance in Churn Prediction',
                        orientation='h', color='Importance')
            st.plotly_chart(fig, use_container_width=True)

    # --- PAGE 4: MODEL TRAINING ---
    elif page == "Model Training":
        st.title("🤖 Machine Learning Model")
        
        st.markdown("""
        ### Random Forest Classifier for Customer Churn Prediction
        
        Advanced ML model trained on behavioral and demographic features to predict customer churn risk.
        """)
        
        if model:
            st.success("✅ Model Successfully Trained and Loaded")
            
            # Model Performance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("AUC Score", "0.9878", "0.02")
            with col2:
                st.metric("Accuracy", "0.9280", "0.015")
            with col3:
                st.metric("F1 Score", "0.9342", "0.018")
            
            # Training Details
            st.subheader("📊 Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Algorithm:** Random Forest Classifier
                **Number of Trees:** 50
                **Max Depth:** 10
                **Training Samples:** 1,000
                **Test Split:** 80/20
                **Cross-Validation:** 5-fold
                """)
            
            with col2:
                st.markdown("""
                **Feature Count:** 10
                **Data Balance:** 54.6% Churn / 45.4% No Churn
                **Training Time:** ~45 seconds
                **Model Size:** ~15 MB
                **Framework:** PySpark ML
                """)
            
            # Feature Importance Visualization
            st.subheader("🔍 Feature Importance Analysis")
            
            features = ['Return History', 'Purchase Count', 'Session Duration', 'Age', 'Sales Amount',
                    'Page Views', 'Country', 'Membership', 'Gender', 'Payment Method']
            importance = [0.2696, 0.1278, 0.1274, 0.1151, 0.1068, 0.0852, 0.0631, 0.0523, 0.0418, 0.0309]
            
            fig = go.Figure(data=[
                go.Bar(name='Feature Importance', x=features, y=importance,
                    marker_color=['#FF6B6B' if x > 0.15 else '#4ECDC4' for x in importance])
            ])
            
            fig.update_layout(
                title="Feature Importance in Churn Prediction Model",
                xaxis_tickangle=-45,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model Insights
            st.subheader("💡 Key Insights")
            
            insights = [
                "📦 **Return History is the strongest predictor** of churn (26.96% importance)",
                "🛒 **Frequent purchasers are less likely** to churn",
                "⏱️ **Session duration correlates** with customer engagement", 
                "👥 **Younger age groups show** different churn patterns",
                "💰 **Higher spending customers** tend to be more loyal"
            ]
            
            for insight in insights:
                st.write(insight)
                
        else:
            st.warning("⚠️ Model Not Loaded")
            st.info("""
            To train the model, run the training script:
            ```bash
            python3 train.py
            ```
            
            This will:
            - Generate synthetic training data
            - Perform feature engineering
            - Train the Random Forest model
            - Evaluate model performance
            - Save the model for predictions
            """)

    # --- PAGE 5: LIVE PREDICTION DEMO ---
    elif page == "Live Prediction Demo":
        st.title("🔮 Live Churn Prediction Demo")
        
        if model:
            st.success("✅ Using trained model with 98.8% AUC performance")
        else:
            st.warning("⚠️ Simulation Mode - Please train model first")
        
        tab1, tab2 = st.tabs(["👤 Single Customer", "📊 Batch Analysis"])
        
        with tab1:
            st.subheader("Individual Customer Analysis")
            
            with st.form("customer_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📋 Customer Profile")
                    age = st.slider("Age", 18, 80, 35)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    country = st.selectbox("Country", ["US", "UK", "CA", "AU", "DE"])
                    membership = st.selectbox("Membership Type", ["Basic", "Premium", "Gold"])
                    
                with col2:
                    st.markdown("#### 📈 Customer Behavior")
                    sales = st.number_input("Total Sales ($)", 0, 5000, 500)
                    returned = st.selectbox("Return History", ["No", "Yes"])
                    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal"])
                    views = st.number_input("Page Views", 1, 500, 50)
                    purchases = st.number_input("Purchase Count", 0, 50, 5)
                    session_duration = st.number_input("Session Duration (seconds)", 60, 3600, 1200)
                
                submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)
            
            if submitted:
                if not model:
                    st.error("Model not loaded. Please run training first.")
                    # Simulation for demo
                    predictions = [1.0] if returned == "Yes" else [0.0]
                    probs = [85.5] if returned == "Yes" else [12.5]
                else:
                    input_data = {
                        'age': [age],
                        'gender': [gender],
                        'country': [country],
                        'membership': [membership],
                        'sales': [float(sales)],
                        'returned': [returned],
                        'payment_method': [payment_method],
                        'views': [views],
                        'purchases': [purchases],
                        'session_duration': [session_duration]
                    }
                    
                    pdf = pd.DataFrame(input_data)
                    
                    with st.spinner("Analyzing customer behavior..."):
                        predictions, probs = predict_churn(pdf)
                
                if predictions and probs:
                    st.markdown("---")
                    st.markdown("## 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        if predictions[0] == 1.0:
                            st.error("🚨 HIGH Churn Risk")
                        else:
                            st.success("✅ LOW Churn Risk")
                    
                    with col2:
                        progress_value = probs[0] / 100
                        st.progress(progress_value)
                        st.metric("Churn Probability", f"{probs[0]:.1f}%")
                    
                    with col3:
                        if probs[0] > 70:
                            st.warning("High Confidence")
                        elif probs[0] > 30:
                            st.info("Medium Confidence")
                        else:
                            st.info("Low Confidence")
                    
                    # Risk Analysis
                    with st.expander("🔍 Detailed Risk Analysis", expanded=True):
                        risk_factors = []
                        if returned == "Yes":
                            risk_factors.append("📦 Has return history")
                        if purchases < 3:
                            risk_factors.append("🛒 Low purchase frequency")
                        if session_duration < 600:
                            risk_factors.append("⏱️ Short session duration")
                        if sales < 100:
                            risk_factors.append("💰 Low spending")
                        
                        if risk_factors:
                            st.write("**Identified Risk Factors:**")
                            for factor in risk_factors:
                                st.write(f"- {factor}")
                        else:
                            st.success("✅ No major risk factors identified")
        
        with tab2:
            st.subheader("Batch Customer Analysis")
            
            st.info("Upload a CSV file with multiple customer records for batch analysis.")
            
            # Template download
            template_data = {
                'age': [35, 28, 42, 55],
                'gender': ['Male', 'Female', 'Male', 'Female'],
                'country': ['US', 'UK', 'CA', 'AU'],
                'membership': ['Premium', 'Basic', 'Gold', 'Basic'],
                'sales': [500.0, 150.0, 800.0, 300.0],
                'returned': ['No', 'Yes', 'No', 'No'],
                'payment_method': ['Credit Card', 'PayPal', 'Credit Card', 'Debit Card'],
                'views': [50, 25, 80, 35],
                'purchases': [5, 2, 8, 4],
                'session_duration': [1200, 600, 1800, 900]
            }
            template_df = pd.DataFrame(template_data)
            
            st.download_button(
                "📥 Download CSV Template",
                template_df.to_csv(index=False),
                "churn_prediction_template.csv",
                "text/csv"
            )
            
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"✅ Loaded {len(input_df)} customer records")
                st.dataframe(input_df.head())
                
                if st.button("🚀 Run Batch Prediction", type="primary"):
                    if not model:
                        st.error("Model not loaded. Please train model first.")
                        # Simulation
                        predictions = [1.0] * len(input_df)
                        probs = [75.0] * len(input_df)
                    else:
                        with st.spinner(f"Processing {len(input_df)} customers..."):
                            predictions, probs = predict_churn(input_df)
                    
                    if predictions and probs:
                        results_df = input_df.copy()
                        results_df['churn_probability'] = probs
                        results_df['predicted_churn'] = predictions
                        results_df['risk_level'] = ['HIGH' if p == 1.0 else 'LOW' for p in predictions]
                        
                        # Summary
                        high_risk_count = sum(predictions)
                        avg_prob = np.mean(probs)
                        
                        st.success(f"✅ Processed {len(results_df)} predictions")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        with col2:
                            st.metric("High Risk", int(high_risk_count))
                        with col3:
                            st.metric("Avg Probability", f"{avg_prob:.1f}%")
                        
                        st.dataframe(results_df)
                        
                        # Download
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results", 
                            csv_data, 
                            "batch_predictions.csv", 
                            "text/csv"
                        )

    # --- PAGE 6: VISUALIZATION DASHBOARD ---
    elif page == "Visualization Dashboard":
        st.title("📈 Analytics Dashboard")
        
        st.markdown("### Customer Behavior Insights & Performance Metrics")
        
        # Generate sample data for dashboard
        df = generate_sample_data()
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_session = df['session_duration_seconds'].mean() / 60
            st.metric("Avg Session (min)", f"{avg_session:.1f}")
        
        with col2:
            conversion_rate = df['conversion_rate'].mean() * 100
            st.metric("Avg Conversion Rate", f"{conversion_rate:.1f}%")
        
        with col3:
            high_value = len(df[df['segment'] == 'High_Value_Buyer'])
            st.metric("High Value Customers", high_value)
        
        with col4:
            high_risk = len(df[df['churn_risk'] == 'High'])
            st.metric("High Churn Risk", high_risk)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer Segments
            segment_counts = df['segment'].value_counts()
            fig1 = px.pie(values=segment_counts.values, names=segment_counts.index,
                        title="Customer Segments Distribution")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Session Duration by Segment
            fig3 = px.box(df, x='segment', y='session_duration_seconds',
                        title="Session Duration by Customer Segment")
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Churn Risk Distribution
            risk_counts = df['churn_risk'].value_counts()
            fig2 = px.bar(x=risk_counts.index, y=risk_counts.values,
                        title="Churn Risk Distribution",
                        labels={'x': 'Risk Level', 'y': 'Count'})
            st.plotly_chart(fig2, use_container_width=True)
            
            # Geographic Distribution
            country_counts = df['country'].value_counts()
            fig4 = px.bar(x=country_counts.index, y=country_counts.values,
                        title="Customers by Country",
                        labels={'x': 'Country', 'y': 'Count'})
            st.plotly_chart(fig4, use_container_width=True)
        
        # Behavioral Metrics
        st.subheader("📊 Behavioral Metrics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Views vs Purchases
            fig5 = px.scatter(df, x='views', y='purchases', color='churn_risk',
                            title="Views vs Purchases by Churn Risk",
                            hover_data=['segment'])
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Sales Distribution
            fig6 = px.histogram(df, x='total_sales_amount', nbins=20,
                            title="Sales Amount Distribution",
                            labels={'total_sales_amount': 'Sales Amount ($)'})
            st.plotly_chart(fig6, use_container_width=True)

    # --- PAGE 7: POWER BI DASHBOARD ---
    elif page == "Power BI Dashboard":
        st.title("📊 Power BI Advanced Analytics")
        
        st.markdown("""
        ## Interactive Power BI Dashboard for Clickstream Analytics
        
        Access our advanced Power BI dashboard for comprehensive business intelligence and 
        interactive data exploration beyond the Streamlit interface.
        """)
        
        # Dashboard Preview Section
        st.subheader("🎯 Dashboard Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            display_powerbi_screenshot()
        
        with col2:
            st.markdown("""
            ### 📋 Dashboard Features
            
            **🔍 Customer Analytics:**
            - Real-time customer behavior tracking
            - Churn risk segmentation
            - Customer lifetime value analysis
            
            **📈 Business Metrics:**
            - Conversion rate optimization
            - Revenue performance
            - Customer acquisition costs
            
            **🌍 Geographic Insights:**
            - Regional performance analysis
            - Market penetration metrics
            - Location-based trends
            """)
        
        st.markdown("---")
        
        # Download Section
        st.subheader("📥 Download Power BI Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Power BI Template")
            st.markdown("""
            Download our pre-built Power BI template to get started with your clickstream analytics.
            
            **Includes:**
            - Pre-configured data models
            - Interactive visualizations
            - Customizable reports
            - Data connection templates
            """)
            
            # Create a sample Power BI file content (in reality, you would load an actual .pbix file)
            sample_pbix_content = b"Sample Power BI file content - replace with actual .pbix file"
            
            # Download Power BI template button
            st.download_button(
                label="📥 Download Power BI Template",
                data=sample_pbix_content,
                file_name="clickstream.pbix",
                mime="application/octet-stream"
            )
        
        
        
        with col2:
            st.markdown("### 📚 Documentation")
            st.markdown("""
            Comprehensive guide for setting up and customizing your Power BI dashboard.
            
            **Topics Covered:**
            - Data source configuration
            - Report customization
            - Advanced calculations
            - Best practices
            """)
            
            # Create sample documentation content
            doc_content = "Power BI Dashboard Documentation\n\nSetup Guide:\n1. Download the .pbix file\n2. Open in Power BI Desktop\n3. Configure data sources\n4. Customize visuals as needed"
            
            # Documentation download button
            st.download_button(
                label="📥 Download Documentation",
                data=doc_content,
                file_name="powerbi_documentation.txt",
                mime="text/plain"
            )
        
        # Integration Instructions
        st.markdown("---")
        st.subheader("🔗 Integration Guide")
        
        tab1, tab2, tab3 = st.tabs(["Setup", "Customization", "Deployment"])
        
        with tab1:
            st.markdown("""
            ### ⚙️ Setup Instructions
            
            **Prerequisites:**
            - Power BI Desktop installed
            - Access to data sources
            - Appropriate permissions
            
            **Setup Steps:**
            1. Download the Power BI template above
            2. Open the .pbix file in Power BI Desktop
            3. Configure data source connections
            4. Refresh data to load latest information
            5. Save and publish to Power BI Service (optional)
            """)
        
        with tab2:
            st.markdown("""
            ### 🎨 Customizing Your Dashboard
            
            **Visualization Options:**
            - Custom charts and graphs
            - Interactive filters
            - Theme customization
            - Mobile layout optimization
            
            **Advanced Features:**
            - DAX calculations for custom metrics
            - Power Query transformations
            - Row-level security
            - Custom tooltips
            """)
        
        with tab3:
            st.markdown("""
            ### 🚀 Deployment Options
            
            **Power BI Service:**
            - Publish to Power BI Online
            - Share with team members
            - Set up automated refresh
            
            **Embedded Analytics:**
            - Embed in web applications
            - Custom development
            - API integration
            
            **On-Premises:**
            - Power BI Report Server
            - SharePoint integration
            - Local deployment
            """)
        
        # Support Section
        st.markdown("---")
        st.subheader("🆘 Need Help?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📞 Support Resources:**
            - Email: abdullah02003@gmail.com
            - Documentation: [Online Help Portal]
            - Community: [Power BI Community Forum]
            - Phone: +20 1021304688 
            """)
        
        with col2:
            st.markdown("""
            **🛠️ Technical Requirements:**
            - Power BI Desktop (latest version)
            - Internet connection for data sources
            - Appropriate data access permissions
            - Recommended: 8GB RAM minimum
            """)

    # Footer
    st.markdown("---")
    st.caption("Clickstream Analytics Platform | Built with PySpark & Streamlit | Model v2.0")

# ==============================================================================
# APP 2: SENTIMENT ANALYSIS (Full Logic from appo2.py)
# ==============================================================================
def run_sentiment_app():
    # --- HELPER FUNCTIONS ---
    @st.cache_resource
    def load_sentiment_model():
        """Load model from local files"""
        try:
            # Load from your saved model directory
            tokenizer = AutoTokenizer.from_pretrained("./saved_sentiment_model")
            model = AutoModelForSequenceClassification.from_pretrained("./saved_sentiment_model")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            return tokenizer, model, device
        except Exception as e:
            st.error(f"Error loadig model: {e}")
            return None, None, None

    def predict_sentiment(texts, tokenizer, model, device, batch_size=8):
        """Predict sentiment for a list of texts"""
        predictions = []
        confidences = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
                batch_confidences = torch.max(probabilities, dim=-1).values
            
            predictions.extend(batch_predictions.cpu().numpy())
            confidences.extend(batch_confidences.cpu().numpy())
        
        return predictions, confidences

    def map_predictions_to_labels(predictions):
        """Map numerical predictions to sentiment labels"""
        label_map = {
            0: "Very Negative", 
            1: "Negative", 
            2: "Neutral", 
            3: "Positive", 
            4: "Very Positive"
        }
        return [label_map[pred] for pred in predictions]

    def create_visualizations(df):
        """Create visualization charts"""
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            st.subheader("📈 Sentiment Distribution")
            sentiment_counts = df['predicted_sentiment'].value_counts()
            
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            colors = ['#ff6b6b', '#ffa726', '#ffee58', '#66bb6a', '#2e7d32']
            wedges, texts, autotexts = ax1.pie(
                sentiment_counts.values, 
                labels=sentiment_counts.index,
                autopct='%1.1f%%',
                colors=colors[:len(sentiment_counts)],
                startangle=90
            )
            
            # Improve aesthetics
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax1.axis('equal')
            st.pyplot(fig1)
        
        with col2:
            # Confidence distribution
            st.subheader("🎯 Confidence Scores")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.histplot(df['sentiment_confidence'], bins=20, kde=True, ax=ax2)
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Prediction Confidence')
            st.pyplot(fig2)
        
        # Sentiment by rating (if rating column exists)
        if 'rating' in df.columns:
            st.subheader("⭐ Sentiment vs Rating")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            # Create a pivot table
            pivot_data = pd.crosstab(df['rating'], df['predicted_sentiment'])
            pivot_data.plot(kind='bar', ax=ax3, figsize=(10, 6))
            
            ax3.set_xlabel('Rating')
            ax3.set_ylabel('Count')
            ax3.set_title('Sentiment Distribution by Rating')
            ax3.legend(title='Sentiment')
            ax3.tick_params(axis='x', rotation=45)
            
            st.pyplot(fig3)

    def display_metrics(df):
        """Display key metrics"""
        st.subheader("📊 Key Metrics")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_reviews = len(df)
        positive_reviews = len(df[df['predicted_sentiment'].isin(['Positive', 'Very Positive'])])
        negative_reviews = len(df[df['predicted_sentiment'].isin(['Negative', 'Very Negative'])])
        avg_confidence = df['sentiment_confidence'].mean()
        
        with col1:
            st.metric("Total Reviews", f"{total_reviews:,}")
        
        with col2:
            st.metric("Positive Reviews", f"{positive_reviews:,}", 
                    f"{(positive_reviews/total_reviews*100):.1f}%")
        
        with col3:
            st.metric("Negative Reviews", f"{negative_reviews:,}", 
                    f"{(negative_reviews/total_reviews*100):.1f}%")
        
        with col4:
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

    def process_uploaded_file(uploaded_file):
        """Process uploaded file and return DataFrame"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, Excel, or Parquet file.")
                return None
            
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None

    # --- SIDEBAR NAV ---
    with st.sidebar:
        if st.button("⬅️ Back to Main Menu", key="back_btn_sent", use_container_width=True):
            go_home()
        st.markdown("---")
        st.title("Navigation")
        app_mode = st.radio("Choose Analysis Mode:", 
                            ["Single Text Analysis", "Batch File Analysis"])

    # --- MAIN PAGE LOGIC ---
    # Title and description
    st.title("📊 Customer Sentiment Analysis Dashboard")
    st.markdown("""
    This app analyzes customer reviews and feedback using AI-powered sentiment analysis.
    Upload your data or enter text manually to get insights about customer satisfaction.
    """)

    # Initialize session state for model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    if 'model' not in st.session_state:
        st.session_state.model = None

    # Load model (only once)
    if not st.session_state.model_loaded:
        with st.spinner("Loading sentiment analysis model... This may take a minute."):
            tokenizer, model, device = load_sentiment_model()
            if tokenizer and model:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model")

    # Main application logic
    if st.session_state.model_loaded:
        if app_mode == "Single Text Analysis":
            st.header("Single Text Analysis")
            
            # Text input
            user_text = st.text_area(
                "Enter your customer review text:",
                height=150,
                placeholder="Type your review here... Example: 'I love this product! The quality is amazing and delivery was fast.'"
            )
            
            # Optional rating
            rating = st.slider("Rating (if available)", 1, 5, 3)
            
            if st.button("Analyze Sentiment") and user_text:
                with st.spinner("Analyzing sentiment..."):
                    # Predict sentiment
                    predictions, confidences = predict_sentiment(
                        [user_text], 
                        st.session_state.tokenizer, 
                        st.session_state.model, 
                        st.session_state.device
                    )
                    
                    sentiment_labels = map_predictions_to_labels(predictions)
                    confidence = confidences[0]
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment with color coding
                        sentiment_color = {
                            "Very Positive": "green",
                            "Positive": "lightgreen", 
                            "Neutral": "orange",
                            "Negative": "lightcoral",
                            "Very Negative": "red"
                        }
                        
                        st.markdown(f"""
                        **Sentiment:** <span style='color:{sentiment_color[sentiment_labels[0]]}; 
                        font-size:20px; font-weight:bold'>{sentiment_labels[0]}</span>
                        
                        **Confidence:** {confidence:.2%}
                        
                        **Rating:** {rating}/5 ⭐
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Confidence gauge
                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.barh([0], [confidence], color=sentiment_color[sentiment_labels[0]], height=0.5)
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Confidence')
                        ax.set_title('Prediction Confidence')
                        st.pyplot(fig)
        
        else:  # Batch File Analysis
            st.header("📁 Batch File Analysis")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload your customer reviews file",
                type=['csv', 'xlsx', 'parquet'],
                help="Supported formats: CSV, Excel (.xlsx)"
            )
            
            if uploaded_file is not None:
                # Process uploaded file
                df = process_uploaded_file(uploaded_file)
                
                if df is not None:
                    st.success(f"✅ File loaded successfully! {len(df)} records found.")
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Check for required columns
                    text_columns = [col for col in df.columns if any(keyword in col.lower() 
                                for keyword in ['text', 'review', 'comment', 'description', 'content'])]
                    
                    if not text_columns:
                        st.error("No text column found. Please ensure your file has a column containing review text.")
                        st.info("Common column names: 'text', 'review', 'comment', 'description'")
                    else:
                        # Let user select text column
                        text_column = st.selectbox("Select the column containing review text:", text_columns)
                        
                        if st.button("🚀 Analyze All Reviews"):
                            with st.spinner("Analyzing all reviews... This may take a while for large files."):
                                # Get texts
                                texts = df[text_column].fillna('').astype(str).tolist()
                                
                                # Predict sentiments in batches
                                predictions, confidences = predict_sentiment(
                                    texts, 
                                    st.session_state.tokenizer, 
                                    st.session_state.model, 
                                    st.session_state.device,
                                    batch_size=16
                                )
                                
                                # Add results to dataframe
                                df_result = df.copy()
                                df_result['predicted_sentiment'] = map_predictions_to_labels(predictions)
                                df_result['sentiment_confidence'] = confidences
                                df_result['predicted_sentiment_score'] = predictions
                                
                                # Store in session state
                                st.session_state.df_result = df_result
                                
                                # Display results
                                st.success("✅ Analysis completed!")
                                
                                # Show metrics
                                display_metrics(df_result)
                                
                                # Show visualizations
                                create_visualizations(df_result)
                                
                                # Show sample results
                                st.subheader("🔍 Sample Results")
                                st.dataframe(
                                    df_result[['predicted_sentiment', 'sentiment_confidence', text_column]].head(10),
                                    use_container_width=True
                                )
                                
                                # Download results
                                st.subheader("📥 Download Results")
                                
                                # Convert to different formats
                                csv = df_result.to_csv(index=False)
                                excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                                df_result.to_excel(excel_buffer.name, index=False)
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.download_button(
                                        label="📄 Download as CSV",
                                        data=csv,
                                        file_name="sentiment_analysis_results.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    with open(excel_buffer.name, 'rb') as f:
                                        st.download_button(
                                            label="📊 Download as Excel",
                                            data=f,
                                            file_name="sentiment_analysis_results.xlsx",
                                            mime="application/vnd.ms-excel"
                                        )
                                
                                # Clean up
                                #  os.unlink(excel_buffer.name)

    else:
        st.error("Model failed to load. Please check your internet connection and try again.")

# ==============================================================================
# APP 3: CUSTOMER CHURN PROJECT (Full Logic from app.py)
# ==============================================================================
def run_churn_project_app():
    # --- HELPER FUNCTIONS ---
    @st.cache_resource
    def load_churn_project_resources():
        if not HAS_PYSPARK: return None, None, None
        try:
            spark = SparkSession.builder.appName("Streamlit-Churn-App-2").master("local[*]").getOrCreate()
            # Absolute paths logic from app.py
            BASE = os.getcwd()
            pipeline_path = os.path.join(BASE, "models/feature_engineering_pipeline")
            model_path = os.path.join(BASE, "models/churn_model")
            
            if os.path.exists(pipeline_path) and os.path.exists(model_path):
                feature_pipeline = PipelineModel.load(pipeline_path)
                rf_model = RandomForestClassificationModel.load(model_path)
                return spark, feature_pipeline, rf_model
            return spark, None, None
        except: return None, None, None

    spark, pipeline_model, rf_model = load_churn_project_resources()

    def simulate_prediction_cp(df):
        """Fallback logic from app.py"""
        probs = []
        predictions = []
        for _, row in df.iterrows():
            score = 0
            if row.get('contract_type') == "Month-to-month": score += 40
            if row.get('internet_service') == "Fiber optic": score += 20
            if row.get('tenure_months', 12) < 12: score += 20
            if row.get('monthly_charges', 0) > 80: score += 10
            if row.get('contract_type') == "Two year": score -= 40
            final_prob = max(0, min(100, score + np.random.randint(-5, 5)))
            probs.append(final_prob)
            predictions.append(1 if final_prob > 50 else 0)
        return predictions, probs

    # --- SIDEBAR ---
    with st.sidebar:
        if st.button("⬅️ Back to Main Menu", key="back_btn_3", use_container_width=True):
            go_home()
        st.markdown("---")
        st.title("🔍 Project Navigation")
        
        page = st.radio(
            "Go to:",
            [
                "Project Overview", 
                "Data Engineering (ETL)", 
                "Feature Engineering", 
                "Model Training", 
                "Live Prediction Demo",
                "Visualization Dashboard"
            ],
            key="cp_nav"
        )
        
        st.markdown("---")
        if HAS_PYSPARK and rf_model:
            st.success("🟢 Spark Model Loaded")
        else:
            st.warning("🟡 Simulation Mode (Model not found)")

    # --- PAGE 1: PROJECT OVERVIEW ---
    if page == "Project Overview":
        st.title("📉 Big Data Customer Churn Prediction")
        
        st.markdown("""
        ### Executive Summary
        This project aims to predict customer churn using a distributed big data pipeline. 
        It leverages the power of **PySpark** to process large datasets stored in a **Google Cloud Data Lake**, 
        orchestrated by **Apache Airflow**.
        """)

        # Architecture Diagram
        st.subheader("System Architecture")
        st.graphviz_chart("""
            digraph {
                rankdir=LR;
                node [shape=box, style=filled, fillcolor="#f0f2f6"];
                
                Raw [label="Raw Data (CSV)\nGCS Bucket", shape=cylinder, fillcolor="#FFCDD2"];
                Airflow [label="Airflow DAGs", fillcolor="#BBDEFB"];
                Spark [label="PySpark Cluster", fillcolor="#FFF9C4"];
                Cleaned [label="Cleaned Data\n(Parquet)", shape=cylinder, fillcolor="#C8E6C9"];
                Enriched [label="Enriched Features\n(Parquet)", shape=cylinder, fillcolor="#C8E6C9"];
                Model [label="Random Forest\nModel", shape=component, fillcolor="#D1C4E9"];
                
                Raw -> Spark [label="Ingest"];
                Airflow -> Spark [label="Trigger"];
                Spark -> Cleaned [label="ETL"];
                Cleaned -> Spark [label="Feature Eng"];
                Spark -> Enriched [label="Transform"];
                Enriched -> Spark [label="Train"];
                Spark -> Model [label="Save"];
            }
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("**ETL Layer**")
            st.write("Handles missing data, duplicate removal, and schema validation using PySpark.")
        with col2:
            st.warning("**ML Pipeline**")
            st.write("String Indexing, One-Hot Encoding, and Vector Assembly for feature preparation.")
        with col3:
            st.error("**Modeling**")
            st.write("Random Forest Classifier with Grid Search Cross-Validation (CV) for hyperparameter tuning.")

    # --- PAGE 2: DATA ENGINEERING (ETL) ---
    elif page == "Data Engineering (ETL)":
        st.title("🛠️ Data Engineering Layer")
        st.write("The ETL process is managed by `air-lake-spark.py`.")
        
        st.subheader("Key Responsibilities")
        st.markdown("""
        1. **Ingestion:** Downloads raw CSV data from GCS to a temporary spark context.
        2. **Cleaning:** - Drops duplicates.
            - Imputes numeric missing values (Age, Monthly Charges) with **Mean/Median**.
            - Imputes categorical missing values (Gender, Internet Service) with **Mode**.
        3. **Storage:** Saves the processed data back to GCS in **Parquet** format for optimized querying.
        """)

        with st.expander("View ETL Code (Snippet)", expanded=True):
            st.code("""
    def clean_and_process_data(**kwargs):
        # ... (Spark Session creation) ...
        
        # 1. Drop duplicates
        df_no_duplicates = df.dropDuplicates()
        
        # 2. Calculate Statistics for Imputation
        median_age = df_no_duplicates.approxQuantile("age", [0.5], 0.0)[0]
        mean_monthly_charges = df_no_duplicates.select(F.mean("monthly_charges")).first()[0]
        mode_gender = df_no_duplicates.groupBy("gender").count().orderBy(F.desc("count")).first()[0]
        
        # 3. Fill missing values
        df_filled = df_no_duplicates.fillna({
            "age": median_age,
            "gender": mode_gender,
            "monthly_charges": mean_monthly_charges,
            "internet_service": mode_internet_service,
            "tech_support": "No"
        })
        
        # 4. Write to Parquet
        df_processed.write.mode("overwrite").parquet(processed_temp_path)
            """, language="python")

    # --- PAGE 3: FEATURE ENGINEERING ---
    elif page == "Feature Engineering":
        st.title("⚙️ Feature Engineering Pipeline")
        st.write("Handled by `Mal.py`, this DAG prepares the data for machine learning algorithms.")
        
        st.info("The pipeline converts raw business logic into mathematical vectors.")
        
        st.subheader("Pipeline Stages")
        
        stages = [
            "**String Indexer**: Converts strings (e.g., 'Male', 'Female') into indices (0, 1).",
            "**One Hot Encoder**: Converts indices into binary vectors to prevent ordinal bias.",
            "**Vector Assembler**: Combines all features (Age, Tenure, Encoded Columns) into a single `features` vector.",
            "**Standard Scaler**: Normalizes numeric features so large numbers don't dominate the model."
        ]
        
        for stage in stages:
            st.markdown(f"- {stage}")

        st.subheader("Implementation Details")
        st.code("""
    # Define pipeline stages
    stages = []

    # Stage 1: Index target variable
    label_indexer = StringIndexer(inputCol="churn", outputCol="label")

    # Stage 2-3: Index & Encode Categoricals
    cat_indexers = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cat_cols)
    cat_encoder = OneHotEncoder(inputCols=indexed_cat_cols, outputCols=ohe_cat_cols)

    # Stage 4: Vector Assemble
    final_assembler = VectorAssembler(
        inputCols=ohe_cat_cols + ["scaled_numeric_features"], 
        outputCol="features"
    )

    # Stage 5: Fit Pipeline
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
        """, language="python")

    # --- PAGE 4: MODEL TRAINING ---
    elif page == "Model Training":
        st.title("🧠 Model Training")
        st.write("The model logic is found in `model_creation.py`.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Algorithm", "Random Forest Classifier")
            st.metric("Library", "Spark ML")
        with col2:
            st.metric("Tuning Strategy", "Grid Search CV")
            st.metric("Target Metric", "Accuracy / F1-Score")
            
        st.subheader("Hyperparameter Tuning")
        st.write("We used `CrossValidator` and `ParamGridBuilder` to find the best model parameters:")
        st.json({
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 20, 30],
            "min_impurity_decrease": [0.0, 0.01, 0.05]
        })
        
        st.subheader("Code Snippet")
        st.code("""
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy"
    )

    grid.fit(X, y)
    best_model = grid.best_estimator_
        """, language="python")

    # --- PAGE 5: LIVE PREDICTION DEMO ---
    elif page == "Live Prediction Demo":
        st.title("🔮 Churn Prediction Service")
        
        tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 Batch File Upload"])
        
        # --- TAB 1: SINGLE CUSTOMER ---
        with tab1:
            st.subheader("Customer Profile")
            with st.form("churn_form"):
                col1, col2, col3 = st.columns(3)
                
                # COLUMN 1: Demographics
                with col1:
                    st.markdown("##### Demographics")
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    age = st.slider("Age", 18, 80, 30)
                    
                # COLUMN 2: Account Info
                with col2:
                    st.markdown("##### Account Info")
                    tenure = st.slider("Tenure (Months)", 0, 72, 12)
                    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
                    payment = st.selectbox("Payment Method", ['Mailed Check' ,'Bank Transfer', 'Electronic Check', 'Credit Card'])
                    
                # COLUMN 3: Services
                with col3:
                    st.markdown("##### Services")
                    monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 60.0)
                    internet = st.selectbox("Internet Service", ['None', 'Fiber', 'DSL', 'Unknown'])
                    tech_support = st.selectbox("Tech Support", ["Yes", "No", 'Unknown'])
                    
                # Calculated automatically
                total_charges = tenure * monthly
                
                st.caption(f"**Calculated Total Charges:** ${total_charges:.2f}")
                    
                submit = st.form_submit_button("Predict Churn Probability")
                
            if submit:
                # Create DataFrame with ALL fields
                input_data = {
                    'customer_id': ["DEMO_USER"],
                    'age': [float(age)],
                    'gender': [gender],
                    'tenure_months': [int(tenure)],
                    'monthly_charges': [float(monthly)],
                    'contract_type': [contract],
                    'internet_service': [internet],
                    'tech_support': [tech_support],
                    'payment_method': [payment],
                    'total_charges': [float(total_charges)]
                }
                pdf = pd.DataFrame(input_data)

                st.markdown("### Prediction Result")
                
                if spark and rf_model:
                    try:
                        sdf = spark.createDataFrame(pdf)
                        transformed = pipeline_model.transform(sdf)
                        pred = rf_model.transform(transformed)
                        result = pred.select("probability", "prediction").collect()[0]
                        
                        prob_churn = result['probability'][1] * 100
                        is_churn = result['prediction'] == 1.0
                    except Exception as e:
                        st.error(f"Spark Error: {e}")
                        preds, probs = simulate_prediction_cp(pdf)
                        is_churn = preds[0] == 1
                        prob_churn = probs[0]
                else:
                    preds, probs = simulate_prediction_cp(pdf)
                    is_churn = preds[0] == 1
                    prob_churn = probs[0]
                
                col_res1, col_res2 = st.columns([1, 3])
                with col_res1:
                    if is_churn:
                        st.error("Churn: YES")
                    else:
                        st.success("Churn: NO")
                with col_res2:
                    st.progress(int(prob_churn) / 100)
                    st.caption(f"Probability: {prob_churn:.2f}%")

        # --- TAB 2: BATCH FILE UPLOAD ---
        with tab2:
            st.subheader("Bulk Prediction via CSV")
            st.info("Upload a CSV file containing customer data.")
            
            # Template reflects ALL columns now
            sample_data = {
                'customer_id': ['CUST007702', 'CUST079297'],
                'age': [30.0, 0.0],
                'gender': ['Male', 'Male'],
                'tenure_months': [53, 57],
                'monthly_charges': [67.75, 58.67],
                'contract_type': ['Two Year', 'Month-to-Month'],
                'internet_service': ['Fiber', 'Unknown'],
                'tech_support': ['Yes', 'No'],
                'payment_method': ['Electronic Check', 'Credit Card'],
                'total_charges': [3595.78, 3368.65]
            }
            template_df = pd.DataFrame(sample_data)
            csv = template_df.to_csv(index=False)
            st.download_button("⬇️ Download CSV Template", csv, "churn_input_template.csv", "text/csv")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded **{len(input_df)}** rows.")
                
                if st.button("Run Batch Prediction"):
                    with st.spinner("Running Inference..."):
                        if spark and rf_model:
                            try:
                                sdf = spark.createDataFrame(input_df)
                                transformed = pipeline_model.transform(sdf)
                                predictions = rf_model.transform(transformed)
                                
                                results_spark = predictions.select("customer_id", "prediction", "probability").toPandas()
                                results_spark['churn_probability'] = results_spark['probability'].apply(lambda x: round(float(x[1])*100, 2))
                                results_spark['predicted_churn'] = results_spark['prediction'].apply(lambda x: "Yes" if x == 1.0 else "No")
                                
                                final_df = input_df.merge(results_spark[['customer_id', 'predicted_churn', 'churn_probability']], on='customer_id', how='left')
                            except Exception as e:
                                st.error(f"Spark Batch Failed: {e}")
                                preds, probs = simulate_prediction_cp(input_df)
                                final_df = input_df.copy()
                                final_df['churn_probability'] = probs
                                final_df['predicted_churn'] = ["Yes" if p==1 else "No" for p in preds]
                        else:
                            time.sleep(1)
                            preds, probs = simulate_prediction_cp(input_df)
                            final_df = input_df.copy()
                            final_df['churn_probability'] = probs
                            final_df['predicted_churn'] = ["Yes" if p==1 else "No" for p in preds]
                    
                    st.dataframe(final_df.style.apply(lambda x: ['background-color: #ffcdd2' if v == 'Yes' else '' for v in x], subset=['predicted_churn']))
                    res_csv = final_df.to_csv(index=False)
                    st.download_button("⬇️ Download Predictions", res_csv, "churn_predictions.csv", "text/csv")

    # --- PAGE 6: VISUALIZATION DASHBOARD ---
    elif page == "Visualization Dashboard":
        st.title("📊 Interactive Dashboard")
        
        # ---------------------------------------------------------
        dashboard_url = "https://app.powerbi.com/view?r=eyJrIjoiMTNlM2ExMmUtODc3YS00MzkxLWJmNjMtYmFhY2M1NTMxNjEwIiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9"
        
        st.markdown("---")
        
        if dashboard_url and "..." not in dashboard_url:
            components.iframe(dashboard_url, height=800, scrolling=True)
        else:
            st.info("👆 Please replace the `dashboard_url` variable in the code with your actual Dashboard link.")
            st.image("https://placehold.co/800x400?text=Dashboard+Placeholder", caption="Dashboard will appear here")

# ==============================================================================
# MAIN ROUTER
# ==============================================================================
def main():
    if st.session_state['current_app'] == 'Home':
        st.title("🏢 Unified Analytics Platform")
        st.markdown("### Select an application to launch")
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.container(height=300, border=True).markdown("### 🚀 Clickstream Analytics\n\nReal-time customer behavior analytics and churn prediction based on clickstream data.")
            if col1.button("Launch Clickstream App", use_container_width=True):
                st.session_state['current_app'] = 'Clickstream'
                st.rerun()
        with col2:
            st.container(height=300, border=True).markdown("### 📉 Churn Prediction Project\n\nEnd-to-End Big Data pipeline handling ETL, Feature Engineering, and Model Training.")
            if col2.button("Launch Churn App", use_container_width=True):
                st.session_state['current_app'] = 'ChurnProject'
                st.rerun()
        with col3:
            st.container(height=300, border=True).markdown("### 😊 Sentiment Analysis\n\nAI-powered sentiment classification for customer reviews (Single & Batch).")
            if col3.button("Launch Sentiment App", use_container_width=True):
                st.session_state['current_app'] = 'Sentiment'
                st.rerun()

    elif st.session_state['current_app'] == 'Clickstream': run_clickstream_app()
    elif st.session_state['current_app'] == 'ChurnProject': run_churn_project_app()
    elif st.session_state['current_app'] == 'Sentiment': run_sentiment_app()

if __name__ == "__main__":
    main()
