import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Clickstream Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHING: LOAD NEW MODEL ---
@st.cache_resource
def load_spark_resources():
    try:
        # Initialize Spark Session
        spark = SparkSession.builder \
            .appName("Clickstream-Analytics") \
            .config("spark.ui.showConsoleProgress", "false") \
            .master("local[*]") \
            .getOrCreate()

        # Load the NEW pre-trained model from local path
        model_path = "/mnt/c/Users/Abdo/Desktop/clickstream/model/new_churn_model"
        model = PipelineModel.load(model_path)

        return spark, model

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None

# Load resources globally
spark, model = load_spark_resources()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
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

# --- HELPER: PREDICTION LOGIC ---
def predict_churn(df):
    """Predict churn using the new pre-trained model."""
    predictions = []
    probs = []
    
    if not model:
        st.warning("Model not loaded. Please run training script first.")
        return [], []

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
    # You can replace this with an actual image path or URL
    st.info("📊 Power BI Dashboard Preview")
    st.image("image.png", 
             caption="Interactive Power BI Dashboard - Clickstream Analytics", 
             use_container_width=True)

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
                            st.metric("High Risk", high_risk_count)
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

# Close Spark session when done
import atexit
@atexit.register
def cleanup():
    if spark:
        spark.stop()