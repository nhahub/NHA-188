import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import os
import shutil
import sys
try:
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import RandomForestClassificationModel
    HAS_PYSPARK = True
    #st.success("PySpark successfully imported")
except Exception as e:
    HAS_PYSPARK = False
    st.error(f"PySpark import failed: {e}")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Project",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHING: LOAD MODEL ONCE ---
@st.cache_resource
def load_spark_resources():
    if not HAS_PYSPARK:
        st.warning("PySpark not installed, using simulation mode.")
        return None, None, None

    try:
        spark = SparkSession.builder \
            .appName("Streamlit-Churn-App") \
            .config("spark.ui.showConsoleProgress", "false") \
            .master("local[*]") \
            .getOrCreate()

        # FIX: absolute paths
        BASE = os.path.dirname(os.path.abspath(__file__))
        pipeline_path = os.path.join(BASE, "models/feature_engineering_pipeline")
        model_path = os.path.join(BASE, "models/churn_model")

        # Debug output
        #st.write("Pipeline path:", pipeline_path, os.path.exists(pipeline_path))
        #st.write("Model path:", model_path, os.path.exists(model_path))

        feature_pipeline = PipelineModel.load(pipeline_path)
        rf_model = RandomForestClassificationModel.load(model_path)

        return spark, feature_pipeline, rf_model

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None, None, None
# Load resources globally
spark, pipeline_model, rf_model = load_spark_resources()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
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
        ]
    )
    
    st.markdown("---")
    
    with st.expander("👨‍💻 About the Developer", expanded=False):
        st.write("""
        **Project Status:** Completed
        
        This project demonstrates a full End-to-End Big Data pipeline using:
        - **Apache Airflow** (Orchestration)
        - **PySpark** (Distributed Processing)
        - **Google Cloud Storage** (Data Lake)
        - **Spark ML** (Machine Learning)
        """)
        st.info("Connect with me on LinkedIn / GitHub")
        
    # Status Indicator for Model
    if HAS_PYSPARK and rf_model:
        st.success("🟢 Spark Model Loaded")
    else:
        st.warning("🟡 Simulation Mode (Model not found)")

# --- HELPER: SIMULATION LOGIC ---
def simulate_prediction(df):
    """Fallback logic if real Spark model is not available"""
    probs = []
    predictions = []
    
    for _, row in df.iterrows():
        score = 0
        # Mock Logic based on EDA
        if row.get('contract_type') == "Month-to-month": score += 40
        if row.get('internet_service') == "Fiber optic": score += 20
        if row.get('tenure_months', 12) < 12: score += 20
        if row.get('monthly_charges', 0) > 80: score += 10
        if row.get('contract_type') == "Two year": score -= 40
        
        # Random noise
        final_prob = max(0, min(100, score + np.random.randint(-5, 5)))
        probs.append(final_prob)
        predictions.append(1 if final_prob > 50 else 0)
        
    return predictions, probs

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
                    preds, probs = simulate_prediction(pdf)
                    is_churn = preds[0] == "Yes"
                    prob_churn = probs[0]
            else:
                preds, probs = simulate_prediction(pdf)
                is_churn = preds[0] == "Yes"
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
                            preds, probs = simulate_prediction(input_df)
                            final_df = input_df.copy()
                            final_df['churn_probability'] = probs
                            final_df['predicted_churn'] = preds
                    else:
                        time.sleep(1)
                        preds, probs = simulate_prediction(input_df)
                        final_df = input_df.copy()
                        final_df['churn_probability'] = probs
                        final_df['predicted_churn'] = preds
                
                st.dataframe(final_df.style.apply(lambda x: ['background-color: #ffcdd2' if v == 'Yes' else '' for v in x], subset=['predicted_churn']))
                res_csv = final_df.to_csv(index=False)
                st.download_button("⬇️ Download Predictions", res_csv, "churn_predictions.csv", "text/csv")
# --- PAGE 6: VISUALIZATION DASHBOARD ---
elif page == "Visualization Dashboard":
    st.title("📊 Interactive Dashboard")
#    st.write("Embed your Looker Studio, Tableau, or PowerBI dashboard here to show business metrics.")
    
    # ---------------------------------------------------------
    dashboard_url = "https://app.powerbi.com/view?r=eyJrIjoiMTNlM2ExMmUtODc3YS00MzkxLWJmNjMtYmFhY2M1NTMxNjEwIiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9"
    
    # Optional: UI Override for testing
 #   use_manual = st.checkbox("Use Manual URL Input", value=False)
#    if use_manual:
#        dashboard_url = st.text_input("Paste Dashboard URL here:", dashboard_url)

    st.markdown("---")
    
    if dashboard_url and "..." not in dashboard_url:
        # Adjust height as needed for your specific dashboard
        components.iframe(dashboard_url, height=800, scrolling=True)
    else:
        st.info("👆 Please replace the `dashboard_url` variable in the code with your actual Dashboard link.")
        st.image("https://placehold.co/800x400?text=Dashboard+Placeholder", caption="Dashboard will appear here")
