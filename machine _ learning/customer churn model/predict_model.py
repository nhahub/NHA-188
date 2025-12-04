from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import os
import tempfile
import shutil

# PySpark and ML imports
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col, when

# GCS imports
from google.cloud import storage

# --- Configuration ---
CREDENTIALS_PATH = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"
GCS_BUCKET = "bigdata-ai-datalake"

# Paths
INPUT_NEW_DATA_PATH = "curated/new_customers_to_predict.parquet" # Data needing prediction
FEATURE_PIPELINE_PATH = "models/feature_engineering_pipeline"    # Saved from DAG 1
ML_MODEL_PATH = "models/churn_model"                  # Your saved RF Model
OUTPUT_PREDICTIONS_PATH = "predictions/churn_predictions.parquet"
# ---------------------

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'customer_churn_prediction',
    default_args=default_args,
    description='Runs inference on new data using saved Feature and ML models',
    schedule_interval=None,
    catchup=False,
    tags=['pyspark', 'prediction', 'ml']
)

def create_spark_session():
    return (SparkSession.builder
            .appName("Churn-Prediction-Pipeline")
            .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.5")
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
            .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", CREDENTIALS_PATH)
            .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
            .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
            .getOrCreate())

def download_from_gcs(gcs_path, local_path):
    """Reuse your existing download logic"""
    client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
    bucket = client.bucket(GCS_BUCKET)
    blobs = list(bucket.list_blobs(prefix=gcs_path))
    if not blobs:
        raise Exception(f"No files found at GCS path: {gcs_path}")
    
    os.makedirs(local_path, exist_ok=True)
    for blob in blobs:
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)
    print(f"Downloaded {gcs_path} to {local_path}")

def upload_to_gcs(local_path, gcs_path):
    """Reuse your existing upload logic"""
    client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
    bucket = client.bucket(GCS_BUCKET)
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            gcs_file_path = os.path.join(gcs_path, relative_path)
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)

def run_prediction_pipeline(**kwargs):
    spark = None
    temp_dirs = []
    
    try:
        spark = create_spark_session()
        
        # 1. Prepare Local Temp Directories
        data_dir = tempfile.mkdtemp()
        pipeline_dir = tempfile.mkdtemp()
        model_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        temp_dirs.extend([data_dir, pipeline_dir, model_dir, output_dir])

        # 2. Download Artifacts from GCS
        print("--- Downloading Assets ---")
        download_from_gcs(INPUT_NEW_DATA_PATH, data_dir)
        download_from_gcs(FEATURE_PIPELINE_PATH, pipeline_dir)
        download_from_gcs(ML_MODEL_PATH, model_dir)
        
        # 3. Load Data and Models
        print("--- Loading Models ---")
        new_data_df = spark.read.parquet(data_dir)
        
        # IMPORTANT: Load the pipeline model that was FITTED on training data
        feature_pipeline = PipelineModel.load(pipeline_dir)
        
        # Load the trained Random Forest model
        rf_model = RandomForestClassificationModel.load(model_dir)
        
        # 4. Transform Data (Feature Engineering)
        print("--- Applying Feature Transformations ---")
        # This applies the exact same scaling/indexing as training
        features_df = feature_pipeline.transform(new_data_df)
        
        # 5. Generate Predictions
        print("--- Running Inference ---")
        predictions = rf_model.transform(features_df)
        
        # 6. Select and Format Output
        # Usually we want the ID, the probability of churn, and the final prediction
        # Assuming 'customerID' exists. If your label was 'churn', prediction is usually 0.0/1.0
        final_results = predictions.select(
            "customerID", 
            "prediction", 
            "probability"
        )
        
        final_results.show(5)
        
        # 7. Save Predictions
        print(f"--- Saving Predictions to {output_dir} ---")
        final_results.write.mode("overwrite").parquet(output_dir)
        
        print(f"--- Uploading to GCS: {OUTPUT_PREDICTIONS_PATH} ---")
        upload_to_gcs(output_dir, OUTPUT_PREDICTIONS_PATH)
        
        print("✅ Prediction pipeline completed successfully!")

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        if spark: spark.stop()
        for d in temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

# --- Tasks ---
start_task = DummyOperator(task_id='start', dag=dag)

predict_task = PythonOperator(
    task_id='generate_churn_predictions',
    python_callable=run_prediction_pipeline,
    dag=dag
)

end_task = DummyOperator(task_id='end', dag=dag)

start_task >> predict_task >> end_task
