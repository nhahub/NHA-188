# apache-airflow==2.7.1
# pyspark==3.4.0
# google-cloud-storage==2.8.0

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
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)

# GCS imports
from google.cloud import storage

# --- Configuration ---
CREDENTIALS_PATH = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"
GCS_BUCKET = "bigdata-ai-datalake"

# Input path for the cleaned data from the previous DAG
INPUT_DATA_PATH = f"curated/customers_churn.parquet"

# Output paths for ML artifacts
TRAIN_DATA_PATH = f"enriched/churn_train.parquet"
TEST_DATA_PATH = f"enriched/churn_test.parquet"
# ---------------------

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'customer_churn_ml_preparation_1',
    default_args=default_args,
    description='Prepares customer churn data for ML modeling',
    schedule_interval=None,  # This DAG can be triggered manually or after the first one
    catchup=False,
    tags=['pyspark', 'gcs', 'ml', 'features']
)

def create_spark_session():
    """Create and return Spark session using the same configuration as your working file"""
    spark = (
        SparkSession.builder
        .appName("BigDataProject-ML-Preparation")
        .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.5")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", CREDENTIALS_PATH)
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .getOrCreate()
    )
    return spark

def download_from_gcs(gcs_path, local_path):
    """Download file/directory from GCS to local path"""
    client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
    bucket = client.bucket(GCS_BUCKET)
    
    # Check if it's a directory (parquet files are stored as directories with part files)
    blobs = list(bucket.list_blobs(prefix=gcs_path))
    
    if not blobs:
        raise Exception(f"No files found at GCS path: {gcs_path}")
    
    # Create local directory
    os.makedirs(local_path, exist_ok=True)
    
    # Download all files in the directory
    for blob in blobs:
        # Get relative path within the directory
        relative_path = os.path.relpath(blob.name, gcs_path)
        local_file_path = os.path.join(local_path, relative_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # Download file
        blob.download_to_filename(local_file_path)
        print(f"Downloaded: {blob.name} -> {local_file_path}")

def upload_to_gcs(local_path, gcs_path):
    """Upload local directory to GCS"""
    client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
    bucket = client.bucket(GCS_BUCKET)
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            gcs_file_path = os.path.join(gcs_path, relative_path)
            
            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded: {local_file_path} -> {gcs_file_path}")

def prepare_features_and_save(**kwargs):
    """
    Loads cleaned data, runs a feature engineering pipeline, splits the data,
    and saves the train/test sets to GCS.
    """
    spark = None
    temp_dirs = []
    
    try:
        spark = create_spark_session()
        
        # Step 1: Download the cleaned data from GCS to local temporary directory
        print(f"Downloading cleaned data from GCS: {INPUT_DATA_PATH}")
        input_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(input_temp_dir)
        
        download_from_gcs(INPUT_DATA_PATH, input_temp_dir)
        print("Download completed successfully!")
        
        # Step 2: Read the parquet data into Spark DataFrame
        print("Reading parquet data into Spark DataFrame...")
        df = spark.read.parquet(input_temp_dir)
        df = df.cache()
        row_count = df.count()
        print(f"Successfully loaded {row_count} rows.")
        
        # Show schema and sample data for verification
        print("Data schema:")
        df.printSchema()
        print("Sample data:")
        df.show(5)

        # --- Define Feature Engineering Pipeline ---

        # 1. Identify feature columns
        label_col = "churn"
        categorical_cols = [
            "gender", "contract_type", "internet_service", 
            "tech_support", "payment_method"
        ]
        numeric_cols = [
            "age", "tenure_months", "monthly_charges", "total_charges"
        ]

        print(f"Categorical columns: {categorical_cols}")
        print(f"Numeric columns: {numeric_cols}")

        # 2. Define pipeline stages
        stages = []
        
        # Stage 1: Index the string-based target variable 'churn' into a numeric 'label'
        label_indexer = StringIndexer(
            inputCol=label_col, 
            outputCol="label", 
            handleInvalid="keep"
        )
        stages.append(label_indexer)
        
        # Stage 2: Index all categorical feature columns
        indexed_cat_cols = [f"{c}_index" for c in categorical_cols]
        cat_indexers = StringIndexer(
            inputCols=categorical_cols, 
            outputCols=indexed_cat_cols, 
            handleInvalid="keep"
        )
        stages.append(cat_indexers)

        # Stage 3: One-hot encode the indexed categorical columns
        ohe_cat_cols = [f"{c}_vec" for c in categorical_cols]
        cat_encoder = OneHotEncoder(
            inputCols=indexed_cat_cols, 
            outputCols=ohe_cat_cols
        )
        stages.append(cat_encoder)

        # Stage 4: Assemble all numeric features into a single vector
        numeric_assembler = VectorAssembler(
            inputCols=numeric_cols, 
            outputCol="numeric_features"
        )
        stages.append(numeric_assembler)

        # Stage 5: Scale the numeric features vector
        scaler = StandardScaler(
            inputCol="numeric_features", 
            outputCol="scaled_numeric_features",
            withStd=True,
            withMean=True
        )
        stages.append(scaler)

        # Stage 6: Assemble the final feature vector from all processed columns
        final_assembler_inputs = ohe_cat_cols + ["scaled_numeric_features"]
        final_assembler = VectorAssembler(
            inputCols=final_assembler_inputs, 
            outputCol="features"
        )
        stages.append(final_assembler)

        # 3. Create and fit the pipeline
        print("Fitting the feature engineering pipeline...")
        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(df)
        print("Pipeline fitted successfully.")

        # 4. Transform the data
        print("Transforming data with the fitted pipeline...")
        ml_ready_df = pipeline_model.transform(df)
        
        # Select only the final columns needed for training
        final_df = ml_ready_df.select("features", "label")

        # Show the transformed data
        print("Transformed data sample:")
        final_df.show(5, truncate=False)

        # 5. Split data into training and test sets
        train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=1234)
        train_count = train_data.count()
        test_count = test_data.count()
        print(f"Data split into training ({train_count} rows) and test ({test_count} rows).")
        print(f"Train proportion: {train_count/row_count:.2%}")
        print(f"Test proportion: {test_count/row_count:.2%}")

        # 6. Save data to temporary local directories
        train_temp_dir = tempfile.mkdtemp()
        test_temp_dir = tempfile.mkdtemp()
        temp_dirs.extend([train_temp_dir, test_temp_dir])
        
        print(f"Saving training data to temporary directory: {train_temp_dir}")
        train_data.write.mode("overwrite").parquet(train_temp_dir)

        print(f"Saving test data to temporary directory: {test_temp_dir}")
        test_data.write.mode("overwrite").parquet(test_temp_dir)

        # 7. Upload to GCS
        print(f"Uploading training data to GCS: {TRAIN_DATA_PATH}")
        upload_to_gcs(train_temp_dir, TRAIN_DATA_PATH)
        
        print(f"Uploading test data to GCS: {TEST_DATA_PATH}")
        upload_to_gcs(test_temp_dir, TEST_DATA_PATH)

        print("✅ ML preparation completed successfully!")
        
        # Push success status
        kwargs['ti'].xcom_push(key='ml_preparation_success', value=True)
        kwargs['ti'].xcom_push(key='train_count', value=train_count)
        kwargs['ti'].xcom_push(key='test_count', value=test_count)

    except Exception as e:
        print(f"Error during ML feature preparation: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise
    finally:
        # Clean up resources
        if spark is not None:
            try:
                if 'df' in locals():
                    df.unpersist()
                spark.stop()
                print("Spark session stopped.")
            except Exception as e:
                print(f"Error during Spark cleanup: {e}")
        
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                print(f"Error cleaning up temp directory {temp_dir}: {e}")

def ml_cleanup(**kwargs):
    """Additional cleanup if needed"""
    print("ML preparation cleanup completed!")
    return True

# --- Define Airflow Tasks ---

start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

feature_engineering_task = PythonOperator(
    task_id='prepare_ml_features_and_save',
    python_callable=prepare_features_and_save,
    provide_context=True,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=ml_cleanup,
    provide_context=True,
    dag=dag,
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# --- Define Task Dependencies ---
start_task >> feature_engineering_task >> cleanup_task >> end_task
