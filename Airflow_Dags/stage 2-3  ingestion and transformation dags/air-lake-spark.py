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

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import col, when, isnull

# GCS imports
from google.cloud import storage

# Default arguments
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
    'customer_churn_processing',
    default_args=default_args,
    description='Process customer churn data from GCS using PySpark',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['pyspark', 'gcs', 'churn']
)

# Configuration
GCS_BUCKET = "bigdata-ai-datalake"
GCS_FILE_PATH = "raw/Cust-churn.csv"
CREDENTIALS_PATH = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"
OUTPUT_FILENAME = "customers_churn.parquet"

def create_spark_session():
    """Create and return Spark session"""
    spark = (
        SparkSession.builder
        .appName("BigDataProject")
        .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.5")
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", CREDENTIALS_PATH)
        .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
        .getOrCreate()
    )
    return spark

def load_spark_df_from_gcs(**kwargs):
    """Download file from GCS and load into Spark DataFrame"""
    ti = kwargs['ti']
    
    try:
        print("Downloading file from GCS...")

        # Initialize GCS client
        client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_FILE_PATH)

        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "customer_churn_data.csv")

        print(f"Downloading to: {temp_path}")
        blob.download_to_filename(temp_path)
        print("File downloaded successfully!")

        # Verify file exists
        file_size = os.path.getsize(temp_path)
        print(f"File size: {file_size} bytes")

        # Create Spark session
        spark = create_spark_session()

        # Schema definition
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("age", FloatType(), True),
            StructField("gender", StringType(), True),
            StructField("tenure_months", IntegerType(), True),
            StructField("monthly_charges", FloatType(), True),
            StructField("contract_type", StringType(), True),
            StructField("internet_service", StringType(), True),
            StructField("tech_support", StringType(), True),
            StructField("payment_method", StringType(), True),
            StructField("total_charges", FloatType(), True),
            StructField("churn", StringType(), True),
        ])

        # Read CSV into Spark DataFrame
        print("Reading CSV file into Spark DataFrame...")
        df = spark.read.csv(temp_path, header=True, schema=schema)

        # Persist and count
        df = df.cache()
        row_count = df.count()
        print(f"DataFrame loaded successfully! Number of rows: {row_count}")

        # Push temp_dir to XCom for cleanup
        ti.xcom_push(key='temp_dir', value=temp_dir)
        
        # Store DataFrame reference (in production, you'd save to temporary storage)
        # For this example, we'll push the row count as a simple metric
        ti.xcom_push(key='row_count', value=row_count)
        ti.xcom_push(key='initial_data_loaded', value=True)

        print("Data loading completed successfully!")
        return True

    except Exception as e:
        print(f"Error loading data from GCS: {e}")
        raise

def clean_and_process_data(**kwargs):
    """Clean and process the DataFrame"""
    ti = kwargs['ti']
    
    # Check if previous step was successful
    data_loaded = ti.xcom_pull(task_ids='load_data', key='initial_data_loaded')
    if not data_loaded:
        raise Exception("Data not loaded successfully from previous step")
    
    try:
        # In a real scenario, you'd retrieve the DataFrame from shared storage
        # For this example, we'll recreate the Spark session and reload the data
        spark = create_spark_session()
        
        # Reload data (in production, you'd read from intermediate storage)
        temp_dir = ti.xcom_pull(task_ids='load_data', key='temp_dir')
        temp_path = os.path.join(temp_dir, "customer_churn_data.csv")
        
        schema = StructType([
            StructField("customer_id", StringType(), True),
            StructField("age", FloatType(), True),
            StructField("gender", StringType(), True),
            StructField("tenure_months", IntegerType(), True),
            StructField("monthly_charges", FloatType(), True),
            StructField("contract_type", StringType(), True),
            StructField("internet_service", StringType(), True),
            StructField("tech_support", StringType(), True),
            StructField("payment_method", StringType(), True),
            StructField("total_charges", FloatType(), True),
            StructField("churn", StringType(), True),
        ])
        
        df = spark.read.csv(temp_path, header=True, schema=schema)
        df = df.cache()

        print("Starting data cleaning...")

        # Drop duplicates
        initial_count = df.count()
        df_cleaned = df.dropDuplicates()
        after_dedup_count = df_cleaned.count()
        print(f"Duplicates removed: {initial_count - after_dedup_count} rows")

        # Handle missing values
        print("Handling missing values...")
        numeric_columns = ["age", "tenure_months", "monthly_charges", "total_charges"]
        for col_name in numeric_columns:
            df_cleaned = df_cleaned.fillna(0, subset=[col_name])

        string_columns = ["gender", "contract_type", "internet_service",
                          "tech_support", "payment_method", "churn"]
        for col_name in string_columns:
            df_cleaned = df_cleaned.fillna("Unknown", subset=[col_name])

        # Fix invalid numeric values
        print("Processing numeric columns...")
        df_processed = df_cleaned
        for col_name in ["age", "tenure_months", "monthly_charges", "total_charges"]:
            df_processed = df_processed.withColumn(
                col_name,
                when(isnull(col(col_name)) | (col(col_name) < 0), 0.0)
                .otherwise(col(col_name))
            )

        # Validate
        final_count = df_processed.count()
        print(f"Final row count: {final_count}")

        # Push processed data info to XCom
        ti.xcom_push(key='processed_row_count', value=final_count)
        ti.xcom_push(key='df_processed', value=True)
        
        # Store the processed DataFrame path (in production, save to temp location)
        processed_temp_dir = tempfile.mkdtemp()
        processed_temp_path = os.path.join(processed_temp_dir, "processed_data.parquet")
        df_processed.write.mode("overwrite").parquet(processed_temp_path)
        
        ti.xcom_push(key='processed_temp_dir', value=processed_temp_dir)
        ti.xcom_push(key='processed_temp_path', value=processed_temp_path)

        print("Data cleaning completed successfully!")
        return True

    except Exception as e:
        print(f"Error during data cleaning: {e}")
        raise

def save_data_to_gcs(**kwargs):
    """Save processed data to GCS"""
    ti = kwargs['ti']
    
    # Check if previous step was successful
    processed = ti.xcom_pull(task_ids='process_data', key='df_processed')
    if not processed:
        raise Exception("Data not processed successfully from previous step")
    
    try:
        processed_temp_path = ti.xcom_pull(task_ids='process_data', key='processed_temp_path')
        
        if not processed_temp_path or not os.path.exists(processed_temp_path):
            raise Exception("Processed data not found")
        
        # Upload to GCS
        client = storage.Client.from_service_account_json(CREDENTIALS_PATH)
        bucket = client.bucket(GCS_BUCKET)
        
        # Upload the entire directory
        for root, dirs, files in os.walk(processed_temp_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                gcs_file_path = os.path.join(
                    f"curated/{OUTPUT_FILENAME}",
                    os.path.relpath(local_file_path, processed_temp_path)
                )
                
                blob = bucket.blob(gcs_file_path)
                blob.upload_from_filename(local_file_path)
                print(f"Uploaded {local_file_path} to {gcs_file_path}")
        
        print("Data uploaded to GCS successfully!")
        ti.xcom_push(key='gcs_upload_success', value=True)
        return True
        
    except Exception as e:
        print(f"Error saving data to GCS: {e}")
        raise

def cleanup_resources(**kwargs):
    """Clean up temporary files and Spark session"""
    ti = kwargs['ti']
    
    try:
        # Clean up temporary directories
        temp_dir = ti.xcom_pull(task_ids='load_data', key='temp_dir')
        processed_temp_dir = ti.xcom_pull(task_ids='process_data', key='processed_temp_dir')
        
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Cleaned up temp directory: {temp_dir}")
            
        if processed_temp_dir and os.path.exists(processed_temp_dir):
            shutil.rmtree(processed_temp_dir, ignore_errors=True)
            print(f"Cleaned up processed temp directory: {processed_temp_dir}")
        
        # Stop Spark session
        spark = create_spark_session()
        spark.stop()
        print("Spark session stopped.")
        
        print("Cleanup completed successfully!")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        # Don't raise exception in cleanup to avoid masking previous errors

# Define tasks
start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_spark_df_from_gcs,
    provide_context=True,
    dag=dag,
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=clean_and_process_data,
    provide_context=True,
    dag=dag,
)

save_data_task = PythonOperator(
    task_id='save_data',
    python_callable=save_data_to_gcs,
    provide_context=True,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_resources,
    provide_context=True,
    dag=dag,
    trigger_rule='all_done',  # Run cleanup regardless of previous task success/failure
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define task dependencies
start_task >> load_data_task >> process_data_task >> save_data_task >> cleanup_task >> end_task

# Alternative: If you want cleanup to run even if intermediate tasks fail
# start_task >> load_data_task >> process_data_task >> save_data_task >> end_task
# [load_data_task, process_data_task, save_data_task] >> cleanup_task
