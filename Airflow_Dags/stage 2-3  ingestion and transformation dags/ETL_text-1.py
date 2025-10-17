from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, concat_ws
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from google.cloud import storage
from pyunpack import Archive
import tempfile
import os
import shutil

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'reviews_processing',
    default_args=default_args,
    description='Process reviews from RAR to curated Parquet',
    schedule_interval=None,
)

def init_spark():
    spark = SparkSession.builder \
        .appName("ReviewsProcessing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    return spark

def download_and_extract(**kwargs):
    print(" Downloading RAR file...")
    cred_path = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"
    client = storage.Client.from_service_account_json(cred_path)
    bucket = client.bucket("bigdata-ai-datalake")
    
    temp_dir = tempfile.mkdtemp()
    rar_path = os.path.join(temp_dir, "reviews.rar")
    blob = bucket.blob("raw/reviews.rar")
    blob.download_to_filename(rar_path)
    print(" Downloaded RAR file")
    
    Archive(rar_path).extractall(temp_dir)
    print(" Extracted RAR file")
    
    # Find JSONL file
    extracted_files = os.listdir(temp_dir)
    jsonl_files = [f for f in extracted_files if f.endswith('.jsonl') or '.json' in f]
    if not jsonl_files:
        raise FileNotFoundError("No JSONL file found")
    
    jsonl_filename = jsonl_files[0]
    jsonl_path = os.path.join(temp_dir, jsonl_filename)
    
    kwargs['ti'].xcom_push(key='temp_dir', value=temp_dir)
    kwargs['ti'].xcom_push(key='jsonl_path', value=jsonl_path)
    kwargs['ti'].xcom_push(key='jsonl_filename', value=jsonl_filename)

def load_and_process_data(**kwargs):
    print(" Reading JSONL with Spark...")
    spark = init_spark()
    
    temp_dir = kwargs['ti'].xcom_pull(key='temp_dir')
    jsonl_path = kwargs['ti'].xcom_pull(key='jsonl_path')
    
    df = spark.read \
        .option("mode", "PERMISSIVE") \
        .option("columnNameOfCorruptRecord", "_corrupt_record") \
        .json(jsonl_path)
    
    df.cache()
    record_count = df.count()
    print(f" Loaded {record_count:,} records directly with Spark")
    
    # Select columns
    print(" Processing data...")
    columns_to_keep = ["rating", "title", "text", "helpful_vote", "verified_purchase"]
    available_columns = [col for col in columns_to_keep if col in df.columns]
    df = df.select(available_columns)
    
    # Basic preprocessing
    initial_count = df.count()
    df = df.dropna().dropDuplicates()
    final_count = df.count()
    print(f" Data cleaning: {initial_count:,} -> {final_count:,} records")
    
    # Text preprocessing
    if "text" in df.columns:
        df = df.withColumn("text", trim(lower(col("text"))))
        tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\s+")
        df = tokenizer.transform(df)
        remover = StopWordsRemover(inputCol="tokens", outputCol="clean_tokens")
        df = remover.transform(df)
        df = df.withColumn("cleaned_text", concat_ws(" ", col("clean_tokens")))
        df = df.drop("tokens", "clean_tokens")
    
    # Save processed data to temporary location
    output_dir = tempfile.mkdtemp()
    processed_path = os.path.join(output_dir, "processed_data.parquet")
    df.write.mode("overwrite").parquet(processed_path)
    
    # Push paths to XCom instead of DataFrame
    kwargs['ti'].xcom_push(key='processed_path', value=processed_path)
    kwargs['ti'].xcom_push(key='final_count', value=final_count)
    
    df.unpersist()
    spark.stop()
    
    print(" Data processing completed")

def save_and_upload(**kwargs):
    print("Saving as Parquet...")
    
    processed_path = kwargs['ti'].xcom_pull(key='processed_path')
    final_count = kwargs['ti'].xcom_pull(key='final_count')
    
    # Reinitialize Spark to read the processed data
    spark = init_spark()
    df = spark.read.parquet(processed_path)
    
    # Continue with upload logic...
    output_dir = tempfile.mkdtemp()
    local_parquet_path = os.path.join(output_dir, "reviews.parquet")
    
    df.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(local_parquet_path)
    
    print("Uploading to GCS...")
    cred_path = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"
    client = storage.Client.from_service_account_json(cred_path)
    bucket = client.bucket("bigdata-ai-datalake")
    
    uploaded_count = 0
    for root, dirs, files in os.walk(local_parquet_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_parquet_path)
            blob_path = f"curated/reviews.parquet/{relative_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)
            uploaded_count += 1
    
    print(f"Total uploaded files: {uploaded_count}")
    print(f"Successfully processed {final_count:,} reviews")
    
    spark.stop()
def cleanup(**kwargs):
    print(" Cleaning up temporary files...")
    temp_dir = kwargs['ti'].xcom_pull(key='temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    print(" Cleanup completed")

# Define tasks
download_task = PythonOperator(
    task_id='download_and_extract',
    python_callable=download_and_extract,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=load_and_process_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='save_and_upload',
    python_callable=save_and_upload,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    dag=dag,
)

# Set task dependencies
download_task >> process_task >> upload_task >> cleanup_task
