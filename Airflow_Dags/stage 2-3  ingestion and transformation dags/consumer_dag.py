from airflow import DAG
from airflow.operators.python_operator import PythonOperator # pyright: ignore[reportMissingImports]
from airflow.utils.dates import days_ago
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, trim, to_date
from google.cloud import storage
from pyunpack import Archive
import tempfile
import os
import shutil
import glob

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

dag = DAG(
    'consumer_complaints_processing',
    default_args=default_args,
    description='Process consumer_complaints RAR to curated Parquet',
    schedule_interval=None,
)

CRED_PATH = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"
BUCKET_NAME = "bigdata-ai-datalake"
RAR_BLOB_PATH = "raw/consumer_complaints.rar"

def init_spark():
    return SparkSession.builder \
        .appName("ConsumerComplaintsProcessing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def download_and_extract(**kwargs):
    client = storage.Client.from_service_account_json(CRED_PATH)
    bucket = client.bucket(BUCKET_NAME)

    temp_dir = tempfile.mkdtemp()
    rar_path = os.path.join(temp_dir, "consumer_complaints.rar")
    bucket.blob(RAR_BLOB_PATH).download_to_filename(rar_path)

    # فك الضغط باستخدام patool أو WinRAR
    try:
        Archive(rar_path).extractall(temp_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to extract RAR: {e}")

    # البحث عن CSV
    csv_files = glob.glob(os.path.join(temp_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError("No CSV file found in RAR")
    csv_path = csv_files[0]

    kwargs['ti'].xcom_push(key='temp_dir', value=temp_dir)
    kwargs['ti'].xcom_push(key='csv_path', value=csv_path)

def process_data(**kwargs):
    spark = init_spark()
    csv_path = kwargs['ti'].xcom_pull(key='csv_path')
    
    df = spark.read.option("header", "true").csv(csv_path)

    # تحويل الأعمدة التي تمثل تواريخ
    for date_col in ["Date received", "Date sent to company"]:
        if date_col in df.columns:
            df = df.withColumn(date_col, to_date(col(date_col), "MM/dd/yyyy"))

    # تنظيف النصوص
    text_cols = ["Product", "Sub-product", "Issue", "Sub-issue",
                 "Consumer complaint narrative", "Company public response",
                 "Company", "State", "Tags", "Submitted via",
                 "Company response to consumer"]
    for c in text_cols:
        if c in df.columns:
            df = df.withColumn(c, trim(lower(col(c))))

    df = df.dropna(subset=["Date received", "Complaint ID"]).dropDuplicates()
    
    # حفظ Parquet مؤقتًا
    output_dir = tempfile.mkdtemp()
    local_parquet_path = os.path.join(output_dir, "consumer_complaints_parquet")
    df.write.mode("overwrite").option("compression", "snappy").parquet(local_parquet_path)

    kwargs['ti'].xcom_push(key='local_parquet_path', value=local_parquet_path)
    kwargs['ti'].xcom_push(key='final_count', value=df.count())

    spark.stop()

def save_and_upload(**kwargs):
    local_parquet_path = kwargs['ti'].xcom_pull(key='local_parquet_path')
    final_count = kwargs['ti'].xcom_pull(key='final_count')

    spark = init_spark()
    df = spark.read.parquet(local_parquet_path)

    client = storage.Client.from_service_account_json(CRED_PATH)
    bucket = client.bucket(BUCKET_NAME)

    uploaded_count = 0
    for root, dirs, files in os.walk(local_parquet_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_parquet_path)
            blob_path = f"curated/consumer_complaints/{relative_path}"
            bucket.blob(blob_path).upload_from_filename(local_file_path)
            uploaded_count += 1

    print(f"Uploaded {uploaded_count} files to GCS.")
    print(f"Total processed rows: {final_count:,}")
    spark.stop()

def cleanup(**kwargs):
    temp_dir = kwargs['ti'].xcom_pull(key='temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("Cleanup completed.")

# Define tasks
download_task = PythonOperator(
    task_id='download_and_extract',
    python_callable=download_and_extract,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
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

# Set dependencies
download_task >> process_task >> upload_task >> cleanup_task
