from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, datediff, when, year, month, avg, count, mean, sum as spark_sum
)
from google.cloud import storage
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
    'consumer_complaints_enrichment',
    default_args=default_args,
    description='Enrich consumer complaints parquet with features and aggregation',
    schedule_interval=None,
)

CRED_PATH = "/home/mansour/datasets/data-lake-473309-cbdf5da98c98.json"
BUCKET_NAME = "bigdata-ai-datalake"
INPUT_PATH = "curated/consumer_complaints"
OUTPUT_PATH = "enriched/consumer_complaints_enriched"

def init_spark():
    return SparkSession.builder \
        .appName("ConsumerComplaintsEnrichment") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

def download_from_gcs(**kwargs):
    client = storage.Client.from_service_account_json(CRED_PATH)
    bucket = client.bucket(BUCKET_NAME)

    temp_dir = tempfile.mkdtemp()
    local_input_dir = os.path.join(temp_dir, "consumer_complaints")
    os.makedirs(local_input_dir, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=INPUT_PATH))
    parquet_files = [b for b in blobs if b.name.endswith(".parquet")]

    if not parquet_files:
        raise FileNotFoundError("No parquet files found in GCS curated/consumer_complaints/")

    for blob in parquet_files:
        dest_path = os.path.join(local_input_dir, os.path.basename(blob.name))
        blob.download_to_filename(dest_path)

    kwargs['ti'].xcom_push(key='local_input_dir', value=local_input_dir)
    kwargs['ti'].xcom_push(key='temp_dir', value=temp_dir)

def process_complaints(**kwargs):
    spark = init_spark()
    local_input_dir = kwargs['ti'].xcom_pull(key='local_input_dir')

    df = spark.read.parquet(local_input_dir)

    if "Date received" not in df.columns or "Date sent to company" not in df.columns:
        raise ValueError("Required date columns are missing in the dataset.")

    df = df.withColumn("complaint_duration", datediff(col("Date sent to company"), col("Date received")))
    df = df.withColumn("is_delayed", when(col("complaint_duration") > 7, 1).otherwise(0))
    df = df.withColumn("year", year(col("Date received")))
    df = df.withColumn("month", month(col("Date received")))

    if "Issue" in df.columns:
        df = df.withColumn(
            "sentiment",
            when(col("Issue").rlike("error|failure|problem"), "negative").otherwise("neutral")
        )


    group_col = "State" if "State" in df.columns else None
    if not group_col:
        raise ValueError("No State column found for aggregation.")

    agg_df = df.groupBy(group_col).agg(
        count("*").alias("total_complaints"),
        avg(col("complaint_duration")).alias("avg_response_time"),
        (spark_sum(col("is_delayed")) / count("*")).alias("delayed_response_rate")
    )

    if "sentiment" in df.columns:
        sentiment_ratio = (
            df.groupBy(group_col, "sentiment")
              .count()
              .groupBy(group_col)
              .pivot("sentiment", ["negative", "neutral"])
              .sum("count")
              .fillna(0)
        )
        agg_df = agg_df.join(sentiment_ratio, on=group_col, how="left")

    output_dir = tempfile.mkdtemp()
    local_output_path = os.path.join(output_dir, "complaints_enriched.parquet")
    agg_df.write.mode("overwrite").parquet(local_output_path)

    kwargs['ti'].xcom_push(key='local_output_path', value=local_output_path)
    spark.stop()

def upload_to_gcs(**kwargs):
    local_output_path = kwargs['ti'].xcom_pull(key='local_output_path')
    client = storage.Client.from_service_account_json(CRED_PATH)
    bucket = client.bucket(BUCKET_NAME)

    for root, dirs, files in os.walk(local_output_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_output_path)
            blob_path = f"{OUTPUT_PATH}/{relative_path}"
            bucket.blob(blob_path).upload_from_filename(local_file_path)

    print(f"Uploaded enriched data to GCS folder: {OUTPUT_PATH}")

def cleanup(**kwargs):
    temp_dir = kwargs['ti'].xcom_pull(key='temp_dir')
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("Cleanup completed.")

download_task = PythonOperator(
    task_id='download_from_gcs',
    python_callable=download_from_gcs,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_complaints',
    python_callable=process_complaints,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_to_gcs',
    python_callable=upload_to_gcs,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup,
    dag=dag,
)

download_task >> process_task >> upload_task >> cleanup_task
