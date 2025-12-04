from datetime import datetime, timedelta
import tempfile
import os
import shutil
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'clickstream_analytics_pipeline',
    default_args=default_args,
    description='Clickstream Analytics Pipeline with Spark and GCS',
    schedule_interval=None,
    start_date=days_ago(1),
    tags=['clickstream', 'analytics', 'spark'],
    catchup=False,
)

# Configuration
GCS_BUCKET = "bigdata-ai-datalake"
INPUT_GCS_PATH = "curated/clickstream_20251011_222648/clickstream_data.csv"
OUTPUT_GCS_PATH = "enriched/clickstream_enriched.parquet"
CREDENTIALS_PATH = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"

def run_clickstream_analytics(**kwargs):
    """Main function to run clickstream analytics using existing Spark"""
    
    # Import inside function
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, to_timestamp, max as spark_max, min as spark_min,
        count, when, sum as spark_sum, lit, concat, date_format, first
    )
    from google.cloud import storage
    
    print("🚀 Starting Clickstream Analytics Job...")
    
    # -------------------------------------------------------------------------------
    # Set up Google Cloud Storage Credentials
    # -------------------------------------------------------------------------------
    print("🔐 Setting up GCS credentials...")
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH
    
    # Verify credentials file exists
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError(f"Credentials file not found: {CREDENTIALS_PATH}")
    print(f"✅ Using credentials from: {CREDENTIALS_PATH}")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    input_temp_path = os.path.join(temp_dir, "input_data.csv")
    output_temp_dir = os.path.join(temp_dir, "output_parquet")
    
    try:
        # -------------------------------------------------------------------------------
        # Initialize Spark Session using existing Spark
        # -------------------------------------------------------------------------------
        print("🔧 Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName("Clickstream_Analytics_Pipeline") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("INFO")
        
        # -------------------------------------------------------------------------------
        # Set up Google Cloud Storage Client
        # -------------------------------------------------------------------------------
        print("📡 Initializing GCS client...")
        storage_client = storage.Client()
        print("✅ GCS client initialized successfully")
        
        def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
            """Download a file from GCS to local filesystem"""
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            print(f"✅ Downloaded {source_blob_name} to {destination_file_name}")

        def upload_directory_to_gcs(bucket_name, source_dir, destination_dir):
            """Upload a directory to GCS"""
            bucket = storage_client.bucket(bucket_name)
            
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, source_dir)
                    gcs_path = os.path.join(destination_dir, relative_path).replace("\\", "/")
                    
                    blob = bucket.blob(gcs_path)
                    blob.upload_from_filename(local_path)
                    print(f"✅ Uploaded {local_path} to {gcs_path}")

        # -------------------------------------------------------------------------------
        # Download Data from GCS
        # -------------------------------------------------------------------------------
        print("📥 Downloading Clickstream Data from GCS...")
        download_from_gcs(GCS_BUCKET, INPUT_GCS_PATH, input_temp_path)
        
        # -------------------------------------------------------------------------------
        # Load Data with Error Handling
        # -------------------------------------------------------------------------------
        print("📊 Loading Data from Local File...")
        try:
            clickstream_df = (
                spark.read.option("header", "true")
                          .option("inferSchema", "true")
                          .csv(input_temp_path)
            )
            
            record_count = clickstream_df.count()
            print(f"✅ Loaded {record_count} records")
            
            # Show schema for verification
            print("🔍 Data Schema:")
            clickstream_df.printSchema()
            
            # Show sample data
            print("📊 Sample Data:")
            clickstream_df.show(5, truncate=False)
            
        except Exception as e:
            print(f"❌ Error loading data from local file: {str(e)}")
            raise

        # -------------------------------------------------------------------------------
        # Data Validation and Feature Extraction
        # -------------------------------------------------------------------------------
        print("🔍 Performing Data Validation and Feature Extraction...")

        # Map existing columns to expected columns based on your actual data
        print("📋 Available columns:", clickstream_df.columns)
        
        try:
            # Create session_id from ip_address + accessed_date (hour level)
            clickstream_df = (
                clickstream_df
                .withColumn("session_id", 
                           concat(col("ip_address"), lit("_"), 
                                 date_format(col("accessed_date"), "yyyyMMddHH")))
                .withColumn("user_id", col("ip_address"))  # Using IP as user_id for demo
                .withColumn("event_type", 
                           when(col("sales_amount") > 0, "purchase")
                           .when(col("bytes") > 0, "view")
                           .otherwise("other"))
                .withColumn("event_time", to_timestamp(col("accessed_date")))
                .dropna(subset=["user_id", "session_id"])
            )
            
            print(f"✅ Data after transformation: {clickstream_df.count()} records")
            print("📊 Transformed Data Sample:")
            clickstream_df.select("user_id", "session_id", "event_type", "event_time", "sales_amount", "bytes").show(5)
            
        except Exception as e:
            print(f"❌ Error during feature extraction: {str(e)}")
            raise

        # -------------------------------------------------------------------------------
        # Session Analytics
        # -------------------------------------------------------------------------------
        print("🧮 Calculating Session Analytics...")

        try:
            session_duration_df = (
                clickstream_df.groupBy("user_id", "session_id")
                .agg(
                    (spark_max("event_time").cast("long") - spark_min("event_time").cast("long")).alias("session_duration_seconds"),
                    count(when(col("event_type") == "view", True)).alias("views"),
                    count(when(col("event_type") == "purchase", True)).alias("purchases"),
                    spark_sum("sales_amount").alias("total_sales_amount"),
                    count(when(col("returned") == "Yes", True)).alias("returns"),
                    spark_min("event_time").alias("session_start"),
                    spark_max("event_time").alias("session_end"),
                    count("*").alias("total_events"),
                    # Additional metrics from your data
                    first("country").alias("country"),
                    first("browser").alias("browser"),
                    first("payment_method").alias("payment_method")
                )
                .filter(col("session_duration_seconds") >= 0)
            )

            # Calculate conversion rate
            session_duration_df = session_duration_df.withColumn(
                "conversion_rate",
                when(col("views") > 0, col("purchases") / col("views")).otherwise(0.0)
            )
            
            # Calculate return rate
            session_duration_df = session_duration_df.withColumn(
                "return_rate",
                when(col("purchases") > 0, col("returns") / col("purchases")).otherwise(0.0)
            )
            
            print(f"✅ Processed {session_duration_df.count()} user sessions")
            
        except Exception as e:
            print(f"❌ Error during session analytics: {str(e)}")
            raise

        # -------------------------------------------------------------------------------
        # Behavior Segmentation
        # -------------------------------------------------------------------------------
        print("🧩 Applying Behavior Segmentation Rules...")

        try:
            session_duration_df = session_duration_df.withColumn(
                "segment",
                when((col("purchases") > 0) & (col("total_sales_amount") > 100), "High_Value_Buyer")
                .when((col("purchases") > 0), "Buyer")
                .when((col("views") >= 5), "Active_Browser") 
                .when((col("views") > 0), "Browser")
                .otherwise("Other")
            )
            
            print("📊 Segment Distribution:")
            session_duration_df.groupBy("segment").count().orderBy("count", ascending=False).show()
            
        except Exception as e:
            print(f"❌ Error during behavior segmentation: {str(e)}")
            raise

        # -------------------------------------------------------------------------------
        # Save Final Output Locally
        # -------------------------------------------------------------------------------
        print("💾 Saving Enriched Data Locally...")

        try:
            (
                session_duration_df
                .repartition(1)
                .write
                .mode("overwrite")
                .option("compression", "snappy")
                .parquet(output_temp_dir)
            )
            
            print("✅ Clickstream Enriched Data Successfully Saved Locally.")
            
        except Exception as e:
            print(f"❌ Error saving data locally: {str(e)}")
            raise

        # -------------------------------------------------------------------------------
        # Upload Results Back to GCS
        # -------------------------------------------------------------------------------
        print(f"☁️ Uploading Enriched Data to GCS: {OUTPUT_GCS_PATH}")

        try:
            upload_directory_to_gcs(GCS_BUCKET, output_temp_dir, OUTPUT_GCS_PATH)
            print("✅ Successfully uploaded enriched data to GCS!")
            
        except Exception as e:
            print(f"❌ Error uploading to GCS: {str(e)}")
            raise

        # -------------------------------------------------------------------------------
        # Verification and Summary
        # -------------------------------------------------------------------------------
        print("🔎 Final Enriched Data Sample:")
        session_duration_df.show(10, truncate=False)

        print("📈 Summary Statistics:")
        session_duration_df.select(
            "session_duration_seconds", 
            "views", 
            "purchases", 
            "total_sales_amount",
            "conversion_rate",
            "return_rate"
        ).describe().show()

        print("🎉 Clickstream Analytics Pipeline Completed Successfully!")
        print(f"📊 Total Sessions Processed: {session_duration_df.count()}")

        # Push result count to XCom
        kwargs['ti'].xcom_push(key='processed_sessions', value=session_duration_df.count())

    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        raise
        
    finally:
        # -------------------------------------------------------------------------------
        # Cleanup Temporary Files
        # -------------------------------------------------------------------------------
        print("🧹 Cleaning up temporary files...")
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"✅ Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Could not clean up temporary files: {e}")

        # Stop Spark session
        if 'spark' in locals():
            spark.stop()
            print("✅ Spark session stopped")

# Define the task
clickstream_analytics_task = PythonOperator(
    task_id='run_clickstream_analytics',
    python_callable=run_clickstream_analytics,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
clickstream_analytics_task
