from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType
import os
import time
from datetime import datetime
from google.cloud import storage

def create_spark_session():
    return SparkSession.builder \
        .appName("KafkaSparkProcessing") \
        .master("spark://spark-master:7077") \
        .config("spark.jars", 
                "/opt/spark/jars/spark-sql-kafka-0-10_2.12-3.5.1.jar,"
                "/opt/spark/jars/kafka-clients-3.4.0.jar") \
        .config("spark.sql.adaptive.enabled", "false") \
        .getOrCreate()

def save_dataframe_as_csv(df, output_dir):
    """Save DataFrame as CSV with improved file handling"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with single partition
        df.coalesce(1) \
          .write \
          .mode("overwrite") \
          .option("header", "true") \
          .option("delimiter", ",") \
          .csv(output_dir)
        
        print("🔄 Waiting for Spark to write files...")
        time.sleep(10)  # Increased wait time
        
        # Find the actual CSV file
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.startswith('part-') and file.endswith('.csv'):
                    csv_file_path = os.path.join(root, file)
                    final_path = os.path.join(output_dir, "clickstream_data.csv")
                    
                    # Rename the file
                    os.rename(csv_file_path, final_path)
                    print(f"✅ CSV file created: {final_path}")
                    print(f"📊 File size: {os.path.getsize(final_path)} bytes")
                    return final_path
        
        print("❌ No data file found in output directory")
        return None
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return None

def save_dataframe_direct_csv(df, output_dir):
    """Alternative method to save DataFrame as CSV"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "clickstream_data.csv")
        
        # Convert to pandas and save (if pandas available)
        try:
            pandas_df = df.toPandas()
            pandas_df.to_csv(final_path, index=False)
            print(f"✅ CSV created using pandas: {final_path}")
            return final_path
        except:
            # Fallback: Use Spark's built-in CSV writer
            df.coalesce(1) \
              .write \
              .mode("overwrite") \
              .option("header", "true") \
              .option("delimiter", ",") \
              .csv(output_dir + "_spark")
            
            time.sleep(5)
            
            # Find and rename the file
            spark_dir = output_dir + "_spark"
            for root, dirs, files in os.walk(spark_dir):
                for file in files:
                    if file.startswith('part-'):
                        source_file = os.path.join(root, file)
                        os.rename(source_file, final_path)
                        print(f"✅ CSV created using Spark: {final_path}")
                        return final_path
            
            return None
            
    except Exception as e:
        print(f"❌ Error in direct CSV save: {e}")
        return None

def upload_to_gcs(local_file_path, timestamp_str):
    """Upload file to Google Cloud Storage"""
    try:
        key_file_path = "/opt/airflow/dags/keys/gcp.json"
        
        if not os.path.exists(local_file_path):
            print(f"❌ Local file not found: {local_file_path}")
            return False

        client = storage.Client.from_service_account_json(key_file_path)
        bucket = client.bucket('bigdata-ai-datalake')
        
        if not bucket.exists():
            print("❌ Bucket does not exist")
            return False

        blob_name = f"curated/clickstream_{timestamp_str}/clickstream_data.csv"
        blob = bucket.blob(blob_name)
        
        print(f"⬆️ Uploading to GCS: {blob_name}")
        blob.upload_from_filename(local_file_path)
        
        if blob.exists():
            print(f"✅ Successfully uploaded to Data Lake")
            print(f"📍 GCS Path: gs://bigdata-ai-datalake/{blob_name}")
            return True
        else:
            print("❌ Upload verification failed")
            return False
            
    except Exception as e:
        print(f"❌ GCS upload failed: {e}")
        return False

def main():
    try:
        # Setup
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/opt/airflow/dags/keys/gcp.json'
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("WARN")
        
        print("🚀 STARTING SPARK KAFKA PROCESSING JOB")
        
        # Read from Kafka
        print("📖 Reading from Kafka topic: clickstream_topic")
        df = spark.read \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:9092") \
            .option("subscribe", "clickstream_topic") \
            .option("startingOffsets", "earliest") \
            .load()

        total_records = df.count()
        print(f"📊 Total Kafka records: {total_records}")

        if total_records == 0:
            print("🟡 No data found in Kafka")
            spark.stop()
            return

        # Parse JSON data
        schema = StructType() \
            .add("accessed_date", StringType()) \
            .add("duration_(secs)", IntegerType()) \
            .add("network_protocol", StringType()) \
            .add("ip", StringType()) \
            .add("bytes", IntegerType()) \
            .add("accessed_Ffom", StringType()) \
            .add("age", IntegerType()) \
            .add("gender", StringType()) \
            .add("country", StringType()) \
            .add("membership", StringType()) \
            .add("language", StringType()) \
            .add("sales", DoubleType()) \
            .add("returned", StringType()) \
            .add("returned_amount", DoubleType()) \
            .add("pay_method", StringType())

        df_parsed = df.selectExpr("CAST(value AS STRING) as json_str") \
            .select(from_json(col("json_str"), schema).alias("data")) \
            .select(
                col("data.accessed_date").alias("accessed_date"),
                col("data.duration_(secs)").alias("duration_secs"),
                col("data.network_protocol").alias("network_protocol"),
                col("data.ip").alias("ip_address"),
                col("data.bytes").alias("bytes"),
                col("data.accessed_Ffom").alias("browser"),
                col("data.age").alias("age"),
                col("data.gender").alias("gender"),
                col("data.country").alias("country"),
                col("data.membership").alias("membership"),
                col("data.language").alias("language"),
                col("data.sales").alias("sales_amount"),
                col("data.returned").alias("returned"),
                col("data.returned_amount").alias("returned_amount"),
                col("data.pay_method").alias("payment_method"),
                current_timestamp().alias("processing_timestamp")
            ).filter(col("ip_address").isNotNull())

        parsed_records = df_parsed.count()
        print(f"📊 Successfully parsed records: {parsed_records}")

        if parsed_records > 0:
            # Show sample
            print("📊 Sample data:")
            df_parsed.show(5, truncate=False)

            # Create output directory
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"/opt/spark/work-dir/output/clickstream_{timestamp_str}"
            
            # Try multiple methods to save CSV
            csv_file = save_dataframe_as_csv(df_parsed, output_dir)
            
            if not csv_file:
                print("🔄 Trying alternative CSV save method...")
                csv_file = save_dataframe_direct_csv(df_parsed, output_dir)
            
            if csv_file and os.path.exists(csv_file):
                print(f"💾 Data saved to: {csv_file}")
                
                # Upload to GCS
                upload_success = upload_to_gcs(csv_file, timestamp_str)
                
                if upload_success:
                    print("🎯 SUCCESS: Data processing and upload completed!")
                else:
                    print("❌ Upload failed")
            else:
                print("❌ All CSV save methods failed")
                
            print(f"\n📈 SUMMARY: Processed {parsed_records} records")
            
        else:
            print("🟡 No valid data to process")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        spark.stop()
        print("🔴 SPARK JOB COMPLETED")

if __name__ == "__main__":
    main()