from kafka import KafkaProducer
from google.cloud import storage
import pandas as pd
import time
import os
import json
import sys
import socket


def get_bootstrap_server():
    try:
        socket.gethostbyname("kafka")
        return "kafka:9092"
    except socket.gaierror:
        return "localhost:29092"


def run_producer():
    bootstrap_server = get_bootstrap_server()
    print(f"🔌 Using Kafka bootstrap server: {bootstrap_server}")

    gcs_cred_path = "/opt/airflow/dags/keys/gcp.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcs_cred_path

    try:
        client = storage.Client()
        bucket_name = "bigdata-ai-datalake"
        blob_name = "raw/E-commerce_Website_Logs.csv"

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_file = "/tmp/clickstream.csv"

        print(f"📥 Downloading {blob_name} from GCS bucket {bucket_name}...")
        blob.download_to_filename(local_file)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Failed to download file from GCS: {e}")
        sys.exit(1)

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_server,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print(f"✅ Connected to Kafka broker at {bootstrap_server}")
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        sys.exit(1)

    try:
        df = pd.read_csv(local_file, low_memory=False)
        print(f"✅ Loaded {len(df)} rows from {local_file}")
    except Exception as e:
        print(f"❌ Failed to read CSV file: {e}")
        sys.exit(1)

    for _, row in df.iterrows():
        message = row.to_dict()
        try:
            producer.send("clickstream_topic", value=message)
            print("📤 Sent:", message)
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
        time.sleep(0.5)

    producer.flush()
    producer.close()
    print("🎉 All messages sent and producer closed.")


if __name__ == "__main__":
    run_producer()
