from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# ==============================
# Config
# ==============================
GITHUB_URL = "https://raw.githubusercontent.com/ABDULLAH-ibrahimm/bigdata-ai-platform/9d55674b154a80ebb5759367ca95750a0a71ba4f/data_sources"
BUCKET = "bigdata-ai-datalake"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ==============================
# DAG Definition
# ==============================
with DAG(
    dag_id="data_ingestion_github_to_gcs",
    default_args=default_args,
    description="Ingest raw data from GitHub into GCS Data Lake",
    schedule="@daily",   # ✅ الجديد بدل schedule_interval
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=["ingestion", "github", "gcs", "datalake"],
) as dag:

    # --------- Transactions (CSV) ---------
    ingest_transactions = BashOperator(
        task_id="ingest_transactions",
        bash_command=f"curl -sSL {GITHUB_URL}/Cust-churn.csv | gsutil cp - gs://{BUCKET}/raw/Cust-churn.csv",
    )

    # --------- Clickstream Logs (CSV) ---------
    ingest_clickstream = BashOperator(
        task_id="ingest_clickstream",
        bash_command=f"curl -sSL {GITHUB_URL}/E-commerce%20Website%20Logs.csv | gsutil cp - gs://{BUCKET}/raw/E-commerce_Website_Logs.csv",
    )

    # --------- Complaints (RAR) ---------
    ingest_complaints = BashOperator(
        task_id="ingest_complaints",
        bash_command=f"curl -sSL {GITHUB_URL}/consumer_complaints.rar | gsutil cp - gs://{BUCKET}/raw/consumer_complaints.rar",
    )

    # --------- Reviews (RAR) ---------
    ingest_reviews = BashOperator(
        task_id="ingest_reviews",
        bash_command=f"curl -sSL {GITHUB_URL}/reviews.rar | gsutil cp - gs://{BUCKET}/raw/reviews.rar",
    )

    # ترتيب التنفيذ (ممكن يتنفذوا كلهم parallel)
    [ingest_transactions, ingest_clickstream, ingest_complaints, ingest_reviews]
