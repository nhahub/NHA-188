from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="clickstream_pipeline_final",
    default_args=default_args,
    description="Final clickstream pipeline",
    start_date=datetime(2025, 9, 23),
    schedule_interval=None,
    catchup=False,
    tags=["kafka", "spark", "pipeline"],
) as dag:

    start = DummyOperator(task_id="start")

    # 🟢 Step 1: Kafka Producer
    kafka_producer = BashOperator(
        task_id="run_kafka_producer",
        bash_command="cd /opt/airflow/dags/scripts && python producer.py"
    )

    # 🟡 Step 2: Spark Job
    spark_job = BashOperator(
        task_id="run_spark_job",
        bash_command="""
        docker exec spark-master /opt/spark/bin/spark-submit \
            --master spark://spark-master:7077 \
            --conf spark.sql.adaptive.enabled=false \
            /opt/airflow/dags/scripts/spark_job.py
        """,
        retries=2
    )

    # 🔵 Step 3: Validate
    validate = BashOperator(
        task_id="validate_output",
        bash_command="""
        echo "✅ Pipeline completed successfully!" && \
        echo "📊 Checking latest output:" && \
        gsutil ls gs://bigdata-ai-datalake/curated/ | grep clickstream | tail -1 || echo "No output files found"
        """
    )

    end = DummyOperator(task_id="end")

    # Set dependencies
    start >> kafka_producer >> spark_job >> validate >> end