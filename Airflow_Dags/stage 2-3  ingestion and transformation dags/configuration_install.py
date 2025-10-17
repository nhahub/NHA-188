from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="setup_spark_environment_simple",
    default_args=default_args,
    description="Simple Spark environment setup",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["setup", "spark", "jars"],
) as dag:

    start = DummyOperator(task_id="start")

    # Install Kafka JARs on master (with error handling)
    install_kafka_jars_master = BashOperator(
        task_id="install_kafka_jars_master",
        bash_command="""
        echo "📦 Installing Kafka JARs on spark-master..." && \
        docker exec spark-master bash -c '
            cd /opt/spark/jars && \
            wget -q https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar || echo "JAR already exists" && \
            wget -q https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.0/kafka-clients-3.4.0.jar || echo "JAR already exists" && \
            wget -q https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar || echo "JAR already exists" && \
            echo "✅ Kafka JARs installed on master"
        '
        """
    )

    # Install Kafka JARs on worker
    install_kafka_jars_worker = BashOperator(
        task_id="install_kafka_jars_worker",
        bash_command="""
        echo "📦 Installing Kafka JARs on spark-worker..." && \
        docker exec spark-worker bash -c '
            cd /opt/spark/jars && \
            wget -q https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.1/spark-sql-kafka-0-10_2.12-3.5.1.jar || echo "JAR already exists" && \
            wget -q https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.0/kafka-clients-3.4.0.jar || echo "JAR already exists" && \
            wget -q https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.1/spark-token-provider-kafka-0-10_2.12-3.5.1.jar || echo "JAR already exists" && \
            echo "✅ Kafka JARs installed on worker"
        '
        """
    )

    # Install commons-pool2
    install_commons_pool = BashOperator(
        task_id="install_commons_pool",
        bash_command="""
        echo "📦 Installing commons-pool2..." && \
        docker exec spark-master bash -c 'cd /opt/spark/jars && wget -q https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar || echo "JAR already exists"' && \
        docker exec spark-worker bash -c 'cd /opt/spark/jars && wget -q https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar || echo "JAR already exists"' && \
        echo "✅ Commons-pool2 installed"
        """
    )

    # Install GCS connector
    install_gcs_connector = BashOperator(
        task_id="install_gcs_connector",
        bash_command="""
        echo "📦 Installing GCS connector..." && \
        docker exec spark-master bash -c 'cd /opt/spark/jars && wget -q https://repo1.maven.org/maven2/com/google/cloud/bigdataoss/gcs-connector/hadoop3-2.2.18/gcs-connector-hadoop3-2.2.18.jar || echo "JAR already exists"' && \
        docker exec spark-worker bash -c 'cd /opt/spark/jars && wget -q https://repo1.maven.org/maven2/com/google/cloud/bigdataoss/gcs-connector/hadoop3-2.2.18/gcs-connector-hadoop3-2.2.18.jar || echo "JAR already exists"' && \
        echo "✅ GCS connector installed"
        """
    )

    # Install Python dependencies
    install_python_deps = BashOperator(
        task_id="install_python_deps",
        bash_command="""
        echo "📦 Installing Python dependencies..." && \
        docker exec -u root spark-master pip install -q google-cloud-storage pandas==1.5.3 && \
        docker exec -u root spark-worker pip install -q google-cloud-storage pandas==1.5.3 && \
        echo "✅ Python dependencies installed"
        """
    )

    # Setup GCP key
    setup_gcp_key = BashOperator(
        task_id="setup_gcp_key",
        bash_command="""
        echo "🔑 Setting up GCP key..." && \
        docker exec -u root spark-master mkdir -p /opt/airflow/dags/keys && \
        docker exec -u root spark-worker mkdir -p /opt/airflow/dags/keys && \
        docker cp /opt/airflow/dags/keys/gcp.json spark-master:/opt/airflow/dags/keys/gcp.json && \
        docker cp /opt/airflow/dags/keys/gcp.json spark-worker:/opt/airflow/dags/keys/gcp.json && \
        echo "✅ GCP key setup completed"
        """
    )

    # Final verification
    verify_setup = BashOperator(
        task_id="verify_setup",
        bash_command="""
        echo "🔍 Verifying setup..." && \
        echo "Master JARs:" && \
        docker exec spark-master ls /opt/spark/jars/ | grep -E "(kafka|gcs)" | head -5 && \
        echo "Worker JARs:" && \
        docker exec spark-worker ls /opt/spark/jars/ | grep -E "(kafka|gcs)" | head -5 && \
        echo "✅ Setup verification completed"
        """
    )

    end = DummyOperator(task_id="end")

    # Set dependencies
    start >> install_kafka_jars_master >> install_kafka_jars_worker >> install_commons_pool >> install_gcs_connector >> install_python_deps >> setup_gcp_key >> verify_setup >> end