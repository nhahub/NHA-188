# Airflow_Dags – End-to-End Big Data & AI Platform Orchestration

This folder contains all Apache Airflow DAGs responsible for orchestrating the data pipeline across the E2E Big Data & AI ecosystem.  
The DAGs implement ingestion, transformation, data lineage tracking, quality validation, and storage into the different Data Lake zones.

The pipeline is aligned with the following architecture:

### **📌 End-to-End Workflow Stages**
1. **Stage 1 – Data Sources**  
   - Reads data from SQL/CSV, JSON/Parquet logs, Excel complaints, surveys, clickstream logs, and streaming events.  
   - Raw files are placed in the `data_sources/` directory.

2. **Stage 2 – Ingestion & Storage**  
   - Airflow triggers Azure Data Factory Batch ingestion or Kafka/Event Hub streaming ingestion.  
   - Data Lake Gen2 Zones:
     - **Raw Zone**
     - **Curated Zone**
     - **Enriched Zone**

3. **Stage 3 – Transformation & ETL**  
   - Airflow + ADF pipelines perform:
     - Data cleaning & normalization  
     - Schema alignment  
     - Metadata & lineage tracking  
     - Data quality validation  

4. **Stage 4 – Processing  Layer**  
   - take data from curated to enriched for prepare data to ml and analytics 

## 📁 Folder Structure Overview

```plaintext
Airflow_Dags/
│
├── Stage-1 (data source dags)/
│     └── DAGs for loading raw data sources into the Raw Zone
│
├── stage 2-3 ingestion and transformation dags/
│     └── DAGs for ingestion into Data Lake + transformation
│
├── stage 4 from cruted to enriched/
│     └── DAGs for Curated → Enriched ETL pipelines
│
└── README.md   (this file)
