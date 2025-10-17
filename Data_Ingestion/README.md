# Data Ingestion Pipeline into Data Lake

This document explains the design and steps of our data ingestion pipeline, including how data moves across the **three zones** of the Data Lake.

---

## 🌐 Data Lake Location
`https://console.cloud.google.com/storage/browser/bigdata-ai-datalake?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&project=elegant-verbena-471612-d8`

---

## 🏗️ Zones Design

### 1️⃣ Raw Zone
- **Purpose**: Store raw data exactly as ingested, without modifications.  
- **Sources**: GitHub repository (CSV & RAR files).  
- **Process**: Data is ingested daily using **Airflow DAG** (`data_ingestion_github_to_gcs`).  
- **Files**:
  - `Cust-churn.csv`
  - `E-commerce_Website_Logs.csv`
  - `consumer_complaints.rar`
  - `reviews.rar`

---

### 2️⃣ Curated Zone
- **Purpose**: Clean, standardize, and prepare data for downstream usage.  
- **Processing**:  
  - **Batch processing** for structured files (CSV, RAR extracted data) using **Spark**.  
  - **Streaming processing** for clickstream logs:
    - Ingested through **Kafka**.  
    - Processed in real-time with **Apache  spark **.  
- **Output**: Curated data is stored in partitioned (e.g., Parquet or csv  format for optimized analytics.

---

### 3️⃣ Enriched Zone
- **Purpose**: Store enriched datasets ready for advanced analytics, ML, and visualization.  
- **Examples**:
  - Aggregated KPIs from churn and complaints data.  
  - Real-time dashboards on website clickstream behavior.  
- **Tools**:
  - BI Tools (e.g., Looker Studio, Power BI, Tableau).  
  - Machine Learning pipelines (future extension).  

---

## ⚙️ Pipeline Flow

1. **Ingestion**  
   Raw files are ingested from GitHub → GCS **Raw Zone**.

2. **Processing**  
   - Batch jobs (Spark) clean and transform data → **Curated Zone**.  
   - Streaming jobs (Kafka + spark ) process real-time clickstream logs → **Curated Zone**.  

3. **Enrichment & Visualization**  
   Final datasets are stored in the **Enriched Zone** for reporting, dashboards, and ML.


