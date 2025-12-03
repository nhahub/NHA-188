# E2E Big Data & AI Platform 🚀

## 🎯 Project Idea
This project aims to build an **End-to-End Data & AI Platform** that covers the entire data lifecycle:

1. **Data Collection** from multiple sources (Structured + Unstructured).  
2. **Storage & Processing** in a unified Data Lake.  
3. **ETL / Data Transformation** using Apache Airflow & Azure Data Factory.  
4. **Analytics** with Batch processing (Apache Spark) and Streaming processing (Apache Flink).  
5. **Machine Learning** models for customer behavior prediction (e.g., churn detection).  
6. **Visualization & Dashboards** through Power BI, and Streamlit.  

---

## 🏗️ Orchestration Diagram
![Orchestration](orchestration.png)

---

## 📂 Stage 1: Data Sources

- **Transactions (CSV):** Sales and financial transaction data.  
- **Clickstream / Logs (CSV/):** User activity events (page_view, add_to_cart, purchase)  >> use for streaming.  
- **Complaints (Excel):** Customer service and complaints records.  
- **Reviews (JSON):** Customer feedback, surveys, and product reviews.  

📌 All raw data sources are stored in the `data_sources/` folder.  

---

## ✅ Next Steps
- Upload raw datasets into the **Data Lake (Raw Zone)**.  
- Prepare **Stage 2: Ingestion & Storage** pipelines.  
