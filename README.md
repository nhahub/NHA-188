# 🚀 End-to-End Enterprise Big Data & AI Platform for Churn, Clickstream Behavior and Sentiment Prediction

A complete **Enterprise-Grade Data & AI Platform** that processes large-scale datasets, automates ETL pipelines, builds ML models, and exposes insights through interactive dashboards.  
This project simulates how modern companies design scalable data systems capable of handling analytics, prediction, and real-time processing.

---

# 📌 Project Overview

This platform integrates **Data Engineering + Machine Learning + BI Dashboards** into one unified system.

It includes:

- Multi-source data ingestion  
- Automated ETL pipelines  
- Batch & streaming processing  
- Feature engineering  
- ML models (Churn, Sentiment, Clickstream behavior)  
- Unified Streamlit analytics app  
- Power BI dashboards  

---

# 👥 Team Members

<table align="center">
  <tr>
    <td align="center">
      <b>Team Leader </b><br>
      <b>Abdullah Ibrahim Mahmoud </b><br>
      <a href="https://www.linkedin.com/in/abdullah-a-ibrahim">LinkedIn</a> |
      <a href="https://github.com/ABDULLAH-ibrahimm">GitHub</a> 
    </td>
    <td align="center">
     <br>
      <b>Mansour Mohamed Mansour</b><br>
      <a href="https://www.linkedin.com/in/mansour-mohamed74">LinkedIn</a> |
      <a href="https://github.com/Mansourmohamed1">GitHub</a> 
    </td>
     <td align="center">
      <b></b>Ezzeldeen Elsayed Mohammed Abdelhamid<br>
      <a href="https://linkedin.com/in/ezzeldeen-farahat">LinkedIn</a> |
      <a href="https://github.com/Ezz194">GitHub</a> 
    </td>
  </tr>
  <tr>
    <td align="center">
      <b></b>Ahmed Mohamed Ahmed Diefallah<br>
      <a href="https://linkedin.com/in/ahmed-diefallah">LinkedIn</a> |
      <a href="https://github.com/ahmeddif900">GitHub</a> 
    </td>
    <td align="center">
      <b></b>Jessica Ashraf Anis<br>
      <a href="https://www.linkedin.com/in/jessica-ashraf">LinkedIn</a> |
      <a href="https://github.com/jessicaAshraf">GitHub</a> 
    </td>
  </tr>
</table>

---

# 🏗️ Platform Architecture

![architecture](orchestration.png)

The platform follows a **6-stage architecture**, each performing a critical role in the data and AI lifecycle.

---

# 🔷 Stage 1 — Data Sources (Collection Layer)

This stage collects **structured, semi-structured, and unstructured** data from different business domains.

### **Data Types Included**
| Data Source | Format | Description |
|-------------|--------|-------------|
| Transactions | CSV | Purchases, payments, revenue metrics |
| Clickstream Logs | CSV / JSON | User navigation events (views, purchases, add-to-cart) |
| Complaints | Excel | Customer support & issue tickets |
| Reviews | JSON / CSV | Customer textual feedback for NLP |

### **Why this stage is important**
✔ Provides centralized raw data for all analytics  
✔ Ensures data diversity (behavioral + text + financial)  
✔ Forms the foundation for ML models  

---

# 🔷 Stage 2 — Ingestion & Storage (Raw Zone)

This layer moves raw data → **Cloud Data Lake**, using:

### **Tools & Technologies**
- **Apache Airflow** → Scheduled DAGs for ingestion  
- **GCP Storage** → Scalable object storage  
- **Kafka Producers** → Real-time clickstream ingestion  

### **Pipeline Responsibilities**
- Download raw files  
- Validate schema & data types  
- Store data in the `raw/` zone of the Data Lake  
- Trigger downstream ETL pipelines  

### **Output Structure**
```text
data_lake/
├── raw/
├── processed/
└── curated/
```

---

# 🔷 Stage 3 — ETL Processing (Processed Zone)

Performed using **Apache Spark**.

### **Key ETL Operations**
✔ Remove duplicates  
✔ Handle missing values  
✔ Convert formats (CSV → Parquet)  
✔ Standardize columns  
✔ Join datasets (transactions + clickstream)  
✔ Extract metrics (sales, frequency, returns)  

### **Why Spark?**
- Distributed compute (handles millions of rows)  
- Fast transformations compared to Pandas  
- Integrates well with cloud storage  

---

# 🔷 Stage 4 — Feature Engineering (Curated Zone)

This stage transforms cleaned data into machine-learning-ready features.

### **Clickstream Feature Engineering**
- Session duration  
- Number of views  
- Add-to-cart behavior  
- Purchases count  
- Conversion rate  
- Return behavior  

### **Customer Feature Engineering**
- Tenure  
- Monthly charges  
- Internet service type  
- Contract type  
- Payment method  
- Total charges  

### **Text Review Feature Engineering**
- Tokenization (Transformers)  
- Sentiment class extraction  
- Confidence scores  

---

# 🔷 Stage 5 — Machine Learning Layer

This stage contains **3 AI models**.

---

## 1️⃣ Big Data Customer Churn Model (PySpark ML)

### **Pipeline Steps**
1. StringIndexer  
2. OneHotEncoder  
3. VectorAssembler  
4. StandardScaler  
5. RandomForestClassifier  
6. Cross-validation (Grid Search)

### **Purpose**
Predict whether a customer will churn using demographic & service features.

---

## 2️⃣ Clickstream Behavior & Churn Model

### **Model Highlights**
- Trained on user activity logs  
- Predicts churn likelihood  
- Generates churn probability score  
- Integrated with Streamlit for real-time demo  

---

## 3️⃣ Sentiment Analysis Model (Transformers)

### **Model Architecture**
- HuggingFace multilingual transformer  
- GPU-enabled inference  
- Batch prediction support  

### **Outputs**
- Very Negative  
- Negative  
- Neutral  
- Positive  
- Very Positive  
- Confidence level  

---

# 🔷 Stage 6 — Visualization Layer

Two main visualization systems:

---

## 🟣 Streamlit Unified Dashboard

Includes 3 full apps:

| App | Function |
|-----|----------|
| **Clickstream App** | Real-time churn prediction & dashboard |
| **Customer Churn App** | ML predictions + batch analysis |
| **Sentiment App** | NLP model for reviews + visual insights |

### **Run the app**
```bash
streamlit run "streamlit and analysis/final_app.py"
```

---

## 🔵 Power BI Dashboard

Includes:

* Sales analytics
* Customer segmentation
* Churn breakdown
* Behavior funnel
* Geo-distribution

---

# 🔧 Technologies Used

| Category      | Tools                     |
| ------------- | ------------------------- |
| Programming   | Python, SQL               |
| Big Data      | Spark, Kafka, Hadoop      |
| Orchestration | Airflow                   |
| ML            | PySpark ML, Transformers  |
| Cloud         | Azure / AWS               |
| Dashboards    | Streamlit, Power BI       |
| DevOps        | Git, Virtual Environments |

---

# 🟩 Run Locally

```bash
pip install -r requirements.txt
streamlit run "streamlit and analysis/final_app.py"
```

---

# ⭐ Future Enhancements

* Add MLflow for experiment tracking
* Deploy Spark cluster on Kubernetes
* Add real-time anomaly detection
* Implement feature store (Feast)
* Automate full MLOps pipeline

---

# 🎉 Credits

Thanks to our amazing team for building a fully functional E2E Data & AI Platform.
