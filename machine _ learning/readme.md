# 🧠 Machine Learning Module

Part of the **E2E Big Data & AI Platform**

This module contains all machine learning pipelines used for:

* Customer churn prediction
* Clickstream behavioral modeling
* Multilingual sentiment analysis
* Model training, evaluation, and deployment
* Exporting models & results to the Data Lake

The ML layer corresponds to **Stage 5** of the platform architecture:
`Raw → Curated → Enriched → ML Models → Dashboards`.

---

## 📁 Folder Structure

```text
machine_learning/
│
├── clickstream_churn_model/
│   ├── model/
│   │   └── new_churn_model/
│   ├── model_traning.py
│
├── customer churn model/
│   ├── models/
│   ├── model_creation.py
│   └── predict_model.py
│
├── review_model/
│   └── sentiment_analysis_model_reviews.py
│
└── README.md
```

---

# 🚀 Module Overview

## 1️⃣ Clickstream Churn Model

**Folder:** `clickstream_churn_model/`

### Purpose

Predict customer churn based on:

* Page views
* Add-to-cart events
* Purchases
* Session-level interactions

### Files

| File                     | Description                                       |
| ------------------------ | ------------------------------------------------- |
| `model/new_churn_model/` | Stores trained churn model artifacts              |
| `model_traning.py`       | Prepares dataset + trains clickstream churn model |

---

## 2️⃣ Customer Churn Model

**Folder:** `customer churn model/`

### Purpose

General churn prediction model using:

* Customer demographics
* Purchase history
* Engagement patterns

### Files

| File                | Description                                       |
| ------------------- | ------------------------------------------------- |
| `model_creation.py` | Full end-to-end churn model training pipeline     |
| `predict_model.py`  | Loads saved model and generates churn predictions |
| `models/`           | Saved model files                                 |

---

## 3️⃣ Review Sentiment Analysis Model

**Folder:** `review_model/`
**File:** `sentiment_analysis_model_reviews.py`

### Purpose

Perform multilingual sentiment analysis using:
`tabularisai/multilingual-sentiment-analysis`

### Pipeline Steps

1. Load review data from **Google Cloud Storage (Parquet)**
2. Clean and preprocess text
3. Run local inference (GPU/CPU)
4. Predict sentiment classes:

   * Very Negative
   * Negative
   * Neutral
   * Positive
   * Very Positive
5. Evaluate model using product ratings
6. Save results (CSV + Parquet)
7. Upload final models & results back to GCS

### Required Environment Variables

Configure before running the script:

```bash
HF_TOKEN="your_huggingface_token"
GCP_SERVICE_ACCOUNT_FILE="C:/keys/service-account.json"
```

*No tokens or keys are stored in the code.*

---

# ⚙️ Installation Requirements

Install ML dependencies:

```bash
pip install torch transformers google-cloud-storage pandas numpy seaborn matplotlib scikit-learn tqdm
```

---

# ▶️ Running the Pipelines

### Train Clickstream Churn Model

```bash
python clickstream_churn_model/model_traning.py
```

### Run Customer Churn Predictions

```bash
python "customer churn model/predict_model.py"
```

### Run Sentiment Analysis Pipeline

```bash
python review_model/sentiment_analysis_model_reviews.py
```

---

# 📊 Output Files

### Clickstream & Customer Churn Models

* Model artifacts stored in `models/`
* Training reports
* Prediction outputs (CSV)

### Sentiment Analysis

* `sentiment_analysis_results.csv`
* `sentiment_analysis_results.parquet`
* Local saved model in `saved_sentiment_model/`

### Google Cloud Upload Locations

```text
gs://bigdata-ai-datalake/models/sentiment_analysis/
gs://bigdata-ai-datalake/results/
```

---

# 🔄 ML Layer in Platform Architecture

```text
Stage 1 → Data Sources
Stage 2 → Ingestion
Stage 3 → ETL / Transformation
Stage 4 → Processing (Spark / Flink)
👉 Stage 5 → Machine Learning (this module)
Stage 6 → Dashboards / UI
```

---

# 🧩 Future Enhancements

* Integrate MLflow for experiment tracking
* Hyperparameter tuning with Optuna
* Convert models to ONNX for faster inference
* Build FastAPI inference service
* Add a Feature Store (Feast)
