# Unified Analytics Platform — Project Summary & Run Guide

Short summary
-------------
This repository contains a set of Streamlit applications for three related analytics projects:
1. Clickstream Analytics — churn prediction and dashboards from web clickstream data.
2. Customer Churn — large-scale churn pipeline using PySpark (ETL, feature engineering, model training).
3. Reviews / Sentiment Analysis — sentiment classification for customer reviews (single and batch).

There is also a unified Streamlit application that integrates the three apps into a single interface for demo and exploration.

Repository layout (based on the screenshot and provided files)
--------------------------------------------------------------
- streamlit and analysis/ (project root)
  - clickstream_churn_app/
    - model/new_churn_model/         ← PySpark PipelineModel folder (if present)
    - clickstream.pbix               ← Power BI report
    - final.py                       ← Clickstream Streamlit app
    - image.png                      ← Dashboard image or placeholder
  - customer churn_app/
    - models/                        ← Trained models (PySpark) (if present)
    - Customer Churn.pbix            ← Power BI report
    - app.py                         ← Churn Streamlit app
  - reviews_app/
    - app.py                         ← Sentiment Streamlit app
    - final_app.py                   ← Alternate/combined sentiment app
  - (optional) unified app file combining the three apps — may exist as a large single Python file

Application descriptions
------------------------
- Clickstream Analytics (final.py)
  - Streamlit interface for system metrics, model loading (PySpark PipelineModel), demo/live predictions, simulated mode if model missing, visualization dashboards, and Power BI integration.

- Customer Churn (app.py)
  - Big Data churn pipeline with ETL (Spark), feature engineering, RandomForest model training and inference. Supports single predictions and batch CSV inference. Falls back to simulation when Spark or models are not available.

- Reviews / Sentiment Analysis (app.py / final_app.py)
  - Uses Hugging Face transformers (local saved model folder `./saved_sentiment_model`) for sentiment classification. Supports single-text analysis and batch processing from CSV/XLSX/Parquet.

Prerequisites
-------------
- Python 3.8+
- pip (or conda)
- If using PySpark: Java (OpenJDK 8/11) installed and JAVA_HOME set
- Enough RAM/disk for running models/Spark locally (8GB+ recommended for moderate workloads)

Suggested installation (virtual environment)
--------------------------------------------
Unix / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Suggested requirements.txt (example)
------------------------------------
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- pyspark         # optional — required for the Spark apps
- transformers    # optional — required for sentiment model
- torch           # optional — if transformers use PyTorch
- openpyxl
- pyarrow

Example:
```
streamlit>=1.20
pandas
numpy
matplotlib
seaborn
plotly
pyspark
transformers
torch
openpyxl
pyarrow
```

How to run each app
-------------------
1. Clickstream app:
   - cd clickstream_churn_app
   - streamlit run final.py
   - If the model folder (new_churn_model) is missing, the app will run in simulated mode.

2. Customer Churn app:
   - cd "customer churn_app"
   - streamlit run app.py
   - If Spark or model artifacts are missing, the app defaults to simulated predictions.

3. Reviews / Sentiment app:
   - cd reviews_app
   - streamlit run app.py
   - Ensure `./saved_sentiment_model` exists or the environment has internet access to download the model.

4. Unified app:
   - If you have a single unified Python file (the large code you supplied), run:
     - streamlit run <unified_app_filename>.py

Notes about models and data
---------------------------
- Model paths in the code may be absolute or relative. Examples:
  - Clickstream: `/home/ezz/.../models/new_churn_model` or `/mnt/c/.../model/new_churn_model`
  - Churn project: `models/churn_model` and `models/feature_engineering_pipeline`
  - Sentiment: `./saved_sentiment_model`
- If models are not found at these paths, the apps show warnings and use simulation fallback logic so the UI remains usable.
- Power BI (.pbix) files in project folders can be opened with Power BI Desktop for interactive reports.

CSV templates and expected columns
---------------------------------
- Batch prediction screens provide CSV templates for upload:
  - churn_input_template.csv
  - churn_prediction_template.csv
  - churn_prediction_template.csv (for clickstream batch demo)
- CSVs should include expected column headers (e.g., customer_id, age, gender, tenure_months, monthly_charges, contract_type, internet_service, tech_support, payment_method, total_charges). Match your input file headers to the expected names.

Common troubleshooting tips
---------------------------
- PySpark errors: verify Java installation and JAVA_HOME, ensure pyspark version matches your Java.
- Transformers / torch load errors: make sure `transformers` and `torch` are installed; if model downloads from Hugging Face, ensure internet access.
- Streamlit issues: update streamlit, ensure virtual environment is activated, check for conflicting ports.

Security & privacy
------------------
- Do not commit or publish PII or sensitive datasets in a public repository.
- Store heavy model artifacts outside the repo (e.g., artifact storage, model registry) and reference them through scripts or environment configuration.

Suggested improvements (I can add these)
----------------------------------------
- Add a real `requirements.txt` and `environment.yml` (conda).
- Add a simple run script (run.sh / run.ps1) to start a chosen app quickly.
- Add an examples/ or data/ folder with sanitized sample CSVs for testing.
- Move large blocks of app code into modular Python modules for maintainability.

