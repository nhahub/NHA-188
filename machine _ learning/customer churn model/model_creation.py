# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (myenv)
#     language: python
#     name: myenv
# ---

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pandas as pd

# +
from google.cloud import storage
import tempfile
import os
import shutil
from pyspark.sql import SparkSession

# Set up Spark session
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json"

spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.jars.packages", "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.11") \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/home/ezz/Desktop/DEPI/lateral-medium-471608-g6-8cdb9e6cafd6.json") \
    .getOrCreate()

def download_parquet_directory(bucket_name, source_dir, local_dir):
    """Download entire Parquet directory from GCS and keep files"""
    storage_client = storage.Client()

    # List all parquet files (excluding .crc files)
    blobs = list(storage_client.list_blobs(bucket_name, prefix=source_dir))
    parquet_files = [blob for blob in blobs if blob.name.endswith('.parquet') and not blob.name.endswith('.crc')]

    print(f"Found {len(parquet_files)} Parquet files")

    # Download all parquet files
    for blob in parquet_files:
        # Extract filename from blob path
        filename = os.path.basename(blob.name)
        local_path = os.path.join(local_dir, filename)

        print(f"Downloading: {blob.name} -> {local_path}")
        blob.download_to_filename(local_path)

    return local_dir

# Create a persistent directory (not temporary)
persistent_dir = "./churn_data_parquet"
os.makedirs(persistent_dir, exist_ok=True)

try:
    # Download files to persistent directory
    download_parquet_directory(
        "bigdata-ai-datalake",
        "enriched/churn_train.parquet/",
        persistent_dir
    )

    # Read the entire directory as a Spark DataFrame
    df = spark.read.parquet(persistent_dir)

    print("Successfully loaded data!")
    print(f"Schema:")
    df.printSchema()
    print(f"Row count: {df.count()}")
    df.show(10, truncate=False)

    # Show some basic stats
    print(f"Column count: {len(df.columns)}")
    print(f"Columns: {df.columns}")

finally:
    # Optional: Clean up only after you're done with all processing
    # shutil.rmtree(persistent_dir)
    print(f"Files are available at: {persistent_dir}")
# -

# --- 2. Define Your GCS Paths ---
# !!! Replace these with your values !!!
GCS_BUCKET = "bigdata-ai-datalake"
DATA_PATH = f"./churn_data_parquet"
MODEL_SAVE_PATH = f"gs://{GCS_BUCKET}/models/"
# bigdata-ai-datalake
# !!! Replace with your actual column names !!!
FEATURES_COL = "features"  # The vector you created
LABEL_COL = "label" # The column you're trying to predict (e.g., 'churn')

# +
import os
from pyspark.sql import SparkSession

# Check what's in the folder
print("Contents of churn_data_parquet folder:")
if os.path.exists("./churn_data_parquet"):
    files = os.listdir("./churn_data_parquet")
    for file in files:
        file_path = os.path.join("./churn_data_parquet", file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file} ({file_size} bytes)")
else:
    print("❌ Folder 'churn_data_parquet' not found!")
    print("Current directory:", os.getcwd())
    print("Available folders:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  - {item}/")

# If folder exists, read the data
if os.path.exists("./churn_data_parquet") and any(file.endswith('.parquet') for file in os.listdir("./churn_data_parquet")):
    spark = SparkSession.builder.appName("ChurnData").getOrCreate()
    df = spark.read.parquet("./churn_data_parquet")

    print("\n✅ Data loaded successfully!")
    print(f"Dataset shape: {df.count()} rows, {len(df.columns)} columns")
    df.show(5)

# +
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# count number of labels
num_labels = df.select("label").distinct().count()

# rows per label
n = 5000 // num_labels

# add a random number
df_rand = df.withColumn("rand", F.rand())

# rank rows per label
w = Window.partitionBy("label").orderBy("rand")

balanced_df = df_rand.withColumn("rn", F.row_number().over(w)) \
                     .filter(F.col("rn") <= n) \
                     .drop("rand", "rn")
# -

balanced_df.groupBy('label').count().show()

pdf = balanced_df.toPandas()

# +
import numpy as np

# Convert vector column to numpy arrays
X = np.array([x.toArray() for x in pdf["features"]])
y = pdf["label"].values

# +
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_impurity_decrease": [0.0, 0.01, 0.05]
}
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X, y)

best_params = grid.best_params_
best_params

# +

rf = RandomForestClassifier(featuresCol=FEATURES_COL, labelCol=LABEL_COL,

    numTrees=best_params["n_estimators"],
    maxDepth=best_params["max_depth"],
    minInfoGain=best_params["min_impurity_decrease"]
)
# -

model = rf.fit(balanced_df)

try:
    model.write().overwrite().save(".")
    # model.write().overwrite().save(MODEL_SAVE_PATH)
    print(f"Best model successfully saved to {MODEL_SAVE_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")


