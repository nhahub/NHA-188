# -*- coding: utf-8 -*-
"""
Sentiment analysis on product reviews using:
- Hugging Face model: tabularisai/multilingual-sentiment-analysis
- Data stored in Google Cloud Storage (Parquet)
- Optional upload of model & results back to GCS

IMPORTANT (for GitHub / production use):
- Do NOT hard-code any secrets (HF_TOKEN, GCP JSON, etc.)
- Configure the following environment variables before running:
    HF_TOKEN                 -> Hugging Face access token (optional: only if using remote inference)
    GCP_SERVICE_ACCOUNT_FILE -> Path to GCP service account JSON (for GCS access)
"""

import os
import tempfile

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import InferenceClient

from google.cloud import storage
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
BUCKET_NAME = "bigdata-ai-datalake"          # change if needed
REVIEWS_PARQUET_PREFIX = "curated/reviews.parquet/"
GCS_MODEL_PATH = "models/sentiment_analysis/"
GCS_RESULTS_CSV_PATH = "results/sentiment_analysis.csv"
GCS_RESULTS_PARQUET_PATH = "results/sentiment_analysis.parquet"

LOCAL_RESULTS_CSV = "sentiment_analysis_results.csv"
LOCAL_RESULTS_PARQUET = "sentiment_analysis_results.parquet"
LOCAL_SAVED_MODEL_DIR = "./saved_sentiment_model"


# -----------------------------------------------------------------------------
# HUGGING FACE: LOCAL & OPTIONAL REMOTE INFERENCE
# -----------------------------------------------------------------------------

def load_local_model(model_name: str = MODEL_NAME, device: str = None):
    """Load tokenizer and model locally (CPU or GPU)."""
    print(f"Loading model '{model_name}' locally...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    print(f"Model loaded successfully on {device_obj}")
    return tokenizer, model, device_obj


def get_hf_inference_client() -> InferenceClient | None:
    """
    Optional remote inference client.
    Requires HF_TOKEN in the environment.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN not set. Remote inference client will not be initialized.")
        return None

    client = InferenceClient(
        provider="auto",
        api_key=hf_token,
    )
    print("✅ Hugging Face InferenceClient initialized (remote inference available).")
    return client


def example_remote_inference(client: InferenceClient):
    """
    Simple example of remote inference (optional).
    """
    if client is None:
        print("⚠️ Remote client is None. Skipping remote inference example.")
        return

    example_text = (
        "This spray is really nice. It smells really good, goes on really fine, "
        "and does the trick. I will say it feels like you need a lot of it though "
        "to get the texture I want."
    )

    print("Running example remote inference via Hugging Face Inference Providers...")
    result = client.text_classification(
        example_text,
        model=MODEL_NAME,
    )
    print("Remote inference result:", result)


# -----------------------------------------------------------------------------
# GOOGLE CLOUD STORAGE HELPERS
# -----------------------------------------------------------------------------

def get_gcs_client() -> storage.Client:
    """
    Initialize a GCS client using a service account JSON path provided via
    environment variable GCP_SERVICE_ACCOUNT_FILE.
    """
    cred_path = os.getenv("GCP_SERVICE_ACCOUNT_FILE")
    if not cred_path:
        raise RuntimeError(
            "Environment variable GCP_SERVICE_ACCOUNT_FILE is not set. "
            "Please point it to your GCP service account JSON file."
        )

    if not os.path.isfile(cred_path):
        raise FileNotFoundError(
            f"GCP service account file not found: {cred_path}"
        )

    print(f"Using GCP credentials file: {cred_path}")
    client = storage.Client.from_service_account_json(cred_path)
    return client


def download_reviews_parquet_from_gcs(
    client: storage.Client,
    bucket_name: str = BUCKET_NAME,
    prefix: str = REVIEWS_PARQUET_PREFIX,
) -> pd.DataFrame:
    """
    Download all Parquet files from a GCS prefix into a temp directory
    and load them into a single pandas DataFrame.
    """
    print(f"Downloading Parquet files from gs://{bucket_name}/{prefix} ...")
    bucket = client.bucket(bucket_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        local_parquet_dir = os.path.join(temp_dir, "reviews")
        os.makedirs(local_parquet_dir, exist_ok=True)

        # List and download all parquet files
        blobs = bucket.list_blobs(prefix=prefix)
        local_files = []

        for blob in blobs:
            if blob.name.endswith(".parquet") and not blob.name.endswith("/"):
                local_path = os.path.join(local_parquet_dir, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
                local_files.append(local_path)
                print(f"✅ Downloaded: {blob.name}")

        if not local_files:
            raise RuntimeError("No Parquet files found in the specified prefix.")

        # Read all parquet files and concatenate
        dfs = [pd.read_parquet(fp) for fp in local_files]
        df = pd.concat(dfs, ignore_index=True)

        print(f"📊 Loaded {len(df)} rows from {len(local_files)} Parquet file(s).")
        print("📋 Columns:", df.columns.tolist())
        print("\n🔍 First 5 rows:")
        print(df.head())

        return df


def upload_to_gcs(client: storage.Client, local_path: str, gcs_path: str):
    """Upload file or folder to GCS."""
    bucket = client.bucket(BUCKET_NAME)

    if os.path.isfile(local_path):
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} to gs://{BUCKET_NAME}/{gcs_path}")

    elif os.path.isdir(local_path):
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                gcs_file_path = os.path.join(gcs_path, relative_path).replace("\\", "/")

                blob = bucket.blob(gcs_file_path)
                blob.upload_from_filename(local_file_path)
                print(f"Uploaded {local_file_path} to gs://{BUCKET_NAME}/{gcs_file_path}")


# -----------------------------------------------------------------------------
# SENTIMENT PREDICTION & EVALUATION
# -----------------------------------------------------------------------------

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned_text from 'title' and 'text' columns."""
    print("Preprocessing text data...")
    df = df.copy()
    df["text"] = df["text"].fillna("")
    df["title"] = df["title"].fillna("")
    df["combined_text"] = df["title"] + " " + df["text"]
    df["cleaned_text"] = df["combined_text"].str.strip()

    print(f"Processed {len(df)} text samples.")
    return df


def predict_sentiment_batch(
    texts: pd.Series,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Predict sentiment for a batch of texts."""
    predictions = []
    confidences = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting sentiment"):
        batch_texts = texts.iloc[i: i + batch_size].tolist()

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            batch_confidences = torch.max(probabilities, dim=-1).values

        predictions.extend(batch_predictions.cpu().numpy())
        confidences.extend(batch_confidences.cpu().numpy())

    return predictions, confidences


def map_predictions_to_labels(predictions):
    """Map numerical predictions to sentiment labels."""
    label_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive",
    }
    return [label_map.get(int(pred), "Unknown") for pred in predictions]


def add_true_sentiment_from_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple true_sentiment label from a numeric rating column:
      rating >= 4 -> Positive
      rating <= 2 -> Negative
      otherwise   -> Neutral
    """
    df = df.copy()

    if "rating" not in df.columns:
        print("⚠️ 'rating' column not found. true_sentiment will not be created.")
        df["true_sentiment"] = np.nan
        return df

    df["true_sentiment"] = df["rating"].apply(
        lambda x: "Positive" if x >= 4 else "Negative" if x <= 2 else "Neutral"
    )
    return df


def evaluate_model(df_result: pd.DataFrame):
    """Print basic evaluation metrics and show confusion matrix."""
    if "true_sentiment" not in df_result.columns:
        print("⚠️ 'true_sentiment' column not found. Skipping evaluation.")
        return

    y_true = df_result["true_sentiment"].dropna()
    y_pred = df_result.loc[y_true.index, "predicted_sentiment"]

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix - Multilingual Sentiment Analysis")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# SAVE / LOAD MODEL
# -----------------------------------------------------------------------------

def save_model_and_tokenizer(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    save_path: str = LOCAL_SAVED_MODEL_DIR,
):
    """
    Save the entire model and tokenizer to a local directory.
    """
    print(f"Saving model and tokenizer to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")


def load_model_and_tokenizer_from_dir(
    save_path: str = LOCAL_SAVED_MODEL_DIR,
    device: str | None = None,
):
    """
    Load model and tokenizer from a local directory.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device)

    print(f"Loading model and tokenizer from {save_path}...")
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.to(device_obj)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model, device_obj


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    # 1) Initialize clients (GCS + optional HF remote client)
    gcs_client = get_gcs_client()
    hf_client = get_hf_inference_client()
    example_remote_inference(hf_client)  # optional demonstration

    # 2) Download reviews data
    df = download_reviews_parquet_from_gcs(gcs_client)

    # 3) Preprocess text
    df = preprocess_reviews(df)

    # 4) Load model locally
    tokenizer, model, device = load_local_model(MODEL_NAME)

    # 5) Predict sentiments
    texts = df["cleaned_text"]
    predictions, confidences = predict_sentiment_batch(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    sentiment_labels = map_predictions_to_labels(predictions)

    df_result = df.copy()
    df_result["predicted_sentiment"] = sentiment_labels
    df_result["sentiment_confidence"] = confidences
    df_result["predicted_sentiment_score"] = predictions

    print("\nSample predictions:")
    print(
        df_result[["title", "predicted_sentiment", "sentiment_confidence"]]
        .head(10)
        .to_string(index=False)
    )

    # 6) Add true labels from rating (if available) and evaluate
    df_result = add_true_sentiment_from_rating(df_result)
    evaluate_model(df_result)

    # 7) Save results locally
    df_result.to_csv(LOCAL_RESULTS_CSV, index=False)
    df_result.to_parquet(LOCAL_RESULTS_PARQUET, index=False)
    print(f"\nResults saved locally to '{LOCAL_RESULTS_CSV}' and '{LOCAL_RESULTS_PARQUET}'")

    # 8) Save model locally
    save_model_and_tokenizer(model, tokenizer, LOCAL_SAVED_MODEL_DIR)

    # 9) Upload model & results to GCS
    upload_to_gcs(gcs_client, LOCAL_SAVED_MODEL_DIR, GCS_MODEL_PATH)
    upload_to_gcs(gcs_client, LOCAL_RESULTS_CSV, GCS_RESULTS_CSV_PATH)
    upload_to_gcs(gcs_client, LOCAL_RESULTS_PARQUET, GCS_RESULTS_PARQUET_PATH)

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
