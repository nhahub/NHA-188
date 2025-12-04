import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from io import StringIO
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Customer Sentiment Analysis Dashboard")
st.markdown("""
This app analyzes customer reviews and feedback using AI-powered sentiment analysis.
Upload your data or enter text manually to get insights about customer satisfaction.
""")

# Initialize session state for model
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
# def load_sentiment_model():
#     """Load the sentiment analysis model"""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
#         model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")
        
#         # Set device
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         model.eval()
        
#         return tokenizer, model, device
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None, None

def load_sentiment_model():
    """Load model from local files"""
    try:
        # Load from your saved model directory
        tokenizer = AutoTokenizer.from_pretrained("./saved_sentiment_model")
        model = AutoModelForSequenceClassification.from_pretrained("./saved_sentiment_model")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loadig model: {e}")
        return None, None, None

def predict_sentiment(texts, tokenizer, model, device, batch_size=8):
    """Predict sentiment for a list of texts"""
    predictions = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            batch_confidences = torch.max(probabilities, dim=-1).values
        
        predictions.extend(batch_predictions.cpu().numpy())
        confidences.extend(batch_confidences.cpu().numpy())
    
    return predictions, confidences

def map_predictions_to_labels(predictions):
    """Map numerical predictions to sentiment labels"""
    label_map = {
        0: "Very Negative", 
        1: "Negative", 
        2: "Neutral", 
        3: "Positive", 
        4: "Very Positive"
    }
    return [label_map[pred] for pred in predictions]

def create_visualizations(df):
    """Create visualization charts"""
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        st.subheader("📈 Sentiment Distribution")
        sentiment_counts = df['predicted_sentiment'].value_counts()
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        colors = ['#ff6b6b', '#ffa726', '#ffee58', '#66bb6a', '#2e7d32']
        wedges, texts, autotexts = ax1.pie(
            sentiment_counts.values, 
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(sentiment_counts)],
            startangle=90
        )
        
        # Improve aesthetics
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.axis('equal')
        st.pyplot(fig1)
    
    with col2:
        # Confidence distribution
        st.subheader("🎯 Confidence Scores")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.histplot(df['sentiment_confidence'], bins=20, kde=True, ax=ax2)
        ax2.set_xlabel('Confidence Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Prediction Confidence')
        st.pyplot(fig2)
    
    # Sentiment by rating (if rating column exists)
    if 'rating' in df.columns:
        st.subheader("⭐ Sentiment vs Rating")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # Create a pivot table
        pivot_data = pd.crosstab(df['rating'], df['predicted_sentiment'])
        pivot_data.plot(kind='bar', ax=ax3, figsize=(10, 6))
        
        ax3.set_xlabel('Rating')
        ax3.set_ylabel('Count')
        ax3.set_title('Sentiment Distribution by Rating')
        ax3.legend(title='Sentiment')
        ax3.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig3)

def display_metrics(df):
    """Display key metrics"""
    st.subheader("📊 Key Metrics")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_reviews = len(df)
    positive_reviews = len(df[df['predicted_sentiment'].isin(['Positive', 'Very Positive'])])
    negative_reviews = len(df[df['predicted_sentiment'].isin(['Negative', 'Very Negative'])])
    avg_confidence = df['sentiment_confidence'].mean()
    
    with col1:
        st.metric("Total Reviews", f"{total_reviews:,}")
    
    with col2:
        st.metric("Positive Reviews", f"{positive_reviews:,}", 
                 f"{(positive_reviews/total_reviews*100):.1f}%")
    
    with col3:
        st.metric("Negative Reviews", f"{negative_reviews:,}", 
                 f"{(negative_reviews/total_reviews*100):.1f}%")
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or Parquet file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Analysis Mode:", 
                           ["Single Text Analysis", "Batch File Analysis"])

# Load model (only once)
if not st.session_state.model_loaded:
    with st.spinner("Loading sentiment analysis model... This may take a minute."):
        tokenizer, model, device = load_sentiment_model()
        if tokenizer and model:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.sidebar.success("Model loaded successfully!")
        else:
            st.sidebar.error("Failed to load model")

# Main application logic
if st.session_state.model_loaded:
    if app_mode == "Single Text Analysis":
        st.header("Single Text Analysis")
        
        # Text input
        user_text = st.text_area(
            "Enter your customer review text:",
            height=150,
            placeholder="Type your review here... Example: 'I love this product! The quality is amazing and delivery was fast.'"
        )
        
        # Optional rating
        rating = st.slider("Rating (if available)", 1, 5, 3)
        
        if st.button("Analyze Sentiment") and user_text:
            with st.spinner("Analyzing sentiment..."):
                # Predict sentiment
                predictions, confidences = predict_sentiment(
                    [user_text], 
                    st.session_state.tokenizer, 
                    st.session_state.model, 
                    st.session_state.device
                )
                
                sentiment_labels = map_predictions_to_labels(predictions)
                confidence = confidences[0]
                
                # Display results
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment with color coding
                    sentiment_color = {
                        "Very Positive": "green",
                        "Positive": "lightgreen", 
                        "Neutral": "orange",
                        "Negative": "lightcoral",
                        "Very Negative": "red"
                    }
                    
                    st.markdown(f"""
                    **Sentiment:** <span style='color:{sentiment_color[sentiment_labels[0]]}; 
                    font-size:20px; font-weight:bold'>{sentiment_labels[0]}</span>
                    
                    **Confidence:** {confidence:.2%}
                    
                    **Rating:** {rating}/5 ⭐
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence gauge
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.barh([0], [confidence], color=sentiment_color[sentiment_labels[0]], height=0.5)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Confidence')
                    ax.set_title('Prediction Confidence')
                    st.pyplot(fig)
    
    else:  # Batch File Analysis
        st.header("📁 Batch File Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your customer reviews file",
            type=['csv', 'xlsx', 'parquet'],
            help="Supported formats: CSV, Excel (.xlsx)"
        )
        
        if uploaded_file is not None:
            # Process uploaded file
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File loaded successfully! {len(df)} records found.")
                
                # Show data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Check for required columns
                text_columns = [col for col in df.columns if any(keyword in col.lower() 
                               for keyword in ['text', 'review', 'comment', 'description', 'content'])]
                
                if not text_columns:
                    st.error("No text column found. Please ensure your file has a column containing review text.")
                    st.info("Common column names: 'text', 'review', 'comment', 'description'")
                else:
                    # Let user select text column
                    text_column = st.selectbox("Select the column containing review text:", text_columns)
                    
                    if st.button("🚀 Analyze All Reviews"):
                        with st.spinner("Analyzing all reviews... This may take a while for large files."):
                            # Get texts
                            texts = df[text_column].fillna('').astype(str).tolist()
                            
                            # Predict sentiments in batches
                            predictions, confidences = predict_sentiment(
                                texts, 
                                st.session_state.tokenizer, 
                                st.session_state.model, 
                                st.session_state.device,
                                batch_size=16
                            )
                            
                            # Add results to dataframe
                            df_result = df.copy()
                            df_result['predicted_sentiment'] = map_predictions_to_labels(predictions)
                            df_result['sentiment_confidence'] = confidences
                            df_result['predicted_sentiment_score'] = predictions
                            
                            # Store in session state
                            st.session_state.df_result = df_result
                            
                            # Display results
                            st.success("✅ Analysis completed!")
                            
                            # Show metrics
                            display_metrics(df_result)
                            
                            # Show visualizations
                            create_visualizations(df_result)
                            
                            # Show sample results
                            st.subheader("🔍 Sample Results")
                            st.dataframe(
                                df_result[['predicted_sentiment', 'sentiment_confidence', text_column]].head(10),
                                use_container_width=True
                            )
                            
                            # Download results
                            st.subheader("📥 Download Results")
                            
                            # Convert to different formats
                            csv = df_result.to_csv(index=False)
                            excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                            df_result.to_excel(excel_buffer.name, index=False)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.download_button(
                                    label="📄 Download as CSV",
                                    data=csv,
                                    file_name="sentiment_analysis_results.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                with open(excel_buffer.name, 'rb') as f:
                                    st.download_button(
                                        label="📊 Download as Excel",
                                        data=f,
                                        file_name="sentiment_analysis_results.xlsx",
                                        mime="application/vnd.ms-excel"
                                    )
                            
                            # Clean up
                            #  os.unlink(excel_buffer.name)

else:
    st.error("Model failed to load. Please check your internet connection and try again.")

# Footer
st.markdown("---")
