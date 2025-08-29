import pandas as pd
import re
import numpy as np

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(input_path):
    """
    Preprocess the labeled dataset
    
    Args:
        input_path: Path to labeled dataset CSV
        
    Returns:
        df: Preprocessed DataFrame
    """
    # Load labeled dataset
    df = pd.read_csv(input_path)
    
    # Clean comments
    df['cleaned_comment'] = df['comment'].apply(clean_text)
    
    # Remove rows with empty comments after cleaning
    df = df[df['cleaned_comment'].str.len() > 0]
    
    # Handle missing values
    df['sentiment_score'] = df['sentiment_score'].fillna(0.5)
    df['unified_score'] = df['unified_score'].fillna(0.5)
    df['quality_label'] = df['quality_label'].fillna(1)  # Default to "not about quality"
    
    # Ensure data types are correct
    df['quality_label'] = df['quality_label'].astype(int)
    df['sentiment_score'] = df['sentiment_score'].astype(float)
    df['unified_score'] = df['unified_score'].astype(float)
    
    print(f"Preprocessed dataset: {len(df)} samples")
    print(f"Quality distribution:")
    print(df['quality_label'].value_counts().sort_index())
    print(f"Sentiment score range: {df['sentiment_score'].min():.2f} - {df['sentiment_score'].max():.2f}")
    print(f"Unified score range: {df['unified_score'].min():.2f} - {df['unified_score'].max():.2f}")
    
    return df

if __name__ == "__main__":
    # Preprocess the quality-labeled dataset
    input_path = '../../data/processed/quality_labeled_comments.csv'
    output_path = '../../data/processed/preprocessed_comments.csv'
    preprocess_dataset(input_path, output_path)