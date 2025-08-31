import pandas as pd
import numpy as np
from pathlib import Path
import re

def clean_comment(comment):
    """Clean comment text"""
    if pd.isna(comment):
        return ""
    
    comment = str(comment)
    
    # Remove extra whitespace
    comment = ' '.join(comment.split())
    
    # Remove very short comments
    if len(comment) < 3:
        return ""
    
    return comment

def analyze_sentiment(comment):
    """Simple sentiment analysis - you can enhance this"""
    if not comment:
        return 0.0
    
    comment_lower = comment.lower()
    
    # Positive words
    positive_words = [
        'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'good', 'love', 
        'like', 'wonderful', 'beautiful', 'perfect', 'brilliant', 'incredible',
        'outstanding', 'superb', 'marvelous', 'fabulous', 'terrific', 'cool',
        'nice', 'sweet', 'cute', 'adorable', 'lovely', 'charming', 'delightful'
    ]
    
    # Negative words
    negative_words = [
        'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'dislike',
        'boring', 'stupid', 'dumb', 'annoying', 'irritating', 'disappointing',
        'useless', 'worthless', 'pathetic', 'disgusting', 'revolting', 'gross',
        'ugly', 'nasty', 'rude', 'mean', 'cruel', 'harsh'
    ]
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in comment_lower)
    negative_count = sum(1 for word in negative_words if word in comment_lower)
    
    # Calculate sentiment score (-1 to 1)
    total_words = len(comment_lower.split())
    if total_words == 0:
        return 0.0
    
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    
    # Normalize to -1 to 1 range
    sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))
    
    return sentiment_score

def create_synthetic_data(base_comments, n_samples=2000):
    """Create synthetic sentiment training data"""
    
    # Positive sentiment templates
    positive_templates = [
        "This is amazing! I love it so much.",
        "Absolutely fantastic content, really enjoyed watching this.",
        "Wonderful video! Great job on this.",
        "Excellent work, this made my day better.",
        "Beautiful and inspiring, thank you for sharing.",
        "Perfect! Exactly what I was looking for.",
        "Outstanding quality, keep up the great work.",
        "This is brilliant! Love the creativity.",
        "Superb content, really well done.",
        "Awesome video! This is so good."
    ]
    
    # Negative sentiment templates
    negative_templates = [
        "This is terrible, I hate it.",
        "Worst video ever, completely boring.",
        "Awful content, waste of time.",
        "Horrible quality, very disappointing.",
        "Bad video, not worth watching.",
        "Disgusting and annoying content.",
        "Pathetic attempt, really bad.",
        "Stupid video, makes no sense.",
        "Terrible quality, poorly made.",
        "Useless content, very disappointing."
    ]
    
    # Neutral sentiment templates
    neutral_templates = [
        "This is okay, nothing special.",
        "Average video, seen better.",
        "It's fine, could be improved.",
        "Decent content, not bad.",
        "Standard video, nothing new.",
        "Alright, but not amazing.",
        "It's watchable, I guess.",
        "Normal content, pretty typical.",
        "Okay video, nothing exciting.",
        "Fair enough, could be worse."
    ]
    
    synthetic_data = []
    
    # Generate samples
    samples_per_category = n_samples // 3
    
    # Positive samples
    for i in range(samples_per_category):
        template = np.random.choice(positive_templates)
        sentiment_score = np.random.uniform(0.3, 1.0)  # Positive range
        synthetic_data.append({
            'comment': template,
            'sentiment_score': sentiment_score
        })
    
    # Negative samples  
    for i in range(samples_per_category):
        template = np.random.choice(negative_templates)
        sentiment_score = np.random.uniform(-1.0, -0.3)  # Negative range
        synthetic_data.append({
            'comment': template,
            'sentiment_score': sentiment_score
        })
    
    # Neutral samples
    for i in range(samples_per_category):
        template = np.random.choice(neutral_templates)
        sentiment_score = np.random.uniform(-0.3, 0.3)  # Neutral range
        synthetic_data.append({
            'comment': template,
            'sentiment_score': sentiment_score
        })
    
    # Add some real comments if available
    if len(base_comments) > 0:
        for comment in base_comments[:min(200, len(base_comments))]:
            cleaned = clean_comment(comment)
            if cleaned:
                sentiment = analyze_sentiment(cleaned)
                synthetic_data.append({
                    'comment': cleaned,
                    'sentiment_score': sentiment
                })
    
    return pd.DataFrame(synthetic_data)

def preprocess_dataset():
    """Preprocess dataset for sentiment analysis only"""
    
    print("🔄 Preprocessing dataset for sentiment analysis...")
    
    # Create directories
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    raw_dir = data_dir / "raw"
    
    processed_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    
    # Try to load existing data
    base_comments = []
    
    # Check for existing comment data
    existing_files = [
        raw_dir / "comments_dataset.csv",
        processed_dir / "labeled_comments.csv",
        processed_dir / "preprocessed_comments.csv"
    ]
    
    for file_path in existing_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if 'comment' in df.columns:
                    base_comments.extend(df['comment'].dropna().tolist())
                    print(f"Loaded {len(df)} comments from {file_path.name}")
                elif 'top_comments' in df.columns:
                    # Handle pipe-separated comments
                    for comments_str in df['top_comments'].dropna():
                        if '|' in str(comments_str):
                            comments = [c.strip() for c in str(comments_str).split('|')]
                            base_comments.extend(comments)
                        else:
                            base_comments.append(str(comments_str))
                break
            except Exception as e:
                print(f"Could not load {file_path}: {e}")
                continue
    
    if not base_comments:
        print("No existing comments found, creating synthetic dataset...")
    
    # Clean base comments
    base_comments = [clean_comment(c) for c in base_comments if clean_comment(c)]
    print(f"Found {len(base_comments)} valid base comments")
    
    # Create synthetic dataset
    df = create_synthetic_data(base_comments, n_samples=2000)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created dataset with {len(df)} samples")
    print(f"Sentiment score range: {df['sentiment_score'].min():.2f} to {df['sentiment_score'].max():.2f}")
    
    # Split into train/test
    train_size = int(0.8 * len(df))
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()
    
    # Save processed data
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    print(f"Created train set: {len(train_df)} samples")
    print(f"Created test set: {len(test_df)} samples")
    
    print(f"\nDataset Statistics:")
    print(f"Sentiment score range: {df['sentiment_score'].min():.2f} to {df['sentiment_score'].max():.2f}")
    print(f"Positive sentiment (>0.1): {(df['sentiment_score'] > 0.1).sum()}/{len(df)} ({(df['sentiment_score'] > 0.1).mean()*100:.1f}%)")
    print(f"Negative sentiment (<-0.1): {(df['sentiment_score'] < -0.1).sum()}/{len(df)} ({(df['sentiment_score'] < -0.1).mean()*100:.1f}%)")
    print(f"Neutral sentiment (-0.1 to 0.1): {((df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)).sum()}/{len(df)} ({((df['sentiment_score'] >= -0.1) & (df['sentiment_score'] <= 0.1)).mean()*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    preprocess_dataset()