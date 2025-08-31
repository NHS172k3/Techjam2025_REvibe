import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer

def process_video_comments(df, comment_column='top_comments'):
    """
    Process video comments using ONLY the trained sentiment classifier
    PURE SENTIMENT ONLY - NO QUALITY ANALYSIS
    Outputs sentiment scores from 0 to 1 (0=very negative, 0.5=neutral, 1=very positive)
    """
    print("🤖 Loading YOUR trained sentiment model...")
    
    try:
        # Import from the trained comment-sentiment-analyzer
        classifier_path = Path(__file__).parent / "comment-sentiment-analyzer" / "src"
        sys.path.append(str(classifier_path))
        
        from models.classifier import SentimentClassifier
        
        # Load YOUR trained sentiment model
        model = SentimentClassifier()
        model_path = Path(__file__).parent / "comment-sentiment-analyzer" / "models" / "best_sentiment_model.pth"
        
        if not model_path.exists():
            print(f"❌ Your sentiment model not found at {model_path}")
            print("Please train the sentiment model first: cd comment-sentiment-analyzer && python train_classifier.py")
            return create_default_df(df)
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        print("✅ YOUR sentiment model loaded successfully!")
        
        # Process each video's comments
        results = []
        
        for idx, row in df.iterrows():
            video_id = row['video_id']
            comments_raw = str(row.get(comment_column, ''))
            
            # Split comments by delimiter
            if '|' in comments_raw:
                comments = [c.strip() for c in comments_raw.split('|') if c.strip()]
            else:
                comments = [comments_raw] if comments_raw.strip() else []
            
            if not comments:
                results.append({
                    'video_id': video_id,
                    'sentiment_score': 0.5,  # 0-1 scale (neutral)
                    'analyzed_comments_count': 0,
                    'sentiment_interpretation': 'No Comments'
                })
                continue
            
            # Analyze each comment with YOUR model
            comment_scores = []
            
            for comment in comments[:10]:  # Analyze up to 10 comments
                if len(comment.strip()) < 3:
                    continue
                    
                try:
                    # Tokenize comment
                    inputs = tokenizer(
                        comment,
                        max_length=128,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    with torch.no_grad():
                        outputs = model(inputs['input_ids'], inputs['attention_mask'])
                        raw_sentiment = outputs['sentiment'].item()  # -1 to 1
                        
                        # Convert from -1,1 to 0,1
                        sentiment_score = (raw_sentiment + 1) / 2
                        comment_scores.append(sentiment_score)
                        
                except Exception as e:
                    print(f"⚠️ Error processing comment '{comment[:30]}...': {e}")
                    continue
            
            if comment_scores:
                # Calculate video-level sentiment (simple average) - now 0 to 1
                avg_sentiment = np.mean(comment_scores)  # 0 to 1
                
                # Interpretation based on 0-1 sentiment
                interpretation = get_sentiment_interpretation(avg_sentiment)
                
                results.append({
                    'video_id': video_id,
                    'sentiment_score': avg_sentiment,               # 0 to 1
                    'analyzed_comments_count': len(comment_scores),
                    'sentiment_interpretation': interpretation,
                    'sentiment_variance': np.var(comment_scores)
                })
            else:
                results.append({
                    'video_id': video_id,
                    'sentiment_score': 0.5,     # Neutral (0-1 scale)
                    'analyzed_comments_count': 0,
                    'sentiment_interpretation': 'Processing Error'
                })
            
            if idx % 50 == 0 and idx > 0:
                print(f"Processed {idx}/{len(df)} videos...")
        
        # Merge results back to dataframe
        results_df = pd.DataFrame(results)
        df_merged = df.merge(results_df, on='video_id', how='left')
        
        # Fill any missing values
        df_merged['sentiment_score'] = df_merged['sentiment_score'].fillna(0.5)  # Neutral
        df_merged['analyzed_comments_count'] = df_merged['analyzed_comments_count'].fillna(0)
        df_merged['sentiment_interpretation'] = df_merged['sentiment_interpretation'].fillna('No Analysis')
        
        print(f"✅ Processed {len(df)} videos with YOUR trained sentiment model")
        print(f"Sentiment range: {df_merged['sentiment_score'].min():.3f} - {df_merged['sentiment_score'].max():.3f}")
        print(f"Sentiment mean: {df_merged['sentiment_score'].mean():.3f}")
        print(f"Sentiment std: {df_merged['sentiment_score'].std():.3f}")
        
        return df_merged
        
    except Exception as e:
        print(f"❌ Error using YOUR trained model: {e}")
        print("Creating default neutral scores...")
        return create_default_df(df)

def get_sentiment_interpretation(sentiment_score):
    """Get interpretation for sentiment score (0 to 1)"""
    if sentiment_score >= 0.8:
        return "Very Positive"
    elif sentiment_score >= 0.6:
        return "Positive"
    elif sentiment_score >= 0.4:
        return "Neutral"
    elif sentiment_score >= 0.2:
        return "Negative"
    else:
        return "Very Negative"

def create_default_df(df):
    """Create dataframe with default neutral scores if model fails"""
    df_default = df.copy()
    df_default['sentiment_score'] = 0.5      # Neutral (0-1 scale)
    df_default['analyzed_comments_count'] = 0
    df_default['sentiment_interpretation'] = 'Model Not Available'
    return df_default

# Test function
def test_sentiment_model():
    """Test the sentiment model with sample comments"""
    test_comments = [
        "This video is amazing! I love it so much!",
        "This is terrible, worst video ever",
        "It's okay, nothing special",
        "Absolutely fantastic content, really enjoyed it",
        "Boring and poorly made"
    ]
    
    test_df = pd.DataFrame({
        'video_id': [f'test_{i}' for i in range(len(test_comments))],
        'top_comments': test_comments
    })
    
    result = process_video_comments(test_df)
    
    print("\nTest Results (0-1 scale):")
    for _, row in result.iterrows():
        print(f"{row['video_id']}: Sentiment={row['sentiment_score']:.3f} ({row['sentiment_interpretation']})")

if __name__ == "__main__":
    test_sentiment_model()