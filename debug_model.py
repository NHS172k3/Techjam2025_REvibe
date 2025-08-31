import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Add video-quality-classifier to path
sys.path.append(str(Path(__file__).parent / "comment-sentiment-analyzer" / "src"))

from models.classifier import IntegratedQualityModel

def debug_model_outputs():
    """Debug what your trained model is actually outputting"""
    
    print("🔍 Debugging trained model outputs...")
    
    # Load model
    model_path = "comment-sentiment-analyzer/models/best_integrated_model.pth"
    if not Path(model_path).exists():
        print("❌ Model not found!")
        return
    
    model = IntegratedQualityModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Test with diverse comments
    test_comments = [
        # Should be HIGH quality/usefulness
        "This tutorial taught me so much! Amazing step-by-step explanation, very helpful for beginners",
        "Excellent educational content, learned valuable techniques that I can apply immediately",
        "Perfect breakdown of complex concepts, super informative and well structured",
        
        # Should be MEDIUM-HIGH (positive sentiment, some usefulness)
        "Love this content! Great explanation and very engaging presentation style",
        "Really enjoyed watching this, informative and entertaining at the same time",
        
        # Should be MEDIUM (pure positive sentiment, no usefulness)
        "This is so cute! Love the cats, they're absolutely adorable",
        "Amazing video quality! Beautiful cinematography and great editing",
        "So funny, made my day! Really entertaining content",
        
        # Should be LOW-MEDIUM (negative sentiment, but some usefulness)
        "Boring presentation but learned something useful from the content",
        "Not very engaging but the information was helpful",
        
        # Should be LOW (negative sentiment, no usefulness)
        "This is terrible, waste of time and not helpful at all",
        "Don't like this content, boring and pointless",
        "Poor quality, couldn't understand anything"
    ]
    
    print("\n" + "="*80)
    print("MODEL OUTPUT ANALYSIS")
    print("="*80)
    
    results = []
    
    for comment in test_comments:
        # Tokenize
        inputs = tokenizer(
            comment,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            
            unified = outputs['unified_score'].item()
            sentiment = outputs['sentiment'].item()
            usefulness = outputs['usefulness_score'].item()
            
            results.append({
                'comment': comment,
                'unified_score': unified,
                'sentiment': sentiment,
                'usefulness': usefulness
            })
            
            print(f"\nComment: '{comment[:60]}...'")
            print(f"  Unified Score: {unified:.4f}")
            print(f"  Sentiment: {sentiment:.4f} (-1 to 1)")
            print(f"  Usefulness: {usefulness:.4f} (-1 to 1)")
    
    # Analysis
    df_results = pd.DataFrame(results)
    
    print(f"\n" + "="*80)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"Unified Score Range: {df_results['unified_score'].min():.4f} to {df_results['unified_score'].max():.4f}")
    print(f"Unified Score Mean: {df_results['unified_score'].mean():.4f}")
    print(f"Unified Score Std: {df_results['unified_score'].std():.4f}")
    
    print(f"\nSentiment Range: {df_results['sentiment'].min():.4f} to {df_results['sentiment'].max():.4f}")
    print(f"Sentiment Mean: {df_results['sentiment'].mean():.4f}")
    print(f"Sentiment Std: {df_results['sentiment'].std():.4f}")
    
    print(f"\nUsefulness Range: {df_results['usefulness'].min():.4f} to {df_results['usefulness'].max():.4f}")
    print(f"Usefulness Mean: {df_results['usefulness'].mean():.4f}")
    print(f"Usefulness Std: {df_results['usefulness'].std():.4f}")
    
    # Check if model is actually learning
    educational_comments = df_results[df_results['comment'].str.contains('tutorial|learn|teach|helpful|informative', case=False)]
    entertainment_comments = df_results[df_results['comment'].str.contains('cute|funny|love|adorable', case=False)]
    
    if len(educational_comments) > 0 and len(entertainment_comments) > 0:
        edu_mean = educational_comments['unified_score'].mean()
        ent_mean = entertainment_comments['unified_score'].mean()
        
        print(f"\n📚 Educational comments avg score: {edu_mean:.4f}")
        print(f"🎬 Entertainment comments avg score: {ent_mean:.4f}")
        print(f"📊 Difference: {abs(edu_mean - ent_mean):.4f}")
        
        if abs(edu_mean - ent_mean) < 0.05:
            print("⚠️  WARNING: Model shows very little discrimination between content types!")
        else:
            print("✅ Model shows reasonable discrimination")
    
    return df_results

def check_training_data_distribution():
    """Check what your model was actually trained on"""
    
    print("\n🔍 Checking training data distribution...")
    
    try:
        train_df = pd.read_csv("comment-sentiment-analyzer/data/processed/train.csv")
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")
        
        # Check score distributions
        print(f"\nSentiment score distribution:")
        print(f"  Range: {train_df['sentiment_score'].min():.3f} to {train_df['sentiment_score'].max():.3f}")
        print(f"  Mean: {train_df['sentiment_score'].mean():.3f}")
        print(f"  Std: {train_df['sentiment_score'].std():.3f}")
        
        print(f"\nUsefulness score distribution:")
        print(f"  Range: {train_df['usefulness_score'].min():.3f} to {train_df['usefulness_score'].max():.3f}")
        print(f"  Mean: {train_df['usefulness_score'].mean():.3f}")
        print(f"  Std: {train_df['usefulness_score'].std():.3f}")
        
        print(f"\nUsefulness mentioned distribution:")
        print(f"  0: {(train_df['usefulness_mentioned'] == 0).sum()}")
        print(f"  1: {(train_df['usefulness_mentioned'] == 1).sum()}")
        
        # Sample some data
        print(f"\nSample training data:")
        sample = train_df.sample(5) if len(train_df) >= 5 else train_df
        for _, row in sample.iterrows():
            print(f"  '{row['comment'][:50]}...' -> S:{row['sentiment_score']:.2f}, U:{row['usefulness_score']:.2f}, M:{row['usefulness_mentioned']}")
            
    except Exception as e:
        print(f"❌ Error loading training data: {e}")

if __name__ == "__main__":
    check_training_data_distribution()
    debug_results = debug_model_outputs()