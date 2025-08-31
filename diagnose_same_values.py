import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def diagnose_pipeline():
    """Comprehensive diagnosis of why values are the same"""
    
    print("🔍 DIAGNOSING PIPELINE ISSUES...")
    print("="*60)
    
    # 1. Check if sentiment model exists
    print("\n1️⃣ CHECKING SENTIMENT MODEL...")
    model_path = Path("video-quality-classifier/models/best_sentiment_model.pth")
    if model_path.exists():
        print("✅ Sentiment model found")
    else:
        print("❌ Sentiment model NOT found - this will cause all sentiment scores to be 0.5")
        print(f"   Expected path: {model_path.absolute()}")
        print("   Solution: Train the sentiment model or use fallback")
    
    # 2. Check input data
    print("\n2️⃣ CHECKING INPUT DATA...")
    try:
        # Check main dataset
        if os.path.exists('full_video_dataset_with_engagement.csv'):
            df = pd.read_csv('full_video_dataset_with_engagement.csv')
            print(f"✅ Main dataset found: {len(df)} videos")
            
            # Check data variety
            print("   Data variety check:")
            for col in ['views', 'likes', 'comment_count', 'shares']:
                if col in df.columns:
                    unique_vals = df[col].nunique()
                    print(f"     {col}: {unique_vals} unique values (range: {df[col].min()}-{df[col].max()})")
                    if unique_vals == 1:
                        print(f"     ⚠️ {col} has only one unique value - this will cause identical scores!")
        else:
            print("❌ Main dataset not found")
            
        # Check JSON input
        if os.path.exists('to_insert.json'):
            with open('to_insert.json', 'r') as f:
                json_data = json.load(f)
            print(f"✅ JSON input found: {json_data.get('video_id', 'N/A')}")
        else:
            print("ℹ️ No JSON input (app.py data)")
            
    except Exception as e:
        print(f"❌ Error checking input data: {e}")
    
    # 3. Test sentiment analysis
    print("\n3️⃣ TESTING SENTIMENT ANALYSIS...")
    try:
        from analyze_sentiment_quality import process_video_comments
        
        # Create test data with DIFFERENT comments
        test_df = pd.DataFrame({
            'video_id': ['test1', 'test2', 'test3'],
            'top_comments': [
                'This is absolutely amazing! I love it!',  # Should be high
                'This is terrible, worst video ever',      # Should be low
                'It is okay, nothing special'              # Should be medium
            ]
        })
        
        result = process_video_comments(test_df)
        
        print("   Test sentiment results:")
        for _, row in result.iterrows():
            print(f"     {row['video_id']}: {row['sentiment_score']:.3f}")
        
        # Check if all values are the same
        sentiment_values = result['sentiment_score'].values
        if len(set(sentiment_values)) == 1:
            print("   ❌ All sentiment scores are identical!")
        else:
            print("   ✅ Sentiment scores vary correctly")
            
    except Exception as e:
        print(f"   ❌ Sentiment analysis failed: {e}")
    
    # 4. Test social good analysis
    print("\n4️⃣ TESTING SOCIAL GOOD ANALYSIS...")
    try:
        from socialgood_quality import score_file
        
        # Create test data with DIFFERENT content
        test_df = pd.DataFrame({
            'video_id': ['social1', 'social2', 'social3'],
            'subtitles': [
                'This tutorial teaches sustainable energy practices and renewable solar power installation',  # High
                'Just a random dance video with no educational content',  # Low
                'This video shows some basic cooking techniques'  # Medium
            ]
        })
        
        result = score_file(test_df, text_col='subtitles')
        
        print("   Test social good results:")
        for _, row in result.iterrows():
            print(f"     {row['video_id']}: total_score={row['total_score']:.3f}, multiplier={row['social_value_multiplier']:.3f}")
        
        # Check if all values are the same
        social_values = result['total_score'].values
        if len(set(np.round(social_values, 3))) == 1:
            print("   ❌ All social good scores are identical!")
        else:
            print("   ✅ Social good scores vary correctly")
            
    except Exception as e:
        print(f"   ❌ Social good analysis failed: {e}")
    
    # 5. Test clustering
    print("\n5️⃣ TESTING CLUSTERING...")
    try:
        from ST_tokenize import sort_category
        
        # Create test data with DIFFERENT content
        test_df = pd.DataFrame({
            'video_id': ['cluster1', 'cluster2', 'cluster3', 'cluster4'],
            'subtitles': [
                'Educational tutorial about mathematics and science',
                'Funny dance challenge video with music',
                'Cooking recipe for healthy sustainable meals',
                'Comedy skit about everyday life situations'
            ],
            'views': [1000, 2000, 1500, 3000],
            'likes': [100, 200, 150, 300]
        })
        
        result = sort_category(test_df)
        
        print("   Test clustering results:")
        cluster_counts = result['cluster'].value_counts()
        print(f"     Number of clusters: {len(cluster_counts)}")
        for cluster, count in cluster_counts.items():
            print(f"     Cluster {cluster}: {count} videos")
        
        if len(cluster_counts) == 1:
            print("   ❌ All videos assigned to same cluster!")
        else:
            print("   ✅ Videos distributed across clusters")
            
    except Exception as e:
        print(f"   ❌ Clustering failed: {e}")
    
    # 6. Test normalization
    print("\n6️⃣ TESTING NORMALIZATION...")
    try:
        from normalize import normalize_and_score
        
        # Create test data with DIFFERENT values
        test_df = pd.DataFrame({
            'video_id': ['norm1', 'norm2', 'norm3'],
            'cluster': [0, 0, 0],  # Same cluster for testing
            'views': [1000, 5000, 10000],     # Different values
            'likes': [100, 500, 1000],        # Different values
            'comment_count': [50, 200, 400],  # Different values
            'shares': [10, 50, 100],          # Different values
            'saved_count': [5, 25, 50],       # Different values
            'viewer_retention_percentage': [60, 75, 90],  # Different values
            'sentiment_score': [0.3, 0.7, 0.9],  # Different values
            'social_value_multiplier': [0.9, 1.0, 1.1],  # Different values
            'content_flag': [True, True, True]
        })
        
        result = normalize_and_score(test_df)
        
        print("   Test normalization results:")
        for _, row in result.iterrows():
            print(f"     {row['video_id']}: video_score={row.get('video_score', 0):.3f}, video_money=${row.get('video_money', 0):.2f}")
        
        # Check if all values are the same
        if 'video_money' in result.columns:
            money_values = result['video_money'].values
            if len(set(np.round(money_values, 2))) == 1:
                print("   ❌ All video earnings are identical!")
            else:
                print("   ✅ Video earnings vary correctly")
        
    except Exception as e:
        print(f"   ❌ Normalization failed: {e}")
    
    print("\n" + "="*60)
    print("🎯 DIAGNOSIS COMPLETE!")
    print("\nMost likely causes of identical values:")
    print("1. Sentiment model not found - using default 0.5 for all")
    print("2. Input data has no variety (all same values)")
    print("3. All videos assigned to same cluster")
    print("4. Social good analysis producing similar scores")
    print("5. Normalization issues with identical inputs")

def create_test_data_with_variety():
    """Create test data with clear variety to test the pipeline"""
    
    print("\n🧪 CREATING TEST DATA WITH VARIETY...")
    
    test_data = {
        "video_id": "test_variety_001",
        "user_id": "test_user_001", 
        "views": 50000,  # High views
        "likes": 5000,   # High likes
        "comment_count": 500,  # High comments
        "shares": 250,   # High shares
        "saved_count": 150,  # High saves
        "viewer_retention_percentage": 85.0,  # High retention
        "captions": "Educational content about sustainable living!",
        "subtitles": "This comprehensive tutorial teaches viewers about renewable energy, solar panel installation, and sustainable living practices. Learn step by step how to reduce your carbon footprint.",  # High social value
        "top_comments": "This is incredibly helpful! | Amazing educational content | Learned so much | Best tutorial ever | Thank you for teaching this!",  # Positive sentiment
        "video_file": ""
    }
    
    # Save test data
    with open("to_insert.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Test data created with:")
    print(f"   High engagement metrics")
    print(f"   Educational/environmental content (high social value)")
    print(f"   Positive comments (high sentiment)")
    
    return test_data

if __name__ == "__main__":
    diagnose_pipeline()
    
    print("\n" + "="*60)
    choice = input("\n🔧 Create test data with variety? (y/n): ").lower()
    if choice == 'y':
        create_test_data_with_variety()
        print("\n✅ Test data created! Now run: python main.py")