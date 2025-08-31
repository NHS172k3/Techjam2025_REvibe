import pandas as pd
import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path
from ST_tokenize import sort_category  
from socialgood_quality import score_file
from normalize import normalize_and_score, allocate_cluster_funds, filter_content, apply_tiktok_specific_rules
from Video_description_generator import generate_video_description
from analyze_sentiment_quality import process_video_comments  # ONLY this import for sentiment

def process_uploaded_video(video_file_name=None):
    """Process uploaded video file if it exists"""
    if not video_file_name:
        return None
    
    video_path = os.path.join(os.getcwd(), video_file_name)
    
    if not os.path.exists(video_path):
        print(f"❌ Uploaded video file '{video_file_name}' not found!")
        return None
    
    print(f"🎥 Processing uploaded video: {video_file_name}")
    
    # Generate description using Gemini if video exists
    print("🤖 Generating video description with Gemini AI...")
    try:
        description = generate_video_description(video_path)
        print("✅ Video description generated successfully")
        return description
    except Exception as e:
        print(f"❌ Error generating description: {e}")
        return "A video uploaded by user."  # Fallback

def process_hardcoded_video():
    """Process the hardcoded cat video (fallback)"""
    video_file = "#catloverlife.mp4"
    
    if not os.path.exists(video_file):
        print(f"❌ Hardcoded video file '{video_file}' not found!")
        return None
    
    print(f"🎥 Processing hardcoded video: {video_file}")
    
    # Generate description using Gemini
    print("🤖 Generating video description with Gemini AI...")
    try:
        description = generate_video_description(video_file)
        print("✅ Video description generated successfully")
    except Exception as e:
        print(f"❌ Error generating description: {e}")
        description = "A video featuring cats and cat-related content."  # Fallback
    
    # Create record for the hardcoded video
    new_video_data = {
        'video_id': video_file.replace('.mp4', ''),
        'user_id': 'new_user_001',
        'views': 10000,
        'likes': 1200,
        'comment_count': 150,
        'shares': 80,
        'saved_count': 45,
        'viewer_retention_percentage': 75.5,
        'subtitles': description,
        'top_comments': 'This is so cute! | Love the cats | Amazing video quality | So adorable | Great content',
        'caption': 'Cat lovers unite! #cats #pets #cute'
    }
    
    return pd.DataFrame([new_video_data])

def run_analysis():
    """Main analysis function that can be called from app.py or run standalone"""
    print("🚀 Starting TikTok Profit Sharing with SENTIMENT + SOCIAL GOOD ONLY...")

    # Load existing data
    print("📂 Loading existing dataset...")
    try:
        df = pd.read_csv('full_video_dataset_with_engagement.csv')
        print(f"Loaded {len(df)} existing videos")
    except FileNotFoundError:
        print("⚠️ Main dataset not found, creating new one...")
        df = pd.DataFrame()
    
    # Load additional data from JSON if exists (from app.py)
    json_file = "to_insert.json"
    uploaded_video_filename = None
    
    if os.path.exists(json_file):
        print("📂 Loading additional data from JSON (from app.py)...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        json_df = pd.DataFrame(data)
        
        # Get uploaded video filename if available
        if 'video_file' in json_df.columns:
            uploaded_video_filename = json_df['video_file'].iloc[0] if not json_df['video_file'].iloc[0] == "" else None
        
        # Process uploaded video description if video file exists
        if uploaded_video_filename:
            generated_description = process_uploaded_video(uploaded_video_filename)
            if generated_description:
                # Update subtitles with generated description
                json_df.loc[0, 'subtitles'] = generated_description
        
        df = pd.concat([df, json_df], ignore_index=True)
        print(f"Added {len(json_df)} videos from JSON (app.py input)")
    
    # If no JSON data, try to process hardcoded video (for standalone runs)
    elif not os.path.exists(json_file):
        print("📂 No JSON data found, trying hardcoded video...")
        hardcoded_video_df = process_hardcoded_video()
        if hardcoded_video_df is not None:
            df = pd.concat([df, hardcoded_video_df], ignore_index=True)
            print(f"✅ Added hardcoded video. Total videos: {len(df)}")
    
    # Ensure required columns exist
    required_columns = ['video_id', 'user_id', 'views', 'likes', 'comment_count', 'shares', 'saved_count', 'viewer_retention_percentage']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0  # Default values
    
    if 'subtitles' not in df.columns:
        df['subtitles'] = ''
    
    if 'top_comments' not in df.columns:
        df['top_comments'] = ''
    
    # Cluster videos by content similarity
    print("🔍 Clustering videos by content similarity...")
    try:
        df_clustered = sort_category(df)
        print(f"Created {df_clustered['cluster'].nunique()} clusters")
    except Exception as e:
        print(f"⚠️ Error in clustering: {e}")
        df['cluster'] = 0  # Single cluster fallback
        df_clustered = df
    
    # Apply ONLY your trained sentiment model (0-1 scale)
    print("🎯 Analyzing comments with YOUR sentiment model...")
    df_analyzed_1 = process_video_comments(df_clustered, comment_column='top_comments')
    
    # Check sentiment results (now 0-1 scale)
    if 'sentiment_score' in df_analyzed_1.columns:
        sentiment_scores = df_analyzed_1['sentiment_score']  # 0 to 1
        
        print(f"✅ Sentiment analysis complete!")
        print(f"   Sentiment range: {sentiment_scores.min():.3f} - {sentiment_scores.max():.3f}")
        print(f"   Sentiment mean: {sentiment_scores.mean():.3f}")
        print(f"   Sentiment std: {sentiment_scores.std():.3f}")
        print(f"   Positive sentiment videos (>0.6): {(sentiment_scores > 0.6).sum()}")
        print(f"   Negative sentiment videos (<0.4): {(sentiment_scores < 0.4).sum()}")
        print(f"   Neutral sentiment videos (0.4-0.6): {((sentiment_scores >= 0.4) & (sentiment_scores <= 0.6)).sum()}")
    else:
        print("⚠️ Sentiment analysis returned default values")

    # Apply social good analysis
    print("🌍 Analyzing videos for social good content...")
    try:
        df_analyzed_2 = score_file(df_analyzed_1, text_col="subtitles")
        social_scores = df_analyzed_2['total_score']
        print(f"✅ Social good analysis complete!")
        print(f"   Social score range: {social_scores.min():.3f} - {social_scores.max():.3f}")
        print(f"   Social score mean: {social_scores.mean():.3f}")
        print(f"   Social multiplier range: {df_analyzed_2['social_value_multiplier'].min():.3f} - {df_analyzed_2['social_value_multiplier'].max():.3f}")
    except Exception as e:
        print(f"⚠️ Error in social good analysis: {e}")
        # Add default social good scores
        df_analyzed_2 = df_analyzed_1.copy()
        df_analyzed_2['social_value_multiplier'] = 1.0
        df_analyzed_2['total_score'] = 0.5
    
    # Create SIMPLE combined multiplier - sentiment + social good ONLY (0.9 to 1.1 range)
    print("🔧 Creating sentiment + social good multiplier...")
    sentiment_weight = 0.7  # Heavy weight on sentiment
    social_weight = 0.3     # Light weight on social good
    
    # Convert sentiment (0-1) to multiplier component (0.9-1.1)
    sentiment_multiplier = 0.9 + 0.2 * df_analyzed_2['sentiment_score']  # 0.9 to 1.1
    social_multiplier = df_analyzed_2.get('social_value_multiplier', 1.0)  # Already 0.9 to 1.1
    
    # Weighted combination to keep in 0.9-1.1 range
    combined_multiplier = (
        sentiment_weight * sentiment_multiplier + 
        social_weight * social_multiplier
    )
    
    # Ensure final range is 0.9 to 1.1
    df_analyzed_2['combined_multiplier'] = np.clip(combined_multiplier, 0.9, 1.1)
    
    # For app.py compatibility - create the expected column names
    df_analyzed_2['quality_sentiment_multiplier'] = df_analyzed_2['combined_multiplier']
    
    print(f"   Sentiment multiplier range: {sentiment_multiplier.min():.3f} - {sentiment_multiplier.max():.3f}")
    print(f"   Social multiplier range: {social_multiplier.min():.3f} - {social_multiplier.max():.3f}")
    print(f"   Combined multiplier range: {df_analyzed_2['combined_multiplier'].min():.3f} - {df_analyzed_2['combined_multiplier'].max():.3f}")
    print(f"   Videos with high multiplier (>1.05): {(df_analyzed_2['combined_multiplier'] > 1.05).sum()}")
    print(f"   Videos with low multiplier (<0.95): {(df_analyzed_2['combined_multiplier'] < 0.95).sum()}")
    
    # Filter for good content (based on sentiment + engagement)
    print("🏆 Filtering for good content...")
    try:
        threshold_multiplier = 0.8  # Lower threshold to include more videos
        df_filtered = filter_content(df_analyzed_2, threshold_multiplier=threshold_multiplier)
        
        good_content = df_filtered['content_flag'].sum()
        if good_content == 0:
            print("⚠️ No videos passed content filter, lowering threshold...")
            df_filtered = filter_content(df_analyzed_2, threshold_multiplier=0.5)
            good_content = df_filtered['content_flag'].sum()
        
        print(f"   Good content videos: {good_content}/{len(df_analyzed_2)} ({good_content/len(df_analyzed_2)*100:.1f}%)")
        
        # Keep all videos but flag the good ones
        df_filtered = df_analyzed_2.copy()
        if 'content_flag' not in df_filtered.columns:
            df_filtered['content_flag'] = True
        
    except Exception as e:
        print(f"⚠️ Error in content filtering: {e}")
        df_filtered = df_analyzed_2.copy()
        df_filtered['content_flag'] = True  # All pass filter
    
    # Enhanced normalization and scoring
    print("📊 Computing scores and normalizing metrics...")
    try:
        df_scored = normalize_and_score(df_filtered, use_enhanced=True)
    except Exception as e:
        print(f"⚠️ Error in normalization: {e}")
        # Simple fallback scoring
        engagement_score = (
            df_filtered['likes'] * 0.25 + 
            df_filtered['comment_count'] * 0.25 + 
            df_filtered['shares'] * 0.25 + 
            df_filtered['saved_count'] * 0.25
        )
        if engagement_score.max() > engagement_score.min():
            engagement_normalized = (engagement_score - engagement_score.min()) / (engagement_score.max() - engagement_score.min())
        else:
            engagement_normalized = pd.Series([0.5] * len(df_filtered), index=df_filtered.index)
        
        # Combine with sentiment + social good (both already 0.9-1.1)
        df_filtered['video_score'] = (
            0.5 * engagement_normalized + 
            0.5 * (df_filtered['combined_multiplier'] - 0.9) / 0.2  # Convert 0.9-1.1 back to 0-1 for scoring
        )
        
        # Calculate earnings
        base_earning = 50
        max_earning = 500
        df_filtered['video_money'] = base_earning + (max_earning - base_earning) * df_filtered['video_score']
        
        df_scored = df_filtered
    
    # Dynamic cluster fund allocation
    print("💰 Allocating funds between clusters...")
    try:
        cluster_allocations = allocate_cluster_funds(df_scored, total_budget=100000)
    except Exception as e:
        print(f"⚠️ Error in fund allocation: {e}")
        cluster_allocations = pd.DataFrame()
    
    # Apply TikTok-specific rules
    print("⚖️ Applying TikTok-specific fairness rules...")
    try:
        df_final = apply_tiktok_specific_rules(df_scored, cluster_allocations)
    except Exception as e:
        print(f"⚠️ Error in TikTok rules: {e}")
        df_final = df_scored
    
    # Save results
    print(f"\n💾 Saving results...")
    df_final.to_csv('final_results_sentiment_social.csv', index=False)
    if not cluster_allocations.empty:
        cluster_allocations.to_csv('cluster_allocations_sentiment_social.csv', index=False)
    
    print("✅ Sentiment + Social Good analysis complete!")
    print("📁 Results saved to: final_results_sentiment_social.csv")
    
    return df_final, cluster_allocations

def main():
    """Standalone main function for command line usage"""
    df_final, cluster_allocations = run_analysis()
    
    # Results Display (only for standalone runs)
    print("\n" + "="*70)
    print("🏆 SENTIMENT + SOCIAL GOOD RESULTS:")
    print("="*70)
    
    # Score distribution summary (0-1 scale)
    if 'sentiment_score' in df_final.columns:
        sentiment_scores = df_final['sentiment_score']
        print(f"\n📊 SENTIMENT SCORES (0-1 scale):")
        print(f"   Range: {sentiment_scores.min():.3f} - {sentiment_scores.max():.3f}")
        print(f"   Mean: {sentiment_scores.mean():.3f}")
        print(f"   Std Dev: {sentiment_scores.std():.3f}")
        print(f"   Very Positive (>0.8): {(sentiment_scores > 0.8).sum()}")
        print(f"   Positive (0.6-0.8): {((sentiment_scores >= 0.6) & (sentiment_scores <= 0.8)).sum()}")
        print(f"   Neutral (0.4-0.6): {((sentiment_scores >= 0.4) & (sentiment_scores <= 0.6)).sum()}")
        print(f"   Negative (0.2-0.4): {((sentiment_scores >= 0.2) & (sentiment_scores < 0.4)).sum()}")
        print(f"   Very Negative (<0.2): {(sentiment_scores < 0.2).sum()}")
    
    if 'total_score' in df_final.columns:
        social_scores = df_final['total_score']
        multipliers = df_final['combined_multiplier']
        print(f"\n🌍 SOCIAL GOOD SCORES (0-1 scale):")
        print(f"   Range: {social_scores.min():.3f} - {social_scores.max():.3f}")
        print(f"   Mean: {social_scores.mean():.3f}")
        print(f"   High Social Value (>0.7): {(social_scores > 0.7).sum()}")
        print(f"\n🔧 COMBINED MULTIPLIERS (0.9-1.1 range):")
        print(f"   Range: {multipliers.min():.3f} - {multipliers.max():.3f}")
        print(f"   Mean: {multipliers.mean():.3f}")
        print(f"   Above 1.05: {(multipliers > 1.05).sum()}")
        print(f"   Below 0.95: {(multipliers < 0.95).sum()}")
    
    if not cluster_allocations.empty:
        print(f"\n📈 CLUSTER FUND ALLOCATIONS:")
        display_cols = ['cluster', 'video_count', 'money_allocation']
        available_cols = [col for col in display_cols if col in cluster_allocations.columns]
        print(cluster_allocations[available_cols])
    
    print(f"\n🏆 TOP EARNING VIDEOS:")
    display_columns = ['video_id', 'user_id', 'cluster', 'video_money', 'sentiment_score', 'sentiment_interpretation', 'combined_multiplier']
    available_display_cols = [col for col in display_columns if col in df_final.columns]
    
    if 'video_money' in df_final.columns:
        top_videos = df_final.nlargest(min(15, len(df_final)), 'video_money')[available_display_cols]
        print(top_videos)
    
    return df_final, cluster_allocations

if __name__ == "__main__":
    final_df, allocations = main()
