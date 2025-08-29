import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from ST_tokenize import sort_category  
from normalize import normalize_and_score, allocate_cluster_funds, filter_quality_content, apply_tiktok_specific_rules

def analyze_video_sentiment_quality(df):
    """Apply the integrated sentiment+quality model to analyze videos"""
    try:
        import sys
        import os
        import pandas as pd
        import numpy as np
        import torch
        from pathlib import Path
        import re
        
        # Define the missing helper functions
        def _split_comments(comments):
            if pd.isna(comments):
                return []
            if isinstance(comments, list):
                return comments
            comments_str = str(comments)
            if '|' in comments_str:
                return [c.strip() for c in comments_str.split('|') if c.strip()]
            return [comments_str]
        
        def _clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Change path to use existing folder name
        analyzer_path = Path(__file__).parent / "video-quality-classifier"
        
        # Add both the parent directory and the analyzer path to sys.path
        sys.path.append(str(Path(__file__).parent))
        sys.path.append(str(analyzer_path))
        
        try:
            # Import directly from video_quality_classifier package
            from video_quality_classifier.src.models.classifier import IntegratedQualityModel
            model_path = analyzer_path / "models" / "best_integrated_model.pth"
        except ImportError as e:
            print(f"Import error: {e}")
            # Try alternative import path
            from src.models.classifier import IntegratedQualityModel
            model_path = analyzer_path / "models" / "best_integrated_model.pth"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print("❌ Integrated sentiment-quality model not found")
            print(f"Expected location: {model_path}")
            df['sentiment_quality_score'] = 0.5  # Default neutral score
            return df
        
        # Initialize model and tokenizer
        from transformers import AutoTokenizer
        
        model = IntegratedQualityModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Set to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"✅ Loaded integrated sentiment+quality analyzer on {device}")
        
        # Process videos
        results = []
        
        for idx, row in df.iterrows():
            video_id = row['video_id']
            comments = _split_comments(row.get('top_comments', ''))
            
            # Process each comment through the integrated model
            comment_scores = []
            
            for comment in comments:
                if not comment or len(comment.strip()) < 3:
                    continue
                    
                # Clean comment
                cleaned_comment = _clean_text(comment)
                
                # Tokenize
                inputs = tokenizer(
                    cleaned_comment,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Get prediction from integrated model
                with torch.no_grad():
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    
                    # Get unified sentiment+quality score (0-1)
                    score = model(input_ids, attention_mask).item()
                    comment_scores.append(score)
            
            # Calculate video-level score
            if comment_scores:
                # Use weighted average - recent comments might be more important
                weights = np.linspace(0.5, 1.0, len(comment_scores))  # More weight to later comments
                avg_score = np.average(comment_scores, weights=weights)
                num_comments = len(comment_scores)
            else:
                avg_score = 0.5  # Neutral if no comments
                num_comments = 0
                
            results.append({
                'video_id': video_id,
                'sentiment_quality_score': avg_score,
                'analyzed_comments_count': num_comments,
                'comment_scores': comment_scores[:5]  # Store first 5 scores for debugging
            })
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} videos...")
        
        # Convert results to DataFrame and merge
        results_df = pd.DataFrame(results)
        df = df.merge(results_df, on='video_id', how='left')
        
        # Fill missing values
        df['sentiment_quality_score'] = df['sentiment_quality_score'].fillna(0.5)
        df['analyzed_comments_count'] = df['analyzed_comments_count'].fillna(0)
        
        # Create interpretation categories
        df['score_interpretation'] = pd.cut(
            df['sentiment_quality_score'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Poor Quality/Negative', 'Below Average', 'Good Quality/Positive', 'Excellent Quality/Very Positive']
        )
        
        print(f"✅ Integrated analyzer applied to {len(df)} videos")
        print(f"📊 Score distribution:")
        print(df['score_interpretation'].value_counts())
        print(f"📊 Average score: {df['sentiment_quality_score'].mean():.3f}")
        
        return df
        
    except Exception as e:
        print(f"⚠️ Could not use integrated sentiment-quality analyzer: {e}")
        print("Setting default scores...")
        df['sentiment_quality_score'] = 0.5  # Default neutral score
        df['analyzed_comments_count'] = 0
        
        # Add the missing score_interpretation column
        df['score_interpretation'] = "Average"  # Default interpretation
        
        return df

def create_visualizations(df_final, cluster_allocations):
    """Create visualizations for analysis"""
    plt.style.use('fivethirtyeight')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Cluster allocation
    cluster_data = cluster_allocations.sort_values('money_allocation', ascending=False)
    sns.barplot(x='cluster', y='money_allocation', data=cluster_data, ax=axes[0,0])
    axes[0,0].set_title('Budget Allocation by Content Category')
    axes[0,0].set_ylabel('Allocation (USD)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Video earnings distribution
    sns.histplot(df_final['video_money'], bins=20, ax=axes[0,1])
    axes[0,1].set_title('Video Earnings Distribution')
    axes[0,1].set_xlabel('Earnings (USD)')
    
    # 3. Sentiment-Quality Score Distribution
    sns.histplot(df_final['sentiment_quality_score'], bins=20, ax=axes[0,2])
    axes[0,2].set_title('Integrated Sentiment-Quality Score Distribution')
    axes[0,2].set_xlabel('Score (0=Poor, 1=Excellent)')
    
    # 4. Score vs. Earnings
    sns.scatterplot(
        x='sentiment_quality_score',
        y='video_money', 
        hue='cluster', 
        size='views',
        sizes=(20, 200),
        alpha=0.7,
        data=df_final,
        ax=axes[1,0]
    )
    axes[1,0].set_title('Sentiment-Quality Score vs. Earnings')
    axes[1,0].set_xlabel('Integrated Score')
    
    # 5. Category performance
    category_performance = df_final.groupby('cluster').agg({
        'sentiment_quality_score': 'mean',
        'video_money': 'mean',
        'views': 'mean'
    }).reset_index()
    
    x = range(len(category_performance))
    width = 0.25
    
    axes[1,1].bar([i - width for i in x], category_performance['sentiment_quality_score'], 
                  width, label='Avg Sentiment-Quality Score', alpha=0.8)
    axes[1,1].bar(x, category_performance['video_money']/category_performance['video_money'].max(), 
                  width, label='Normalized Avg Earnings', alpha=0.8)
    axes[1,1].bar([i + width for i in x], category_performance['views']/category_performance['views'].max(), 
                  width, label='Normalized Avg Views', alpha=0.8)
    
    axes[1,1].set_title('Category Performance Comparison')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(category_performance['cluster'], rotation=45)
    axes[1,1].legend()
    
    # 6. Score interpretation breakdown
    score_breakdown = df_final['score_interpretation'].value_counts()
    axes[1,2].pie(score_breakdown.values, labels=score_breakdown.index, autopct='%1.1f%%')
    axes[1,2].set_title('Video Quality Distribution')
    
    plt.tight_layout()
    plt.savefig('integrated_sentiment_quality_analysis.png', dpi=300, bbox_inches='tight')
    
    return fig

def main():
    print("🚀 Starting TikTok Profit Sharing with Integrated Sentiment-Quality Analysis...")
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv('full_video_dataset_with_engagement.csv')
    # df = df.head(50)  # Uncomment to test with first 50 videos
    print(f"Loaded {len(df)} videos")
    
    # Cluster videos by content similarity
    print("🔍 Clustering videos by content similarity...")
    df_clustered = sort_category(df)
    print(f"Created {df_clustered['cluster'].nunique()} clusters")
    
    # Apply integrated sentiment-quality analysis (replaces separate sentiment analysis)
    print("🏆 Analyzing videos with integrated sentiment+quality model...")
    df_analyzed = analyze_video_sentiment_quality(df_clustered)
    
    # Filter for quality content using the integrated score
    print("🏆 Filtering for above-average content...")
    df_filtered = filter_quality_content(df_analyzed)
    
    # Enhanced normalization and scoring
    print("📊 Computing scores and normalizing metrics...")
    df_scored = normalize_and_score(df_filtered, use_enhanced=True)
    
    # Dynamic cluster fund allocation
    print("💰 Allocating funds between clusters...")
    cluster_allocations = allocate_cluster_funds(df_scored, total_budget=100000)
    
    # Apply TikTok-specific rules
    print("⚖️ Applying TikTok-specific fairness rules...")
    df_final = apply_tiktok_specific_rules(df_scored, cluster_allocations)
    
    # Results
    print("\n" + "="*60)
    print("📈 CLUSTER FUND ALLOCATIONS:")
    print("="*60)
    print(cluster_allocations[['cluster', 'video_count', 'avg_score', 'money_allocation']])
    
    print("\n" + "="*60)
    print("🏆 TOP 10 EARNING VIDEOS:")
    print("="*60)
    
    # Use only columns that definitely exist
    display_columns = ['video_id', 'user_id', 'cluster', 'video_score', 'video_money', 'sentiment_quality_score']
    
    # Add score_interpretation if it exists
    if 'score_interpretation' in df_final.columns:
        display_columns.append('score_interpretation')
    
    top_videos = df_final.nlargest(10, 'video_money')[display_columns]
    print(top_videos)
    
    print("\n" + "="*60)
    print("📊 CLUSTER SUMMARY:")
    print("="*60)
    cluster_summary = df_final.groupby('cluster').agg({
        'video_money': ['sum', 'mean', 'count'],
        'video_score': 'mean',
        'sentiment_quality_score': 'mean',
        'quality_flag': 'mean'
    }).round(3)
    cluster_summary.columns = ['total_earnings', 'avg_earnings', 'video_count', 'avg_video_score', 'avg_sentiment_quality', 'quality_pct']
    print(cluster_summary)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    fig = create_visualizations(df_final, cluster_allocations)
    
    # Save results
    print("\n💾 Saving results...")
    df_final.to_csv('final_results_integrated.csv', index=False)
    cluster_allocations.to_csv('cluster_allocations_integrated.csv', index=False)
    
    print("✅ Analysis complete!")
    print("📊 Generated: integrated_sentiment_quality_analysis.png")
    
    return df_final, cluster_allocations, fig

if __name__ == "__main__":
    final_df, allocations, fig = main()
