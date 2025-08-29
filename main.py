import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ST_tokenize import sort_category  
from Senti_Analysis import senti_analysis  
from normalize import normalize_and_score, allocate_cluster_funds, filter_quality_content, apply_tiktok_specific_rules

def create_visualizations(df_final, cluster_allocations):
    """Create visualizations for analysis"""
    # Set up plotting style
    plt.style.use('fivethirtyeight')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Cluster allocation
    cluster_data = cluster_allocations.sort_values('money_allocation', ascending=False)
    sns.barplot(x='cluster', y='money_allocation', data=cluster_data, ax=axes[0,0])
    axes[0,0].set_title('Budget Allocation by Content Category')
    axes[0,0].set_ylabel('Allocation (USD)')
    axes[0,0].set_xlabel('Content Category')
    
    # 2. Video earnings distribution
    sns.histplot(df_final['video_money'], bins=20, ax=axes[0,1])
    axes[0,1].set_title('Video Earnings Distribution')
    axes[0,1].set_xlabel('Earnings (USD)')
    
    # 3. Quality vs. Earnings
    sns.scatterplot(
        x='video_score', 
        y='video_money', 
        hue='cluster', 
        size='views',
        sizes=(20, 200),
        alpha=0.7,
        data=df_final,
        ax=axes[1,0]
    )
    axes[1,0].set_title('Quality Score vs. Earnings')
    
    # 4. Category performance
    sentiment_by_cluster = df_final.groupby('cluster')['sent_score_avg'].mean().reset_index()
    retention_by_cluster = df_final.groupby('cluster')['viewer_retention_percentage'].mean().reset_index()
    
    x = sentiment_by_cluster['cluster']
    width = 0.35
    
    axes[1,1].bar(x - width/2, sentiment_by_cluster['sent_score_avg'], width, label='Sentiment')
    axes[1,1].bar(x + width/2, retention_by_cluster['viewer_retention_percentage']/100, width, label='Retention')
    axes[1,1].set_title('Category Performance Metrics')
    axes[1,1].set_ylim(0, 1)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('tiktok_profit_sharing_analysis.png', dpi=300)
    
    return fig

def main():
    print("🚀 Starting TikTok Profit Sharing Analysis...")
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv('full_video_dataset_with_engagement.csv')
    # df = df.head(20)  # Uncomment to test with first 20 videos
    print(f"Loaded {len(df)} videos")
    
    # Cluster videos
    print("🔍 Clustering videos by content similarity...")
    df_clustered = sort_category(df)
    print(f"Created {df_clustered['cluster'].nunique()} clusters")
    
    # Sentiment analysis (adds 'sent_score_avg' and 'sent_label_majority')
    print("😊 Analyzing sentiment...")
    df_analysed = senti_analysis(df_clustered)
    
    # NEW STEP 1: Apply quality filter (above mean performance)
    print("🏆 Filtering for above-average quality content...")
    df_filtered = filter_quality_content(df_analysed)
    
    # Enhanced normalization and scoring
    print("📊 Computing quality scores and normalizing metrics...")
    df_scored = normalize_and_score(df_filtered, use_enhanced=True)
    
    # Dynamic cluster fund allocation
    print("💰 Allocating funds between clusters...")
    cluster_allocations = allocate_cluster_funds(df_scored, total_budget=100000)
    
    # NEW STEP 2: Apply TikTok-specific rules and edge cases
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
    top_videos = df_final.nlargest(10, 'video_money')[
        ['video_id', 'user_id', 'cluster', 'video_score', 'video_money', 'sent_score_avg', 'quality_flag']
    ]
    print(top_videos)
    
    print("\n" + "="*60)
    print("📊 CLUSTER SUMMARY:")
    print("="*60)
    cluster_summary = df_final.groupby('cluster').agg({
        'video_money': ['sum', 'mean', 'count'],
        'video_score': 'mean',
        'sent_score_avg': 'mean',
        'quality_flag': 'mean'
    }).round(2)
    cluster_summary.columns = ['total_earnings', 'avg_earnings', 'video_count', 'avg_quality', 'avg_sentiment', 'quality_pct']
    print(cluster_summary)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    fig = create_visualizations(df_final, cluster_allocations)
    
    # Save results
    print("\n💾 Saving results...")
    df_final.to_csv('final_results.csv', index=False)
    cluster_allocations.to_csv('cluster_allocations.csv', index=False)
    
    print("✅ Analysis complete!")
    return df_final, cluster_allocations, fig

if __name__ == "__main__":
    final_df, allocations, fig = main()
