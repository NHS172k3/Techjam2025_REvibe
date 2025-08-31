import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Base stats - ONLY engagement metrics + sentiment (NO quality)
# sentiment_score is now 0-1 instead of -1,1
base_stats = ['likes', 'comment_count', 'shares', 'saved_count', 'viewer_retention_percentage', 'sentiment_score']

def calculate_enhanced_metrics(df):
    """Calculate additional metrics using ONLY sentiment + engagement"""
    df = df.copy()
    
    # Engagement rate (industry standard)
    df['engagement_rate'] = (df['likes'] + df['comment_count'] + df['shares']) / df['views'].clip(lower=1)
    
    # Viral coefficient (shares as predictor of viral potential)
    df['viral_coefficient'] = df['shares'] / df['views'].clip(lower=1)
    
    # Save rate (strong indicator of content quality)
    df['save_rate'] = df['saved_count'] / df['views'].clip(lower=1)
    
    # Overall score combining retention and sentiment (both 0-1 scale)
    df['overall_score'] = (
        0.6 * df['sentiment_score'] +                       # Sentiment already 0-1
        0.4 * (df['viewer_retention_percentage'] / 100)     # Retention 0-1
    )
    
    return df

def normalize_and_score(df, money_per_cluster=1000, use_enhanced=True):
    """
    Enhanced normalization using ONLY sentiment + social good (NO quality)
    """
    if use_enhanced:
        df = calculate_enhanced_metrics(df)
        enhanced_stats = base_stats + ['engagement_rate', 'viral_coefficient', 'save_rate', 'overall_score']
        stats = enhanced_stats
    else:
        stats = base_stats.copy()
    
    def normalize_group(group):
        scaler = StandardScaler()
        
        try:
            normalized = scaler.fit_transform(group[stats])
            normed_df = pd.DataFrame(normalized, columns=[f'norm_{s}' for s in stats], index=group.index)
            
            # Min-Max scaling to ensure 0-1 range
            for col in normed_df.columns:
                col_data = normed_df[col]
                min_val, max_val = col_data.min(), col_data.max()
                if max_val > min_val:
                    normed_df[col] = (col_data - min_val) / (max_val - min_val)
                else:
                    normed_df[col] = 0.5
            
            group_normed = group.copy()
            group_normed = pd.concat([group_normed, normed_df], axis=1)
            
            # Weighted video score - emphasize sentiment
            if use_enhanced:
                weights = {
                    'norm_likes': 0.15,
                    'norm_comment_count': 0.12,
                    'norm_shares': 0.15,
                    'norm_saved_count': 0.10,
                    'norm_viewer_retention_percentage': 0.15,
                    'norm_sentiment_score': 0.25,  # Heavy weight on sentiment (already 0-1)
                    'norm_engagement_rate': 0.08
                }
                group_normed['video_score'] = sum(group_normed[col] * weight for col, weight in weights.items())
            else:
                group_normed['video_score'] = group_normed[[f'norm_{s}' for s in stats]].mean(axis=1)
            
        except Exception as e:
            print(f"Warning: Normalization failed for cluster, using fallback: {e}")
            normed = group[stats].apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5)
            group_normed = group.copy()
            group_normed[[f'norm_{s}' for s in stats]] = normed
            group_normed['video_score'] = normed.mean(axis=1)
        
        return group_normed

    df_normed = df.groupby('cluster', group_keys=False).apply(normalize_group)
    category_scores = df_normed.groupby('cluster')['video_score'].sum().reset_index().rename(columns={'video_score': 'category_total_score'})
    df_final = df_normed.merge(category_scores, on='cluster', how='left')
    df_final['video_money'] = (df_final['video_score'] / df_final['category_total_score']) * money_per_cluster
    
    return df_final

def filter_content(df, cluster_column='cluster', threshold_multiplier=1.0):
    """
    Filter videos using ONLY sentiment + engagement (NO quality)
    """
    df = df.copy()
    key_metrics = ['views', 'likes', 'shares', 'viewer_retention_percentage', 'sentiment_score']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    df['content_flag'] = False
    
    for cluster in df[cluster_column].unique():
        cluster_mask = df[cluster_column] == cluster
        cluster_data = df[cluster_mask]
        
        # Create composite score using z-scores
        z_scores = {}
        for metric in available_metrics:
            mean = cluster_data[metric].mean()
            std = cluster_data[metric].std() if cluster_data[metric].std() > 0 else 1
            z_scores[metric] = (cluster_data[metric] - mean) / std
        
        # Weights emphasizing sentiment and engagement
        weights = {
            'views': 0.20,
            'likes': 0.20,
            'shares': 0.20,
            'viewer_retention_percentage': 0.20,
            'sentiment_score': 0.20  # Equal weight to sentiment (already 0-1)
        }

        composite_score = pd.Series(0, index=cluster_data.index)
        for m in available_metrics:
            composite_score += z_scores[m] * weights.get(m, 0.0)

        # Apply social good multiplier only
        if 'social_value_multiplier' in cluster_data.columns:
            composite_score *= cluster_data['social_value_multiplier']

        content_threshold = 0.0 * threshold_multiplier
        df.loc[cluster_mask, 'content_flag'] = composite_score > content_threshold
    
    content_pct = df['content_flag'].mean() * 100
    print(f"Content filter: {content_pct:.1f}% of videos are above average")
    
    return df

def allocate_cluster_funds(df, total_budget=100000):
    """Allocation strategy using ONLY sentiment + social good"""
    import numpy as np
    from scipy import stats
    
    cluster_stats = df.groupby('cluster').agg({
        'video_score': ['mean', 'median', 'std', 'count', 'sum'],
        'views': 'sum',
        'content_flag': 'mean',
        'sentiment_score': ['mean', 'std'],  # Use sentiment score (0-1)
        'analyzed_comments_count': 'mean'
    })
    
    # Flatten multi-level columns
    cluster_stats.columns = ['avg_score', 'median_score', 'std_score', 
                            'video_count', 'total_score', 'total_views',
                            'content_ratio', 'avg_sentiment', 'std_sentiment',
                            'avg_comments_analyzed']
    
    # Performance factor
    performance_factor = (0.6 * cluster_stats['total_score'] / cluster_stats['total_score'].sum() + 
                         0.4 * cluster_stats['total_views'] / cluster_stats['total_views'].sum())
    
    # Diversity factor
    diversity_factor = 1 / cluster_stats['video_count']
    diversity_factor = diversity_factor / diversity_factor.sum()
    
    # Sentiment factor (already 0-1 scale)
    sentiment_factor = cluster_stats['avg_sentiment'] * cluster_stats['content_ratio']
    sentiment_factor = sentiment_factor / sentiment_factor.sum()
    
    # Emerging category factor (already 0-1 scale)
    potential_factor = (cluster_stats['avg_sentiment'].clip(0, 1) * 
                       (1 / np.sqrt(cluster_stats['video_count'])))
    potential_factor = potential_factor / potential_factor.sum()
    
    # Combined allocation
    cluster_stats['money_allocation'] = (
        0.40 * performance_factor + 
        0.20 * diversity_factor + 
        0.30 * sentiment_factor + 
        0.10 * potential_factor
    ) * total_budget
    
    # Convert index to column so 'cluster' becomes accessible
    cluster_stats = cluster_stats.reset_index()
    
    return cluster_stats

def apply_tiktok_specific_rules(df, cluster_allocations):
    """Apply TikTok-specific rules using ONLY sentiment (0-1 scale)"""
    df = df.copy()
    
    # 1. Creator caps (prevent single creator domination)
    creator_earnings = df.groupby('user_id')['video_money'].sum().reset_index()
    top_earner_threshold = creator_earnings['video_money'].quantile(0.90)
    
    # Identify creators above threshold
    high_earners = creator_earnings[creator_earnings['video_money'] > top_earner_threshold]['user_id'].tolist()
    
    # Apply diminishing returns to additional videos from high earners
    for user_id in high_earners:
        user_videos = df[df['user_id'] == user_id].sort_values('video_money', ascending=False)
        
        # First video gets 100%, second 90%, third 80%, etc.
        for i, idx in enumerate(user_videos.index[1:], 1):
            diminishing_factor = max(0.5, 1.0 - (i * 0.1))
            df.loc[idx, 'video_money'] *= diminishing_factor
    
    # 2. Small creator boost (TikTok boosts new/smaller creators)
    video_counts = df.groupby('user_id').size().reset_index(name='video_count')
    small_creators = video_counts[video_counts['video_count'] <= 2]['user_id'].tolist()
    
    # Apply small creator boost (10% bonus)
    df.loc[df['user_id'].isin(small_creators), 'video_money'] *= 1.1
    
    # 3. Positive sentiment bonus (exceptional sentiment gets extra rewards)
    for cluster in df['cluster'].unique():
        cluster_mask = df['cluster'] == cluster
        cluster_sentiments = df.loc[cluster_mask, 'sentiment_score']
        positive_threshold = cluster_sentiments.mean() + (1.5 * cluster_sentiments.std())
        
        # Apply positive sentiment bonus (sentiment already 0-1)
        positive_mask = (df['cluster'] == cluster) & (df['sentiment_score'] > positive_threshold)
        df.loc[positive_mask, 'video_money'] *= 1.15
    
    # 4. Minimum payout threshold
    min_payout = df['video_money'].median() * 0.1
    low_payout_mask = df['video_money'] < min_payout
    
    df.loc[low_payout_mask & df['content_flag'], 'video_money'] = min_payout
    df.loc[low_payout_mask & ~df['content_flag'], 'video_money'] = 0
    
    # Recalculate total to ensure we stay within budget
    total_allocated = df['video_money'].sum()
    original_total = cluster_allocations['money_allocation'].sum()
    
    # Normalize to maintain budget
    df['video_money'] = df['video_money'] * (original_total / total_allocated)
    
    return df

# Legacy function names for compatibility
filter_quality_content = filter_content

if __name__ == "__main__":
    df = pd.read_csv('full_video_dataset_with_engagement.csv')
    df_final = normalize_and_score(df)
    print(df_final[['video_id', 'cluster', 'video_score', 'category_total_score', 'video_money']])