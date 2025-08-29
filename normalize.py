import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

base_stats = ['likes', 'comment_count', 'shares', 'saved_count', 'viewer_retention_percentage', 'sent_score_avg']

def calculate_enhanced_metrics(df):
    """Calculate additional quality metrics"""
    df = df.copy()
    
    # Engagement rate (industry standard)
    df['engagement_rate'] = (df['likes'] + df['comment_count'] + df['shares']) / df['views'].clip(lower=1)
    
    # Viral coefficient (shares as predictor of viral potential)
    df['viral_coefficient'] = df['shares'] / df['views'].clip(lower=1)
    
    # Save rate (strong indicator of content quality)
    df['save_rate'] = df['saved_count'] / df['views'].clip(lower=1)
    
    # Quality score combining sentiment and retention
    df['quality_score_enhanced'] = (df['viewer_retention_percentage'] / 100) * (1 + df['sent_score_avg'].clip(-1, 1))
    
    return df

def normalize_and_score(df, money_per_cluster=1000, use_enhanced=True):
    """
    Enhanced normalization with additional quality metrics
    """
    if use_enhanced:
        df = calculate_enhanced_metrics(df)
        # Add enhanced metrics to stats
        enhanced_stats = base_stats + ['engagement_rate', 'viral_coefficient', 'save_rate']
        stats = enhanced_stats
    else:
        stats = base_stats.copy()
    
    def normalize_group(group):
        # Use StandardScaler for better normalization
        scaler = StandardScaler()
        
        try:
            normalized = scaler.fit_transform(group[stats])
            
            # Convert to 0-1 scale
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
            
            # Weighted video score (give more weight to engagement and sentiment)
            if use_enhanced:
                weights = {
                    'norm_likes': 0.15,
                    'norm_comment_count': 0.1,
                    'norm_shares': 0.1,
                    'norm_saved_count': 0.1,
                    'norm_viewer_retention_percentage': 0.2,
                    'norm_sent_score_avg': 0.15,
                    'norm_engagement_rate': 0.15,
                    'norm_viral_coefficient': 0.05
                }
                group_normed['video_score'] = sum(group_normed[col] * weight for col, weight in weights.items())
            else:
                group_normed['video_score'] = group_normed[[f'norm_{s}' for s in stats]].mean(axis=1)
            
        except Exception as e:
            print(f"Warning: Normalization failed for cluster, using fallback: {e}")
            # Fallback to simple min-max
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

# ADD: Dynamic fund allocation function
def allocate_cluster_funds(df, total_budget=100000):
    """
    Advanced allocation strategy that handles different quality distributions
    and ensures fair allocation regardless of score skewness
    """
    import numpy as np
    from scipy import stats
    
    # Calculate extensive cluster statistics
    cluster_stats = df.groupby('cluster').agg({
        'video_score': ['mean', 'median', 'std', 'count', 'sum'],
        'views': 'sum',
        'quality_flag': 'mean',  # Percentage of quality content
        'sent_score_avg': 'mean'
    })
    
    # Flatten multi-level columns
    cluster_stats.columns = ['avg_score', 'median_score', 'std_score', 
                            'video_count', 'total_score', 'total_views',
                            'quality_ratio', 'avg_sentiment']
    
    # Calculate distribution characteristics for each cluster
    distribution_data = []
    for cluster in df['cluster'].unique():
        cluster_scores = df[df['cluster'] == cluster]['video_score'].values
        
        if len(cluster_scores) >= 3:  # Need at least 3 points for skewness
            skewness = stats.skew(cluster_scores)
            percentile_75 = np.percentile(cluster_scores, 75)
            percentile_25 = np.percentile(cluster_scores, 25)
        else:
            skewness = 0
            percentile_75 = percentile_25 = cluster_scores.mean() if len(cluster_scores) > 0 else 0
            
        distribution_data.append({
            'cluster': cluster,
            'skewness': skewness,
            'percentile_ratio': percentile_75 / max(percentile_25, 0.001)  # Avoid division by zero
        })
    
    distribution_df = pd.DataFrame(distribution_data)
    cluster_stats = cluster_stats.reset_index().merge(distribution_df, on='cluster')
    
    # Calculate multiple allocation factors
    
    # 1. Performance factor (35%) - rewards high quality regardless of distribution
    performance_factor = (0.7 * cluster_stats['total_score'] / cluster_stats['total_score'].sum() + 
                         0.3 * cluster_stats['total_views'] / cluster_stats['total_views'].sum())
    
    # 2. Diversity factor (25%) - inversely proportional to cluster size
    diversity_factor = 1 / cluster_stats['video_count']
    diversity_factor = diversity_factor / diversity_factor.sum()
    
    # 3. Quality distribution factor (25%) - adjusts for skewed distributions
    # For right-skewed distributions (few excellent videos), boost if quality ratio is good
    # For left-skewed distributions (many good videos), boost to reward consistency
    
    # Convert skewness to a 0-1 scale where 0.5 is no skew
    normalized_skew = 1 / (1 + np.exp(-cluster_stats['skewness']))
    
    # Calculate quality density (how concentrated the quality is)
    quality_density = cluster_stats['quality_ratio'] * normalized_skew
    
    # Adjust for distribution shape: favor right-skewed only if quality is high
    distribution_factor = (cluster_stats['quality_ratio'] * (1 + 0.5 * normalized_skew))
    distribution_factor = distribution_factor / distribution_factor.sum()
    
    # 4. Emerging category factor (15%) - boost categories with high sentiment but fewer videos
    potential_factor = (cluster_stats['avg_sentiment'].clip(0, 1) * 
                       (1 / np.sqrt(cluster_stats['video_count'])))
    potential_factor = potential_factor / potential_factor.sum()
    
    # Combined allocation with balanced weights
    cluster_stats['money_allocation'] = (
        0.35 * performance_factor + 
        0.25 * diversity_factor + 
        0.25 * distribution_factor + 
        0.15 * potential_factor
    ) * total_budget
    
    # Add distribution information for reference
    cluster_stats['distribution_type'] = np.where(
        cluster_stats['skewness'] > 0.5, 'Right-skewed (few excellent videos)',
        np.where(cluster_stats['skewness'] < -0.5, 'Left-skewed (many good videos)', 
                'Balanced distribution')
    )
    
    return cluster_stats

def filter_quality_content(df, cluster_column='cluster', threshold_multiplier=1.0):
    """
    Filter videos to only include those above mean metrics within their cluster
    """
    df = df.copy()
    key_metrics = ['views', 'likes', 'shares', 'viewer_retention_percentage', 'sent_score_avg']
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    # Add quality flag column (initialized as False)
    df['quality_flag'] = False
    
    # Calculate mean metrics per cluster
    for cluster in df[cluster_column].unique():
        cluster_mask = df[cluster_column] == cluster
        cluster_data = df[cluster_mask]
        
        # Create a composite score using z-scores
        z_scores = {}
        for metric in available_metrics:
            mean = cluster_data[metric].mean()
            std = cluster_data[metric].std() if cluster_data[metric].std() > 0 else 1
            z_scores[metric] = (cluster_data[metric] - mean) / std
        
        # Composite quality score (weighted average of z-scores)
        weights = {
            'views': 0.25,
            'likes': 0.2, 
            'shares': 0.2, 
            'viewer_retention_percentage': 0.25,
            'sent_score_avg': 0.1
        }
        
        composite_score = pd.Series(0, index=cluster_data.index)
        for m in available_metrics:
            composite_score += z_scores[m] * weights.get(m, 0.1)
        
        # Flag videos above threshold (positive composite score = above average)
        quality_threshold = 0.0 * threshold_multiplier
        df.loc[cluster_mask, 'quality_flag'] = composite_score > quality_threshold
    
    # Calculate percentage of videos flagged as quality
    quality_pct = df['quality_flag'].mean() * 100
    print(f"Quality filter: {quality_pct:.1f}% of videos are above average quality")
    
    return df

def apply_tiktok_specific_rules(df, cluster_allocations):
    """Apply TikTok-specific rules and handle edge cases"""
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
    
    # 3. Viral bonus (exceptional videos get extra rewards)
    for cluster in df['cluster'].unique():
        cluster_mask = df['cluster'] == cluster
        cluster_views = df.loc[cluster_mask, 'views']
        viral_threshold = cluster_views.mean() + (1.5 * cluster_views.std())
        
        # Apply viral bonus
        viral_mask = (df['cluster'] == cluster) & (df['views'] > viral_threshold)
        df.loc[viral_mask, 'video_money'] *= 1.2
    
    # 4. Minimum payout threshold or exclude if low quality
    min_payout = df['video_money'].median() * 0.1
    low_payout_mask = df['video_money'] < min_payout
    
    df.loc[low_payout_mask & df['quality_flag'], 'video_money'] = min_payout
    df.loc[low_payout_mask & ~df['quality_flag'], 'video_money'] = 0
    
    # Recalculate total to ensure we stay within budget
    total_allocated = df['video_money'].sum()
    original_total = cluster_allocations['money_allocation'].sum()
    
    # Normalize to maintain budget
    df['video_money'] = df['video_money'] * (original_total / total_allocated)
    
    return df

if __name__ == "__main__":
    df = pd.read_csv('full_video_dataset_with_engagement.csv')
    df_final = normalize_and_score(df)
    print(df_final[['video_id', 'cluster', 'video_score', 'category_total_score', 'video_money']])