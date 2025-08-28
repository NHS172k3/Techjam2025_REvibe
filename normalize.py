import pandas as pd

stats = ['likes', 'comment_count', 'shares', 'saved_count', 'viewer_retention_percentage']

def normalize_and_score(df, money_per_cluster=1000):
    """
    Takes a DataFrame with a 'cluster' column and engagement stats,
    normalizes stats within each cluster, computes video/category scores,
    and assigns money proportionally.
    Returns the updated DataFrame.
    """
    def normalize_group(group):
        normed = group[stats].apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0)
        group_normed = group.copy()
        group_normed[[f'norm_{s}' for s in stats]] = normed
        group_normed['video_score'] = normed.mean(axis=1)
        return group_normed

    df_normed = df.groupby('cluster', group_keys=False).apply(normalize_group)
    category_scores = df_normed.groupby('cluster')['video_score'].sum().reset_index().rename(columns={'video_score': 'category_total_score'})
    df_final = df_normed.merge(category_scores, on='cluster', how='left')
    df_final['video_money'] = (df_final['video_score'] / df_final['category_total_score']) * money_per_cluster
    return df_final

if __name__ == "__main__":
    df = pd.read_csv('full_video_dataset_with_engagement.csv')
    df_final = normalize_and_score(df)
    print(df_final[['video_id', 'cluster', 'video_score', 'category_total_score', 'video_money'] + [f'norm_{s}' for s in stats]])