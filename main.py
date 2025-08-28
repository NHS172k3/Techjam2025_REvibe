import pandas as pd
from ST_tokenize import sort_category  # Use the correct function name from ST_tokenize.py
from normalize import normalize_and_score

# Load data
df = pd.read_csv('full_video_dataset_with_engagement.csv')

# Cluster videos using sort_category (returns df with 'cluster' column)
df_clustered = sort_category(df)

# Normalize and score
df_final = normalize_and_score(df_clustered)

# Show results
print(df_final[['video_id', 'cluster', 'video_score', 'category_total_score', 'video_money']])

