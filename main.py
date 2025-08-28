import pandas as pd
from ST_tokenize import sort_category  
from Senti_Analysis import senti_analysis  
from normalize import normalize_and_score

# Load data
df = pd.read_csv('full_video_dataset_with_engagement.csv')

# Cluster videos
df_clustered = sort_category(df)

# Sentiment analysis (adds 'sent_score_avg' and 'sent_label_majority')
df_analysed = senti_analysis(df_clustered)

# Normalize and score, now including sentiment score
df_final = normalize_and_score(df_analysed)

# Show results
print(df_final[['video_id', 'cluster', 'video_score', 'category_total_score', 'video_money', 'sent_score_avg']])

