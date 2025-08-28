import pandas as pd
from ST_tokenize import sort_category  
from Senti_Analysis import senti_analysis  
from normalize import normalize_and_score

# Load data
df = pd.read_csv('full_video_dataset_with_engagement.csv')

df_clustered = sort_category(df)
print(df_clustered)

df_analysed = senti_analysis(df_clustered)
print(df_analysed["sent_score_avg"])

df_final = normalize_and_score(df_analysed)

# Show results
print(df_final[['video_id', 'cluster', 'video_score', 'category_total_score', 'video_money']])

