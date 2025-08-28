from transformers import pipeline
import pandas as pd
from collections import Counter
import re

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

df = pd.read_csv('full_video_dataset_with_engagement.csv')

# Split the concatenated comments (they’re joined with " | ")
split_comments = df["top_comments"].astype(str).str.split(r"\s*\|\s*", regex=True)

# Flatten to one big list, keeping order
flat = [c for lst in split_comments for c in lst]

# Batch classify
preds = pipe(flat, truncation=True, batch_size=32)

# Regroup into chunks of 5 (since each video has 5 comments)
grouped = [preds[i:i+5] for i in range(0, len(preds), 5)]

def normalize_label(lbl: str) -> str:
    l = lbl.lower()
    if "pos" in l: return "positive"
    if "neg" in l: return "negative"
    return "neutral"

majority_labels = []
avg_scores = []

for g in grouped:
    labels = [normalize_label(p["label"]) for p in g]
    scores = [p["score"] for p in g]
    majority = Counter(labels).most_common(1)[0][0]
    majority_labels.append(majority)
    avg_scores.append(sum(scores)/len(scores))

df["sent_label_majority"] = majority_labels
df["sent_score_avg"] = avg_scores

df.to_csv("full_video_dataset_with_comments_scored_per_comment.csv", index=False)
print(df[["video_id","sent_label_majority","sent_score_avg"]])