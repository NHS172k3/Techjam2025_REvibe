from transformers import pipeline
import pandas as pd
from collections import Counter
import re

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

df = pd.read_csv('full_video_dataset_with_engagement.csv')

def senti_analysis(df):

    split_comments = df["top_comments"].astype(str).str.split(r"\s*\|\s*", regex=True)

    flat = [c for lst in split_comments for c in lst]

    preds = pipe(flat, truncation=True, batch_size=32)

    grouped = [preds[i:i+5] for i in range(0, len(preds), 5)]

    def normalize_label(lbl: str) -> str:
        l = lbl.lower()
        if "pos" in l: return "positive"
        if "neg" in l: return "negative"
        return "neutral"

    majority_labels = []
    avg_scores = []

    for g in grouped:
        total_score = 0
        for record in g:
            if record['label'] == 'Negative':
                total_score += record['score']
            elif record['label'] == 'Positive':
                total_score -= record['score']
        final_score = total_score / len(g)
        if final_score>0.2:
            majority_labels.append("positive")
        elif final_score < -0.2:
            majority_labels.append("negative")
        else:
            majority_labels.append("neutral")
        avg_scores.append(final_score)
    df = df.copy()

    df["sent_label_majority"] = majority_labels
    df["sent_score_avg"] = avg_scores

    #df.to_csv("full_video_dataset_with_comments_scored_per_comment.csv", index=False)
    print(df[["video_id","sent_label_majority","sent_score_avg"]])

    return df