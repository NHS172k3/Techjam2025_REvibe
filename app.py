import streamlit as st
import pandas as pd
import json
import os
from main import run_analysis

st.title("TikTok Video Uploader & Metrics Entry")

# Define all fields from to_insert.json
fields = [
    "video_id", "user_id", "views", "captions", "subtitles", "top_comments",
    "likes", "comment_count", "shares", "saved_count", "viewer_retention_percentage"
]

# Video upload
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# Metrics entry
st.header("Enter Video Metadata & Metrics")
inputs = {}
for field in fields:
    if field == "video_id" or field == "user_id":
        inputs[field] = st.number_input(field.replace('_', ' ').title(), min_value=0, value=0)
    elif field in ["views", "likes", "comment_count", "shares", "saved_count"]:
        inputs[field] = st.number_input(field.replace('_', ' ').title(), min_value=0, value=0)
    elif field == "viewer_retention_percentage":
        inputs[field] = st.number_input("Viewer Retention (%)", min_value=0.0, max_value=100.0, value=50.0)
    elif field in ["captions", "subtitles", "top_comments"]:
        inputs[field] = st.text_area(field.replace('_', ' ').title())

if st.button("Submit"):
    # Save all fields and video filename to to_insert.json
    record = inputs.copy()
    record["video_file"] = video_file.name if video_file else ""
    with open("to_insert.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    st.success("Video and metrics saved! Running analysis...")

    # Run analysis and get df_final directly
    df_final, allocations = run_analysis()

    # Show the last video's multipliers and money
    last_row = df_final.iloc[-1]
    st.subheader("Your Video's Results")
    st.markdown(f"""
    | Field                      | Value |
    |---------------------------|-------|
    | **Video ID**              | {last_row.get('video_id', None)} |
    | **Quality Sentiment Multiplier** | {last_row.get('quality_sentiment_multiplier', None)} |
    | **Social Value Multiplier**      | {last_row.get('social_value_multiplier', None)} |
    | **Video Money**           | {last_row.get('video_money', None)} |
    """)
