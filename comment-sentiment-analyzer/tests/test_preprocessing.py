import pytest
import pandas as pd
from src.data.preprocessing import clean_comments, normalize_text, handle_missing_values

def test_clean_comments():
    raw_comments = pd.Series([
        "This video is amazing! | I love it!",
        "Terrible quality. | Not worth watching.",
        "Great content, but the audio is bad.",
        None,
        "  "
    ])
    
    cleaned_comments = clean_comments(raw_comments)
    
    expected_output = pd.Series([
        "This video is amazing! I love it!",
        "Terrible quality. Not worth watching.",
        "Great content, but the audio is bad.",
        "",
        ""
    ])
    
    pd.testing.assert_series_equal(cleaned_comments, expected_output)

def test_normalize_text():
    text = "This is a GREAT video!!!"
    normalized_text = normalize_text(text)
    expected_output = "this is a great video"
    
    assert normalized_text == expected_output

def test_handle_missing_values():
    df = pd.DataFrame({
        'comments': ["Good video", None, "Bad video", "", "Excellent!"],
        'ratings': [5, None, 2, 0, 4]
    })
    
    cleaned_df = handle_missing_values(df)
    
    expected_output = pd.DataFrame({
        'comments': ["Good video", "No comment", "Bad video", "No comment", "Excellent!"],
        'ratings': [5, 0, 2, 0, 4]
    })
    
    pd.testing.assert_frame_equal(cleaned_df, expected_output)