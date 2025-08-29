from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import numpy as np
import torch
import numpy as np
from collections import Counter
import re
import sys
import os
from pathlib import Path

# Use better sentiment model
try:
    # Try to use RoBERTa for better accuracy
    sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    use_roberta = True
    print("Using RoBERTa model for sentiment analysis")
except:
    # Fallback to your original model
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    use_roberta = False
    print("Using fallback sentiment model")

def advanced_sentiment_analysis(comments_list):
    """Advanced sentiment analysis with RoBERTa"""
    if not use_roberta:
        return fallback_sentiment_analysis(comments_list)
    
    sentiments = []
    
    for comments in comments_list:
        comment_texts = str(comments).split('|')
        comment_scores = []
        
        for comment in comment_texts[:5]:  # Limit to first 5 comments
            comment = comment.strip()
            if len(comment) > 0:
                try:
                    inputs = sentiment_tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
                    
                    with torch.no_grad():
                        outputs = sentiment_model(**inputs)
                        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        
                        # Convert to sentiment score (-1 to 1)
                        neg_prob, neu_prob, pos_prob = probabilities[0].tolist()
                        sentiment_score = pos_prob - neg_prob
                        comment_scores.append(sentiment_score)
                except:
                    comment_scores.append(0)  # Neutral if error
        
        avg_sentiment = np.mean(comment_scores) if comment_scores else 0
        sentiments.append(avg_sentiment)
    
    return sentiments

def fallback_sentiment_analysis(comments_list):
    """Your original sentiment analysis as fallback"""
    split_comments = pd.Series(comments_list).astype(str).str.split(r"\s*\|\s*", regex=True)
    flat = [c for lst in split_comments for c in lst]
    preds = pipe(flat, truncation=True, batch_size=32)
    grouped = [preds[i:i+5] for i in range(0, len(preds), 5)]
    
    
    avg_scores = []
    for g in grouped:
        total_score = 0
        for record in g:
            if record['label'] == 'Negative':
                total_score -= record['score']  # Fixed: negative should subtract
                total_score -= record['score']  # Fixed: negative should subtract
            elif record['label'] == 'Positive':
                total_score += record['score']  # Fixed: positive should add
                total_score += record['score']  # Fixed: positive should add
        final_score = total_score / len(g)
        avg_scores.append(final_score)
    
    return avg_scores

def senti_analysis(df, use_ml_classifier=True):
    """Enhanced sentiment analysis function with optional ML classifier"""
    print("Running advanced sentiment analysis...")
    
    # Get sentiment scores (your existing code)
    if use_roberta:
        sent_scores = advanced_sentiment_analysis(df["top_comments"].tolist())
    else:
        sent_scores = fallback_sentiment_analysis(df["top_comments"].tolist())
    
    # Convert scores to labels
    majority_labels = []
    for score in sent_scores:
        if score > 0.2:
            majority_labels.append("positive")
        elif score < -0.2:
        elif score < -0.2:
            majority_labels.append("negative")
        else:
            majority_labels.append("neutral")
    
    
    df = df.copy()
    df["sent_label_majority"] = majority_labels
    df["sent_score_avg"] = sent_scores
    
    print(f"Sentiment analysis complete. Positive: {majority_labels.count('positive')}, "
          f"Negative: {majority_labels.count('negative')}, Neutral: {majority_labels.count('neutral')}")
    
    # Apply quality-aware sentiment analysis
    df = quality_aware_sentiment(df)
    
    # Use ML classifier if enabled
    if use_ml_classifier:
        df = use_quality_classifier(df)
    
    if 'engagement_rate' in df.columns:
        # Calibrate sentiment by cluster
        df = calibrate_sentiment_by_cluster(df)
        
        # Create engagement-sentiment matrix
        df = engagement_sentiment_matrix(df)
        
        # Use the context-aware sentiment score
        df['sent_score_final'] = df['content_quality_score']
    else:
        # If engagement metrics not yet calculated, use raw sentiment
        df['sent_score_final'] = (df['sent_score_avg'] + 1) / 2  # Convert to 0-1 scale
    
    return df

# Precompile regex patterns for better performance
POS_PATTERNS = [
    r'(good|great|excellent|high|amazing|impressive)\s+(quality|production|editing|sound)',
    r'(well|beautifully|professionally)\s+(made|edited|produced|shot)',
    r'production\s+(value|quality)\s+(is|was)\s+(good|great|excellent)',

]
NEG_PATTERNS = [
    r'(poor|bad|terrible|low|awful)\s+(quality|production|editing|sound)',
    r'(badly|poorly|amateurishly)\s+(made|edited|produced|shot)',
    r'production\s+(value|quality)\s+(is|was)\s+(poor|bad|terrible)',
    
]

POS_RE = [re.compile(p, flags=re.I) for p in POS_PATTERNS]
NEG_RE = [re.compile(p, flags=re.I) for p in NEG_PATTERNS]
NEGATION_RE = re.compile(r"\b(not|no|never|n't|don't|doesn't|isn't|wasn't|weren't|aren't|cannot|can't)\b", flags=re.I)

def _split_comments(field):
    """Robustly split comments field into a list of strings."""
    if pd.isna(field):
        return []
    if isinstance(field, list):
        return [str(c).strip() for c in field if c and str(c).strip()]
    
    s = str(field).strip()
    # Try common separators
    if '|' in s:
        parts = [p.strip() for p in s.split('|') if p.strip()]
    elif '\n' in s:
        parts = [p.strip() for p in s.splitlines() if p.strip()]
    else:
        parts = [s] if s else []
    return parts

def _comment_quality_score(comment):
    """
    Returns a per-comment quality score in range [-1, 1].
    Positive -> production praised; Negative -> production criticized.
    """
    if not comment or len(comment.strip()) < 3:
        return 0.0

    text = comment.lower()
    pos_matches = sum(1 for r in POS_RE if r.search(text))
    neg_matches = sum(1 for r in NEG_RE if r.search(text))

    # If no pattern matched, return 0
    if pos_matches == 0 and neg_matches == 0:
        return 0.0

    # Negation heuristic: if negation exists, reduce confidence rather than blindly swapping
    has_negation = bool(NEGATION_RE.search(text))
    score = pos_matches - neg_matches

    # Normalize by max possible matches count to bound to [-1,1]
    max_patterns = max(len(POS_RE), len(NEG_RE))
    norm_score = score / max_patterns

    # If there's negation, attenuate magnitude (less confident)
    if has_negation:
        norm_score *= 0.6

    # Clip to [-1, 1]
    return max(-1.0, min(1.0, norm_score))

def quality_aware_sentiment(df):
    """
    Enhanced approach to separate quality sentiment from content opinion.
    Fixes issues with negation handling, score normalization, and engagement calculation.
    """
    df = df.copy()
    quality_scores = []
    adjusted_sentiments = []

    for _, row in df.iterrows():
        comments = _split_comments(row.get('top_comments', ''))
        per_comment_scores = [_comment_quality_score(c) for c in comments]

        # Video-level production sentiment: average of per-comment scores
        if per_comment_scores:
            prod_sent = float(np.mean(per_comment_scores))
            quality_mentions = sum(1 for s in per_comment_scores if s != 0.0)
        else:
            prod_sent = 0.0
            quality_mentions = 0

        # Engagement ratio: likes/views (0..1), robust calculation
        views = max(1.0, float(row.get('views', 0)))
        likes = float(row.get('likes', 0))
        engagement_ratio = min(1.0, likes / views)  # Now a proper fraction 0..1

        # Scale engagement_factor to a reasonable range
        engagement_factor = engagement_ratio * 0.2  # Maps 0..1 -> 0..0.2

        # Controversial content heuristic
        total_eng = float(row.get('likes', 0)) + float(row.get('comment_count', 0)) + float(row.get('shares', 0))
        high_engagement = (total_eng / views) > 0.05
        sent_score_avg = float(row.get('sent_score_avg', 0.0))
        controversial = high_engagement and sent_score_avg < 0

        # Combine rules based on content type
        if quality_mentions > 0:
            # When quality is explicitly mentioned, prioritize it
            adjusted = 0.8 * prod_sent + 0.2 * sent_score_avg
        elif controversial:
            # For controversial but engaging content, boost sentiment
            adjusted = 0.3 * sent_score_avg + 0.7 * engagement_factor
        else:
            # Default case - blend sentiment with engagement
            adjusted = 0.6 * sent_score_avg + 0.4 * engagement_factor

        # Ensure adjusted is in reasonable range
        adjusted = max(-1.0, min(1.0, adjusted))

        quality_scores.append(prod_sent)
        adjusted_sentiments.append(adjusted)

    # Add results to dataframe
    df['quality_sentiment'] = quality_scores
    df['adjusted_sentiment'] = adjusted_sentiments
    
    return df

def calibrate_sentiment_by_cluster(df):
    """Calibrate sentiment scores based on cluster norms"""
    
    df = df.copy()
    
    # Calculate cluster sentiment baselines
    cluster_sentiment = df.groupby('cluster')['sent_score_avg'].agg(['mean', 'std'])
    
    # Normalize sentiment within each cluster context
    for cluster in df['cluster'].unique():
        mask = df['cluster'] == cluster
        cluster_mean = cluster_sentiment.loc[cluster, 'mean']
        cluster_std = max(cluster_sentiment.loc[cluster, 'std'], 0.1)  # Prevent division by zero
        
        # Calculate z-score within cluster
        df.loc[mask, 'normalized_sentiment'] = (df.loc[mask, 'sent_score_avg'] - cluster_mean) / cluster_std
        
        # Convert to 0-1 scale
        df.loc[mask, 'calibrated_sentiment'] = 1 / (1 + np.exp(-df.loc[mask, 'normalized_sentiment']))
    
    
    return df

def engagement_sentiment_matrix(df):
    """
    Create a 2D matrix scoring system for engagement vs sentiment
    
    High engagement + negative sentiment = controversial but valuable content
    High engagement + positive sentiment = universally appealing content
    Low engagement + negative sentiment = genuinely poor content
    Low engagement + positive sentiment = niche but good content
    """
    
    df = df.copy()
    
    # Calculate normalized engagement (0-1)
    for cluster in df['cluster'].unique():
        mask = df['cluster'] == cluster
        engagement = df.loc[mask, 'engagement_rate']
        min_eng = engagement.min()
        max_eng = engagement.max()
        
        if max_eng > min_eng:
            df.loc[mask, 'norm_engagement'] = (engagement - min_eng) / (max_eng - min_eng)
        else:
            df.loc[mask, 'norm_engagement'] = 0.5
    
    # Map into quality quadrants
    df['content_quality_score'] = (
        # Universally good: high engagement + positive sentiment
        (df['norm_engagement'] > 0.5) & (df['calibrated_sentiment'] > 0.5) * 1.0 +
        
        # Controversial but valuable: high engagement + negative sentiment
        (df['norm_engagement'] > 0.7) & (df['calibrated_sentiment'] < 0.3) * 0.7 +
        
        # Niche but good: low engagement + positive sentiment
        (df['norm_engagement'] < 0.3) & (df['calibrated_sentiment'] > 0.7) * 0.6 +
        
        # Poor content: low engagement + negative sentiment
        (df['norm_engagement'] < 0.3) & (df['calibrated_sentiment'] < 0.3) * 0.2 +
        
        # Everything else - middle ground
        ((df['norm_engagement'] >= 0.3) & (df['norm_engagement'] <= 0.7) |
         (df['calibrated_sentiment'] >= 0.3) & (df['calibrated_sentiment'] <= 0.7)) * 0.5
    )
    
    return df

def use_quality_classifier(df):
    """
    Uses the dedicated video quality classifier to analyze comments
    Returns the dataframe with quality scores added
    """
    try:
        # Add video-quality-classifier to path
        classifier_path = Path(__file__).parent / "video-quality-classifier"
        sys.path.append(str(classifier_path))
        
        # Import the classifier
        from video_quality_classifier.src.main import predict_quality
        
        # Extract comments from dataframe
        comments_data = []
        for idx, row in df.iterrows():
            video_id = row['video_id']
            comments = _split_comments(row.get('top_comments', ''))
            for comment in comments:
                if comment and len(comment.strip()) > 3:
                    comments_data.append({
                        'video_id': video_id,
                        'comment': comment
                    })
        
        # Get quality predictions
        quality_results = predict_quality(comments_data)
        
        # Aggregate by video_id
        video_quality = {}
        for result in quality_results:
            video_id = result['video_id']
            if video_id not in video_quality:
                video_quality[video_id] = []
            video_quality[video_id].append(result['quality_score'])
        
        # Add to dataframe
        df['ml_quality_score'] = df['video_id'].map(
            lambda vid: np.mean(video_quality.get(vid, [0]))
        )
        
        # Blend with existing scores
        df['quality_sentiment'] = 0.7 * df['ml_quality_score'] + 0.3 * df.get('quality_sentiment', 0)
        
        print(f"✅ Quality classifier successfully applied to {len(quality_results)} comments")
        return df
        
    except Exception as e:
        print(f"⚠️ Could not use quality classifier: {e}")
        print("Falling back to pattern-based quality detection")
        return df

def apply_quality_classifier(df, use_regression=True):
    """
    Apply the trained quality model to video comments
    
    Args:
        df: DataFrame with videos and comments
        use_regression: Whether to use regression model for continuous scores
        
    Returns:
        df: DataFrame with quality scores added
    """
    try:
        import sys
        import os
        import pandas as pd
        import numpy as np
        import torch
        from pathlib import Path
        import re
        
        # Split comments function
        def _split_comments(comments):
            if pd.isna(comments):
                return []
            if isinstance(comments, list):
                return comments
            comments_str = str(comments)
            if '|' in comments_str:
                return [c.strip() for c in comments_str.split('|') if c.strip()]
            return [comments_str]
        
        # Clean text function
        def _clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Add paths
        classifier_path = Path(__file__).parent / "video-quality-classifier"
        sys.path.append(str(classifier_path))
        
        # Import the model
        if use_regression:
            from video_quality_classifier.src.models.classifier import QualityRegressionModel as QualityModel
            model_path = Path(__file__).parent / "models" / "best_quality_regressor.pth"
        else:
            from video_quality_classifier.src.models.classifier import QualityClassifier as QualityModel
            model_path = Path(__file__).parent / "models" / "best_quality_classifier.pth"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print("Model not found, falling back to pattern-based approach")
            return df
        
        # Initialize model and tokenizer
        from transformers import AutoTokenizer
        
        model = QualityModel()
        model.load_state_dict(torch.load(model_path))
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Set to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Process videos
        results = []
        
        for _, row in df.iterrows():
            video_id = row['video_id']
            comments = _split_comments(row.get('top_comments', ''))
            
            # Clean and process each comment
            quality_scores = []
            
            for comment in comments:
                if not comment or len(comment.strip()) < 3:
                    continue
                    
                # Clean comment
                cleaned_comment = _clean_text(comment)
                
                # Tokenize
                inputs = tokenizer(
                    cleaned_comment,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Get prediction
                with torch.no_grad():
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    
                    if use_regression:
                        # Regression score (0-1)
                        score = outputs.item()
                        # Convert to -1 to 1 range
                        adjusted_score = 2 * score - 1
                        quality_scores.append(adjusted_score)
                    else:
                        # Classification
                        probs = torch.softmax(outputs, dim=1)
                        _, pred = torch.max(outputs, 1)
                        pred_class = pred.item()
                        
                        # Calculate quality score: positive (2) - negative (0)
                        # Only if the comment is about quality (pred_class != 1)
                        if pred_class != 1:  # Not neutral/not about quality
                            # Map 0->-1, 2->1
                            score = 1.0 if pred_class == 2 else -1.0
                            quality_scores.append(score)
            
            # Calculate average quality score for this video
            if quality_scores:
                avg_quality = np.mean(quality_scores)
            else:
                avg_quality = 0.0
                
            results.append({
                'video_id': video_id,
                'ml_quality_score': avg_quality,
                'quality_mentions': len(quality_scores)
            })
        
        # Convert results to DataFrame and merge
        results_df = pd.DataFrame(results)
        df = df.merge(results_df, on='video_id', how='left')
        
        # Fill missing values
        df['ml_quality_score'] = df['ml_quality_score'].fillna(0.0)
        
        # Scale to 0-1 for compatibility with other metrics
        df['quality_sentiment'] = (df['ml_quality_score'] + 1) / 2
        
        print(f"✅ Quality classifier successfully applied to videos")
        return df
        
    except Exception as e:
        print(f"⚠️ Could not use quality classifier: {e}")
        print("Falling back to pattern-based quality detection")
        return df

def detect_quality_sentiment_patterns(comment):
    """Fallback pattern-based quality detection"""
    if not isinstance(comment, str) or len(comment) < 3:
        return 0  # Neutral
        
    comment = comment.lower()
    
    # Quality-related patterns
    quality_patterns = [
        r'(quality|production|editing|edited|resolution)',
        r'(camera|shot|filmed|lighting)',
        r'(unwatchable|eyes need therapy|couldn\'t finish)',
        r'chef\'s kiss',
        r'vibes'
    ]
    
    # Check if comment is about quality
    is_quality = any(re.search(pattern, comment) for pattern in quality_patterns)
    if not is_quality:
        return 0  # Not about quality
    
    # Negative patterns
    neg_patterns = [
        r'(bad|terrible|awful|poor|low) (quality|production|editing)',
        r'unwatchable',
        r'eyes need therapy',
        r'painful to'
    ]
    
    # Positive patterns
    pos_patterns = [
        r'(good|great|amazing|excellent) (quality|production|editing)',
        r'chef\'s kiss',
        r'well (made|shot|edited)'
    ]
    
    # Score calculation
    neg_score = sum(1 for pattern in neg_patterns if re.search(pattern, comment))
    pos_score = sum(1 for pattern in pos_patterns if re.search(pattern, comment))
    
    if neg_score > pos_score:
        return -1  # Negative quality
    elif pos_score > neg_score:
        return 1  # Positive quality
    else:
        return 0  # Neutral

def analyze_video_quality_sentiment(df):
    """
    Apply the trained quality+sentiment model to video comments
    
    Args:
        df: DataFrame with videos and comments
        
    Returns:
        df: DataFrame with quality+sentiment scores added
    """
    try:
        import sys
        import os
        import pandas as pd
        import numpy as np
        import torch
        from pathlib import Path
        import re
        
        # Split comments function
        def _split_comments(comments):
            if pd.isna(comments):
                return []
            if isinstance(comments, list):
                return comments
            comments_str = str(comments)
            if '|' in comments_str:
                return [c.strip() for c in comments_str.split('|') if c.strip()]
            return [comments_str]
        
        # Clean text function
        def _clean_text(text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Add paths
        classifier_path = Path(__file__).parent / "video-quality-classifier"
        sys.path.append(str(classifier_path))
        
        # Import the model
        from video_quality_classifier.src.models.classifier import IntegratedQualityModel
        model_path = Path(__file__).parent / "models" / "best_integrated_model.pth"
        
        # Check if model exists
        if not os.path.exists(model_path):
            print("Integrated model not found, falling back to pattern-based approach")
            return df
        
        # Initialize model and tokenizer
        from transformers import AutoTokenizer
        
        model = IntegratedQualityModel()
        model.load_state_dict(torch.load(model_path))
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Set to evaluation mode
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Process videos
        results = []
        
        for _, row in df.iterrows():
            video_id = row['video_id']
            comments = _split_comments(row.get('top_comments', ''))
            
            # Clean and process each comment
            quality_sentiment_scores = []
            
            for comment in comments:
                if not comment or len(comment.strip()) < 3:
                    continue
                    
                # Clean comment
                cleaned_comment = _clean_text(comment)
                
                # Tokenize
                inputs = tokenizer(
                    cleaned_comment,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Get prediction
                with torch.no_grad():
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    
                    # Get unified quality+sentiment score
                    score = model(input_ids, attention_mask).item()
                    quality_sentiment_scores.append(score)
            
            # Calculate average score for this video
            if quality_sentiment_scores:
                avg_score = np.mean(quality_sentiment_scores)
            else:
                avg_score = 0.5  # Neutral if no comments
                
            results.append({
                'video_id': video_id,
                'quality_sentiment_score': avg_score,
                'comment_count': len(quality_sentiment_scores)
            })
        
        # Convert results to DataFrame and merge
        results_df = pd.DataFrame(results)
        df = df.merge(results_df, on='video_id', how='left')
        
        # Fill missing values
        df['quality_sentiment_score'] = df['quality_sentiment_score'].fillna(0.5)
        
        print(f"✅ Quality+Sentiment analyzer successfully applied to videos")
        return df
        
    except Exception as e:
        print(f"⚠️ Could not use quality+sentiment analyzer: {e}")
        print("Falling back to basic analysis")
        return df