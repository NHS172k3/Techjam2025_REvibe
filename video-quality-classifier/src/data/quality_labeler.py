import pandas as pd
import re
import numpy as np

def detect_quality_and_sentiment(df):
    """
    Detect both quality-related comments and general sentiment
    
    Args:
        df: DataFrame with comments
        
    Returns:
        df: DataFrame with quality and sentiment labels
    """
    # Quality-related keywords and patterns
    quality_patterns = {
        'positive': [
            # Production value positive
            r'(good|great|nice|amazing) (quality|production|editing|edit)',
            r'(well|beautifully|professionally) (made|edited|produced|shot|filmed)',
            r'(high|excellent) (quality|resolution|definition)',
            r'(beautiful|amazing|great) (shots|filming|camera work)',
            
            # Specific phrases
            r'chef\'s kiss',
            r'the vibes',
            r'the editing\?',
            r'crystal clear',
            r'smooth (transitions|editing)',
            r'(cinematic|theatrical)',
            r'this edit'
        ],
        'negative': [
            # Production value negative
            r'(bad|poor|terrible|awful) (quality|production|editing|edit)',
            r'(badly|poorly) (made|edited|produced|shot|filmed)',
            r'(low|poor) (quality|resolution|definition)',
            r'(shaky|blurry|choppy) (video|footage|camera)',
            
            # Specific phrases
            r'unwatchable',
            r'eyes need therapy',
            r'painful to (watch|sit through)',
            r'couldn\'t finish',
            r'had to scroll',
            r'potato (quality|camera)',
            r'filmed with a (potato|toaster)',
            r'my eyes hurt'
        ]
    }
    
    # General sentiment patterns
    sentiment_patterns = {
        'positive': [
            r'(love|like|enjoy|awesome|amazing|great|good|nice|best|perfect|fantastic)',
            r'(hilarious|funny|lol|lmao|rofl)',
            r'(favorite|fav)',
            r'(beautiful|gorgeous|stunning)',
            r'(subscribed|followed)',
            r'(made my day|needed this)',
            r'(chef\'s kiss|chef kiss)',
            r'(deserve more views|underrated)',
            r'(obsessed|addicted)'
        ],
        'negative': [
            r'(hate|dislike|awful|terrible|worst|bad|poor|garbage|trash)',
            r'(waste of time|wasted)',
            r'(cringe|cringing|cringey)',
            r'(delete this|take this down)',
            r'(unfollowed|unsubscribed)',
            r'(disappointed|disappointing)',
            r'(boring|bored|fell asleep)',
            r'(stop making|quit)',
            r'(regret watching|scrolling away)'
        ]
    }
    
    # Create compiled regex patterns
    quality_pos_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in quality_patterns['positive']]
    quality_neg_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in quality_patterns['negative']]
    
    sentiment_pos_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in sentiment_patterns['positive']]
    sentiment_neg_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in sentiment_patterns['negative']]
    
    # Function to analyze comment for both quality and sentiment
    def analyze_comment(row):
        comment = row['comment']
        
        if pd.isna(comment):
            return {
                'is_quality': False, 
                'quality_label': 1, 
                'sentiment_score': 0.5, 
                'unified_score': 0.5
            }
            
        comment = str(comment).lower()
        
        # Check for quality mentions
        quality_pos_matches = sum(1 for pattern in quality_pos_patterns if pattern.search(comment))
        quality_neg_matches = sum(1 for pattern in quality_neg_patterns if pattern.search(comment))
        
        # Check for sentiment signals
        sentiment_pos_matches = sum(1 for pattern in sentiment_pos_patterns if pattern.search(comment))
        sentiment_neg_matches = sum(1 for pattern in sentiment_neg_patterns if pattern.search(comment))
        
        # Determine if it's about quality
        is_quality = (quality_pos_matches > 0 or quality_neg_matches > 0)
        
        # Calculate overall sentiment score (0-1)
        total_sentiment = sentiment_pos_matches + sentiment_neg_matches
        if total_sentiment > 0:
            sentiment_score = sentiment_pos_matches / total_sentiment
        else:
            # If no clear sentiment, use original label if available
            if 'label' in row and row['label'] in ['positive', 'negative', 'neutral']:
                sentiment_score = 1.0 if row['label'] == 'positive' else (
                    0.0 if row['label'] == 'negative' else 0.5)
            else:
                sentiment_score = 0.5  # Neutral
        
        # Assign quality label (0=negative, 1=neutral/not about quality, 2=positive)
        if not is_quality:
            quality_label = 1  # Not about quality
        elif quality_pos_matches > quality_neg_matches:
            quality_label = 2  # Positive quality
        elif quality_neg_matches > quality_pos_matches:
            quality_label = 0  # Negative quality
        else:
            # If tied, use sentiment to determine
            quality_label = 2 if sentiment_score > 0.5 else (0 if sentiment_score < 0.5 else 1)
        
        # Create unified score that considers both quality and sentiment
        if is_quality:
            # For quality comments, give more weight to quality
            quality_sentiment = 1.0 if quality_label == 2 else (0.0 if quality_label == 0 else 0.5)
            unified_score = 0.7 * quality_sentiment + 0.3 * sentiment_score
        else:
            # For non-quality comments, rely more on general sentiment
            unified_score = sentiment_score
            
        return {
            'is_quality': is_quality,
            'quality_label': quality_label,
            'sentiment_score': sentiment_score,
            'unified_score': unified_score
        }
    
    # Apply the function to each row
    analysis_results = df.apply(analyze_comment, axis=1)
    
    # Add columns to dataframe
    df['is_quality_related'] = analysis_results.apply(lambda x: x['is_quality'])
    df['quality_label'] = analysis_results.apply(lambda x: x['quality_label'])
    df['sentiment_score'] = analysis_results.apply(lambda x: x['sentiment_score'])
    df['unified_score'] = analysis_results.apply(lambda x: x['unified_score'])
    
    # Print statistics
    quality_count = df['is_quality_related'].sum()
    print(f"Total comments: {len(df)}")
    print(f"Quality-related comments: {quality_count} ({quality_count/len(df)*100:.1f}%)")
    print(f"Positive quality: {sum(df['quality_label'] == 2)}")
    print(f"Negative quality: {sum(df['quality_label'] == 0)}")
    print(f"Not about quality: {sum(df['quality_label'] == 1)}")
    print(f"Average sentiment score: {df['sentiment_score'].mean():.2f}")
    print(f"Average unified score: {df['unified_score'].mean():.2f}")
    
    return df

if __name__ == "__main__":
    # Load the dataset
    csv_path = '../../data/raw/comments_dataset.csv'
    df = pd.read_csv(csv_path)
    
    # Process the dataset
    labeled_df = detect_quality_and_sentiment(df)
    
    # Save the labeled dataset
    output_path = '../../data/processed/quality_sentiment_labeled_comments.csv'
    labeled_df.to_csv(output_path, index=False)
    print(f"Labeled dataset saved to {output_path}")