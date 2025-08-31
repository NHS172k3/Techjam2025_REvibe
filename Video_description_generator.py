from google import genai
from google.genai.types import HttpOptions, Content, Part, Blob
import os
import pandas as pd
from dotenv import load_dotenv

def generate_video_description(video_file_name):
    """Generate description for a single video file"""
    load_dotenv()  # Load API key from .env file

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    # Create client
    client = genai.Client(api_key=api_key)
    
    # Optimized prompt for the pipeline
    prompt = """
    Analyze this video and provide a comprehensive description that includes:
    
    1. Main subjects and activities shown
    2. Setting and environment 
    3. Any educational or instructional content
    4. Skills, techniques, or knowledge being demonstrated
    5. Overall purpose and value of the content
    6. Any environmental or sustainability themes
    
    Focus on factual observations and the practical value the content provides to viewers. 
    Keep the description detailed but concise (4-6 sentences).
    """

    # Read video file
    with open(video_file_name, 'rb') as f:
        video_bytes = f.read()

    # Generate content
    response = client.models.generate_content(
        model='models/gemini-2.5-flash',
        contents=Content(
            parts=[
                Part(
                    inline_data=Blob(data=video_bytes, mime_type='video/mp4')
                ),
                Part(text=prompt)
            ]
        )
    )

    return response.text

def process_video_with_full_analysis(video_file_name):
    """Complete pipeline: generate description and run full analysis"""
    from socialgood_quality import score_file
    
    print(f"Generating description for {video_file_name}...")
    description = generate_video_description(video_file_name)
    
    print("Video Description Generated:")
    print("=" * 50)
    print(description)
    print("=" * 50)
    
    # Create DataFrame with the description as subtitles
    video_data = pd.DataFrame({
        'video_id': [video_file_name.replace('.mp4', '')],
        'subtitles': [description],
        'filename': [video_file_name],
        'user_id': ['new_user_001'],
        'views': [10000],
        'likes': [1200],
        'comment_count': [150],
        'shares': [80],
        'saved_count': [45],
        'viewer_retention_percentage': [75.5],
        'top_comments': ['This is so cute! | Love the cats | Amazing video quality | So adorable'],
        'caption': ['Cat lovers unite! #cats #pets #cute']
    })
    
    print("\nAnalyzing social good quality...")
    
    # Analyze social good quality
    scored_data = score_file(
        input_df=video_data,
        text_col="subtitles",
        nli_model="facebook/bart-large-mnli",
        device=-1
    )
    
    print("\nSocial Good Analysis Results:")
    print("=" * 50)
    
    result = scored_data.iloc[0]
    print(f"Educational Score: {result['score_educational']:.3f}")
    print(f"Environmental Score: {result['score_eco']:.3f}")
    print(f"Overall Social Value Score: {result['total_score']:.3f}")
    print(f"Social Value Multiplier: {result['social_value_multiplier']:.3f}")
    
    print("\nDetailed Breakdown:")
    print(f"  - Educational (Keyword): {result['score_edu_kw']:.3f}")
    print(f"  - Educational (Zero-shot): {result['score_edu_zs']:.3f}")
    print(f"  - Environmental (Keyword): {result['score_eco_kw']:.3f}")
    print(f"  - Environmental (Zero-shot): {result['score_eco_zs']:.3f}")
    
    # Save results
    output_file = f"{video_file_name.replace('.mp4', '')}_analysis_results.csv"
    scored_data.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return scored_data

if __name__ == "__main__":
    video_file_name = "#catloverlife.mp4"
    
    # Check if file exists
    if not os.path.exists(video_file_name):
        print(f"Error: Video file '{video_file_name}' not found!")
        exit(1)
    
    try:
        results = process_video_with_full_analysis(video_file_name)
    except Exception as e:
        print(f"Error processing video: {str(e)}")