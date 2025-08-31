import streamlit as st
import pandas as pd
import json
import os
from main import run_analysis

st.set_page_config(page_title="TikTok Video Analyzer", layout="wide")

st.title("🎥 TikTok Video Analyzer & Profit Sharing Calculator")
st.markdown("Upload your video and enter metrics to see how much you'd earn based on sentiment and social good!")

# Create two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📱 Upload Video & Enter Metrics")
    
    # Video upload
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    
    # Save uploaded video to disk if provided
    video_filename = ""
    if video_file is not None:
        video_filename = video_file.name
        with open(video_filename, "wb") as f:
            f.write(video_file.getbuffer())
        st.success(f"✅ Video '{video_filename}' uploaded successfully!")
    
    # Metrics entry
    st.subheader("Enter Video Metadata & Metrics")
    
    # Basic info
    video_id = st.text_input("Video ID", value=f"user_video_{hash(video_filename) % 10000}" if video_filename else "user_video_001")
    user_id = st.text_input("User ID", value="new_user_001")
    
    # Engagement metrics
    st.subheader("📊 Engagement Metrics")
    col_left, col_right = st.columns(2)
    
    with col_left:
        views = st.number_input("Views", min_value=0, value=10000, step=100)
        likes = st.number_input("Likes", min_value=0, value=1200, step=10)
        comment_count = st.number_input("Comment Count", min_value=0, value=150, step=5)
    
    with col_right:
        shares = st.number_input("Shares", min_value=0, value=80, step=5)
        saved_count = st.number_input("Saved Count", min_value=0, value=45, step=5)
        viewer_retention = st.slider("Viewer Retention (%)", min_value=0.0, max_value=100.0, value=75.5, step=0.1)
    
    # Content
    st.subheader("📝 Content Information")
    captions = st.text_area("Captions", value="Check out this amazing content! #viral #fyp", height=60)

    # Description source choice
    if video_file is not None:
        description_choice = st.radio(
            "Description Source:",
            ["🤖 Auto-generate from video (AI)", "✍️ Use my custom description", "🔄 Combine both"],
            index=0,
            help="Choose how to create the video description"
        )
        
        if description_choice == "✍️ Use my custom description":
            subtitles = st.text_area("Custom Description", 
                                    value="This video shows interesting content that viewers will enjoy.", 
                                    height=80)
            use_ai_description = False
            combine_descriptions = False
        elif description_choice == "🔄 Combine both":
            subtitles = st.text_area("Your Description (will be combined with AI analysis)", 
                                    value="This video shows interesting content that viewers will enjoy.", 
                                    height=80)
            use_ai_description = True
            combine_descriptions = True
        else:  # AI only
            subtitles = st.text_area("Description Preview", 
                                    value="AI will analyze your video and generate this automatically...", 
                                    height=80,
                                    disabled=True)
            use_ai_description = True
            combine_descriptions = False
    else:
        subtitles = st.text_area("Subtitles/Description", 
                                value="This video shows interesting content that viewers will enjoy.", 
                                height=80)
        use_ai_description = False
        combine_descriptions = False
    
    top_comments = st.text_area("Top Comments (separate with |)", 
                               value="This is amazing! | Love this content | Great video | So cool! | Awesome work",
                               height=100,
                               help="Enter comments separated by | symbol")

# Submit button
if st.button("🚀 Analyze Video & Calculate Earnings", type="primary"):
    with st.spinner("🤖 Analyzing your video with AI..."):
        
        # Prepare data for analysis
        record = {
            "video_id": video_id,
            "user_id": user_id,
            "views": views,
            "likes": likes,
            "comment_count": comment_count,
            "shares": shares,
            "saved_count": saved_count,
            "viewer_retention_percentage": viewer_retention,
            "captions": captions,
            "subtitles": subtitles,
            "top_comments": top_comments,
            "video_file": video_filename,
            "use_ai_description": use_ai_description,      # ← Add these flags
            "combine_descriptions": combine_descriptions   # ← Add these flags
        }
        
        # Save to JSON for main.py to process
        with open("to_insert.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        # Run the analysis
        try:
            df_final, allocations = run_analysis()
            
            # Get results for the uploaded video (last row)
            if len(df_final) > 0:
                your_video = df_final[df_final['video_id'] == video_id]
                if your_video.empty:
                    # If exact match not found, get last row (most recent)
                    your_video = df_final.tail(1)
                
                video_info = your_video.iloc[0]
                
                # Display results in second column
                with col2:
                    st.header("📊 Your Video Analysis Results")
                    
                    # Earnings highlight
                    earnings = video_info.get('video_money', 0)
                    st.metric("💰 Estimated Earnings", f"${earnings:.2f}")
                    
                    # Create tabs for different result views
                    tab1, tab2, tab3 = st.tabs(["📈 Scores", "🏆 Performance", "📋 Details"])
                    
                    with tab1:
                        st.subheader("📈 AI Analysis Scores")
                        
                        # Sentiment analysis
                        if 'sentiment_score' in video_info:
                            sentiment_score = video_info['sentiment_score']
                            sentiment_interpretation = video_info.get('sentiment_interpretation', 'N/A')
                            
                            st.metric("😊 Sentiment Score", f"{sentiment_score:.3f}/1.0", 
                                     help="Based on comment analysis (0=very negative, 1=very positive)")
                            st.write(f"**Interpretation:** {sentiment_interpretation}")
                            
                            # Sentiment progress bar
                            st.progress(sentiment_score)
                            
                            # Comments analyzed
                            comments_analyzed = video_info.get('analyzed_comments_count', 0)
                            st.write(f"📝 Comments Analyzed: {comments_analyzed}")
                        
                        # Social good analysis
                        if 'total_score' in video_info:
                            social_score = video_info['total_score']
                            st.metric("🌍 Social Good Score", f"{social_score:.3f}/1.0",
                                     help="Educational and environmental content value")
                            st.progress(social_score)
                        
                        # Combined multiplier
                        if 'combined_multiplier' in video_info:
                            multiplier = video_info['combined_multiplier']
                            st.metric("🔧 Earning Multiplier", f"{multiplier:.3f}x",
                                     help="Combined sentiment + social good multiplier (0.9-1.1 range)")
                    
                    with tab2:
                        st.subheader("🏆 Performance Metrics")
                        
                        # Cluster info
                        cluster = video_info.get('cluster', 'N/A')
                        st.write(f"🎯 **Content Category:** {cluster}")
                        
                        # Ranking
                        if 'video_money' in df_final.columns:
                            rank = (df_final['video_money'] > earnings).sum() + 1
                            total_videos = len(df_final)
                            percentile = (total_videos - rank + 1) / total_videos * 100
                            
                            st.metric("🏆 Rank", f"#{rank} out of {total_videos}")
                            st.metric("📊 Percentile", f"{percentile:.1f}%")
                        
                        # Engagement rate calculation
                        engagement_rate = (likes + comment_count + shares) / max(views, 1) * 100
                        st.metric("📱 Engagement Rate", f"{engagement_rate:.2f}%")
                        
                        # Video score
                        if 'video_score' in video_info:
                            video_score = video_info['video_score']
                            st.metric("⭐ Overall Video Score", f"{video_score:.3f}/1.0")
                    
                    with tab3:
                        st.subheader("📋 Detailed Breakdown")
                        
                        # Create a clean summary table
                        summary_data = {
                            "Metric": [],
                            "Value": []
                        }
                        
                        # Add key metrics
                        metrics_to_show = [
                            ("Video ID", video_info.get('video_id', 'N/A')),
                            ("Views", f"{views:,}"),
                            ("Likes", f"{likes:,}"),
                            ("Comments", f"{comment_count:,}"),
                            ("Shares", f"{shares:,}"),
                            ("Retention", f"{viewer_retention:.1f}%"),
                            ("Sentiment Score", f"{video_info.get('sentiment_score', 0):.3f}"),
                            ("Social Good Score", f"{video_info.get('total_score', 0):.3f}"),
                            ("Final Multiplier", f"{video_info.get('combined_multiplier', 1):.3f}"),
                            ("Estimated Earnings", f"${earnings:.2f}")
                        ]
                        
                        for metric, value in metrics_to_show:
                            summary_data["Metric"].append(metric)
                            summary_data["Value"].append(value)
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Success message
                st.success("✅ Analysis complete! Your video has been processed and earnings calculated.")
                
                # Insights based on results
                st.subheader("💡 AI Insights & Recommendations")
                
                if 'sentiment_score' in video_info:
                    sent = video_info['sentiment_score']
                    if sent > 0.8:
                        st.info("🔥 **Excellent!** Your comments are overwhelmingly positive! This content resonates very well with viewers.")
                    elif sent > 0.6:
                        st.info("😊 **Good!** Your comments are generally positive. Consider encouraging more engagement.")
                    elif sent > 0.4:
                        st.warning("😐 **Neutral** Comment sentiment is mixed. Try creating more engaging content.")
                    else:
                        st.warning("😞 **Needs Work** Comment sentiment is negative. Consider revising your content strategy.")
                
                if 'total_score' in video_info:
                    social = video_info['total_score']
                    if social > 0.7:
                        st.info("🌍 **Great Impact!** Your content has high social value with educational or environmental benefits.")
                    elif social > 0.4:
                        st.info("🌱 **Some Impact** Your content has moderate social value. Consider adding more educational elements.")
                    else:
                        st.info("📚 **Growth Opportunity** Consider adding educational or environmental themes to increase social impact.")
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.write("Please check your inputs and try again.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
    **Our AI analyzes your video using:**
    
    1. **🤖 Sentiment Analysis**
       - Analyzes comments to determine viewer sentiment
       - Scores from 0 (very negative) to 1 (very positive)
    
    2. **🌍 Social Good Analysis** 
       - Evaluates educational and environmental content
       - Rewards content that benefits society
    
    3. **📊 Engagement Metrics**
       - Views, likes, comments, shares, retention
       - Combined with AI scores for final earnings
    
    4. **💰 Earnings Calculation**
       - Base earnings + multipliers
       - Range: 0.9x to 1.1x based on AI analysis
       - Fair distribution across content categories
    """)
    
    st.header("🎯 Tips for Higher Earnings")
    st.markdown("""
    - **Create positive content** that generates good comments
    - **Add educational value** to your videos
    - **Include environmental themes** when relevant
    - **Encourage engagement** through calls-to-action
    - **Maintain high retention** with compelling content
    """)
