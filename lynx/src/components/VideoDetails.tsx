import type { Video } from "../types/video.js";
import { distributionPictures } from "../assets/pictures.js";

export default function VideoDetails({
  video,
  setIndivVideoData,
}: {
  video: Video;
  setIndivVideoData: (video: Video | null) => void;
}) {
    
    const distImage = distributionPictures.find(p => p.name === video.fileName);
  
    return (
    <view style="padding: 16px; font-family: sans-serif; margin: 50px 0 8px;">
        <text style="margin: 0 0 12px 10px" bindtap={() => setIndivVideoData(null)}>← Back</text>
        <view style="display: flex; flex-direction: row; align-items: center; gap: 12px; margin-bottom: 16px;">
        <image
            src={video.thumbnail}
            style="width: 120px; height: 160px; object-fit: cover; border-radius: 8px;"
        />
        <view style="display: flex; flex-direction: column;">
            <text style="font-size: 16px; font-weight: bold;">{video.title}</text>
            <text style="font-size: 12px; color: #aaa;">Posted on: {video.datePosted}</text>
            <text style="font-size: 12px;">Category: {video.category}</text>
        </view>
    </view>

      <view
        style="display: flex; justify-content: space-around; margin-bottom: 16px;"
      >
        <text>👁️ {video.metrics.views.at(-1)}</text>
        <text>❤️ {video.metrics.likes.at(-1)}</text>
        <text>🔁 {video.metrics.shares.at(-1)}</text>
        <text>💬 {video.metrics.comments.at(-1)}</text>
      </view>

      <view class="card" style="padding: 12px; margin-bottom: 16px;">
        <text style="font-weight: bold;">Aggregate Score</text>
        <text style="font-size: 24px; margin-top: 8px;">{video.aggregateScore}</text>
        <view style="height: 200px; display: flex; align-items: center; justify-content: center;">
          <image 
            src={distImage?.src}
            mode="aspectFit" 
            style="width:400px;height:200px" 
          />
        </view>
        <text style="font-size: 12px; color: #aaa; text-align: center;">
          Your video surpasses {video.categoryPercentage}% of others in {video.category}.
        </text>
      </view>

      <view class="card" style="padding: 12px;">
        <text style="font-weight: bold;">Advanced Metrics</text>

        <view style="margin-top: 12px;">
          <text>📊 Engagement Z-score: {video.engagementZScore}</text>
        </view>
        <view>
          <text>🧠 Comment Sentiment: {video.commentSentimentMultiplier}</text>
        </view>
        <view>
          <text>🌍 Societal Impact: {video.societalImpactMultiplier}</text>
        </view>
        <view>
          <text>⭐ Aggregate Score: {video.aggregateScore}</text>
        </view>
        <view>
          <text>💵 Total Payout: ${video.totalPayout}</text>
        </view>
      </view>
    </view>
  );
}