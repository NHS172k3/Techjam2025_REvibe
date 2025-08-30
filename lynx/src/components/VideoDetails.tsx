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

        <view style="padding: 16px; border-radius: 8px; background-color: rgb(33,33,33); text-align: center;">

            <view style="display: flex; justify-content: space-around; margin-top: 12px;">
                <view style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                    <text style="font-size: 20px;">{video.engagementZScore} ×</text>
                    <text></text>
                    <text style="display: block; font-weight: bold; font-size: 10px;">Engagement</text>
                    <text style="font-size: 10px;">Score</text>
                </view>
                <view style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                    <text style="font-size: 20px;">{video.commentSentimentMultiplier} ×</text>
                    <text></text>
                    <text style="display: block; font-weight: bold; font-size: 10px;">Comment</text>
                    <text style="font-size: 10px;">Sentiment</text>
                </view>
                <view style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                    <text style="font-size: 20px;">{video.societalImpactMultiplier} =</text>
                    <text></text>
                    <text style="display: block; font-weight: bold; font-size: 10px;">Societal</text>
                    <text style="font-size: 10px;">Impact</text>
                </view>
                <view style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                    <text style="font-size: 20px; font-weight: bold">{video.aggregateScore}</text>
                    <text></text>
                    <text style="display: block; font-weight: bold; font-size: 10px;">Aggregate</text>
                    <text style="font-size: 10px; font-weight: bold;">Score</text>
                </view>
            </view>

            

        </view>

        <text style="margin: 10px 0px; font-weight: bold;">Total Payout: ${video.totalPayout.toFixed(2)}</text>
        
      </view>
    </view>
  );
}