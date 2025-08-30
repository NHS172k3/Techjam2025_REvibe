export default function VideoDetails({ video }: { video: any }) {
  return (
    <view style="padding: 16px; font-family: sans-serif; margin: 50px 0 8px;">
        <view style="display: flex; flex-direction: row; align-items: center; gap: 12px; margin-bottom: 16px;">
        <image
            src={video.thumbnail}
            style="width: 120px; height: 160px; object-fit: cover; border-radius: 8px;"
        />
        <view style="display: flex; flex-direction: column;">
            <text style="font-size: 16px; font-weight: bold;">{video.title}</text>
            <text style="font-size: 12px; color: #aaa;">Posted on: {video.datePosted}</text>
            <text style="font-size: 12px;">Category: Education</text>
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
        <text style="font-size: 24px; margin-top: 8px;">87</text>
        <view style="height: 100px; display: flex; align-items: center; justify-content: center;">
          <text>[Graph Placeholder]</text>
        </view>
        <text style="font-size: 12px; color: #aaa; text-align: center;">
          Your video surpasses 90% of others in Education.
        </text>
      </view>

      <view class="card" style="padding: 12px;">
        <text style="font-weight: bold;">Advanced Metrics</text>

        <view style="margin-top: 12px;">
          <text>📊 Engagement Z-score: 1.42</text>
        </view>
        <view>
          <text>🧠 Comment Sentiment: Positive</text>
        </view>
        <view>
          <text>🌍 Societal Impact: Moderate</text>
        </view>
        <view>
          <text>⭐ Aggregate Score: 87</text>
        </view>
        <view>
          <text>💵 Total Payout: $36.28</text>
        </view>
      </view>
    </view>
  );
}