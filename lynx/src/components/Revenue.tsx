import { videoData } from "../../demo-data.js";

export default function RevenuePage() {
  const totalPayout = videoData
    .map(video => video.totalPayout || 0)
    .reduce((sum, val) => sum + val, 0);

  return (
    <view style="padding: 24px; font-family: sans-serif; color: white; margin: 50px 0">
      <view style="margin-bottom: 30px;">
        <text style="font-size: 24px; font-weight: bold;">Total Revenue</text>
        <text style="display: block; font-size: 20px; margin-top: 10px;">
          ${totalPayout.toFixed(2)}
        </text>
      </view>

      <view>
        <text style="font-size: 20px; font-weight: bold; margin-bottom: 16px;">Payout by Video</text>
        {videoData.map(video => (
          <view key={video.title} style="margin-bottom: 20px; padding: 12px; border: 1px solid #555; border-radius: 8px;">
            <view style="display: flex; align-items: center;">
              <image src={video.thumbnail} mode="aspectFit" style="width: 80px; height: 50px; border-radius: 4px; margin-right: 12px;" />
              <view>
                <text style="font-size: 16px; font-weight: bold;">{video.title}</text>
                <text style="display: block; font-size: 14px; margin-top: 4px;">Posted: {video.datePosted}</text>
              </view>
            </view>
            <text style="display: block; margin-top: 8px; font-size: 16px;">
              Payout: ${video.totalPayout.toFixed(2)}
            </text>
          </view>
        ))}
      </view>
    </view>
  );
}