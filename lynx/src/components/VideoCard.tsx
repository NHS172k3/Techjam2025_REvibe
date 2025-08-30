import { type Video } from "../types/video.js";

interface VideoCardProps {
  video: Video;
  setIndivVideoData: (video: Video) => void;
}

export default function VideoCard({ video, setIndivVideoData }: VideoCardProps) {

  return (
    <view
      class="card"
      style="margin-bottom: 16px; display: flex; flex-direction: row; gap: 16px;"
      bindtap={() => setIndivVideoData(video)}
    >
      <image
        src={video.thumbnail}
        style="width: 60px; height: 80px; object-fit: cover; border-radius: 8px;"
      />

      <view style="flex: 1;">
        <text style="font-weight: bold; font-size: 16px; margin-bottom: 4px;">
          {video.title}
        </text>
        <text style="font-size: 12px; color: #aaa; margin-bottom: 8px;">
          Posted: {video.datePosted}
        </text>

        <view style="display: flex; justify-content: space-between; flex-wrap: wrap;">
          <text>👁️ {video.metrics.views.at(-1)}</text>
          <text>❤️ {video.metrics.likes.at(-1)}</text>
          <text>💬 {video.metrics.comments.at(-1)}</text>
          <text>🔁 {video.metrics.shares.at(-1)}</text>
        </view>
      </view>
    </view>
  );
}