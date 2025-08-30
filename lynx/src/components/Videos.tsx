import { useState } from "@lynx-js/react/legacy-react-runtime";
import { videoData } from "../../demo-data.js";
import VideoCard from "./VideoCard.js";
import { type Video } from "../types/video.js";
import VideoDetails from "./VideoDetails.js";

export default function VideosPage() {

  const [indivVideoData, setIndivVideoData] = useState<Video | null>(null);

  return (
    <view>
        {
            indivVideoData === null &&
            <view style="padding: 16px; font-family: sans-serif;">
                <text style="font-size: 24px; font-weight: bold; margin: 50px 0 8px;">
                    🎬 Your Videos
                </text>

                {videoData.map((video) => (
                    <VideoCard video={video} setIndivVideoData={setIndivVideoData} />
                ))}
            </view>
        }

        {
            indivVideoData != null &&
            <VideoDetails video={indivVideoData} setIndivVideoData={setIndivVideoData} />
        }
    </view>

    
    
  );
}