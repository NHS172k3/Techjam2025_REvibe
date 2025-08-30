import { type Video } from "../types/video.js";

type NavBarProps = {
  currentPage: string;
  setCurrentPage: (page: string) => void;
  setIndivVideoData?: (page: Video | null) => void;
};

export default function NavBar(props: NavBarProps) {

    return (
        <view class="navbar">
            <view style="display: flex; justify-content: space-around; padding: 15px 0 30px; border-top: 1px solid #777777ff;">
                <text 
                    style={{
                        fontWeight: props.currentPage === "videos" ? "bold" : "normal"
                    }}
                    bindtap={() => props.setCurrentPage('videos')}
                >
                    📹 My Videos
                </text>
                <text
                    style={{
                        fontWeight: props.currentPage === "home" ? "bold" : "normal"
                    }}
                    bindtap={() => {
                        props.setCurrentPage('home');
                        if (props.setIndivVideoData) props.setIndivVideoData(null);
                    }
                }>🏠 Home</text>
                <text
                    style={{
                        fontWeight: props.currentPage === "revenue" ? "bold" : "normal"
                    }}
                    bindtap={() => props.setCurrentPage('revenue')}
                >
                    💰 Revenue
                </text>
            </view>
        </view>
    )
}