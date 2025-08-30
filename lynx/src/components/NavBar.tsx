type NavBarProps = {
  setCurrentPage: (page: string) => void;
};

export default function NavBar(props: NavBarProps) {

    return (
        <view class="navbar">
            <view style="display: flex; justify-content: space-around; padding: 15px 0 30px; border-top: 1px solid #777777ff;">
                <text bindtap={() => props.setCurrentPage('videos')}>📹 My Videos</text>
                <text style="font-weight: bold;" bindtap={() => props.setCurrentPage('home')}>🏠 Home</text>
                <text  bindtap={() => props.setCurrentPage('revenue')}>💰 Revenue</text>
            </view>
        </view>
    )
}