import graph from "../assets/overall_chart.png";

export default function HomePage() {

    return (
        <view style="padding: 16px; font-family: sans-serif;">

            <text style="font-size: 24px; font-weight: bold; margin: 50px 0 8px;">
                Welcome back, WH!
            </text>

            <view class="card" style="display: flex; justify-content: space-between;">
                <text style="font-size: 18px; margin-bottom: 16px;">
                    👁️‍🗨️ Views:
                </text>
                <text style="font-size: 30px; margin-bottom: 4px;">
                    20,050
                </text>
            </view>

            

            <view style="display: flex; justify-content: space-between; gap: 12px;">

                <view class="card" style="flex: 1;">
                    <text style="font-weight: bold;">❤️ Likes</text>
                    <text style="margin-top: 8px;">12.4k</text>
                </view>

                <view class="card" style="flex: 1;">
                    <text style="font-weight: bold;">↱ Shares</text>
                    <text style="margin-top: 8px;">3.1k</text>
                </view>

                <view class="card" style="flex: 1;">
                    <text style="font-weight: bold;">💬 Comment</text>
                    <text style="margin-top: 8px;">980</text>
                </view>

            </view>

            <view class="card" style="height: 220px; display: flex; align-items: center; justify-content: center;">
                <image src={graph} mode="aspectFit" style="width:400px;height:200px" />
            </view>

        </view>
    )

}