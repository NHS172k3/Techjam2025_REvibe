export interface VideoMetrics {
  views: number[];
  likes: number[];
  shares: number[];
  comments: number[];
}

export interface Video {
  title: string;
  thumbnail: string;
  datePosted: string;
  metrics: VideoMetrics;
}