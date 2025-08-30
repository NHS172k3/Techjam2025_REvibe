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
  category: string;
  categoryPercentage: number;
  filePath: string;
  metrics: VideoMetrics;
  engagementZScore: number;
  commentSentimentMultiplier: number;
  societalImpactMultiplier: number;
  aggregateScore: number;
  totalPayout: number;
}