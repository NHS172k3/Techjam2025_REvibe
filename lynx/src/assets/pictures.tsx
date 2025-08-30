import cat1_dist_23_0 from "./cat1_dist_23_0.png";
import cat1_dist_86_9 from "./cat1_dist_86_9.png";
import cat3_dist_71_0 from "./cat3_dist_71_0.png";

export interface Picture {
    src: string;
    name: string;
}

export const distributionPictures: Picture[] = [
  {
    name: "cat1_dist_23_0",
    src: cat1_dist_23_0,
  },
  {
    name: "cat1_dist_86_9",
    src: cat1_dist_86_9,
  },
  {
    name: "cat3_dist_71_0",
    src: cat3_dist_71_0,
  },
];