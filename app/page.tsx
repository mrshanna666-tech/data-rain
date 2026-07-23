import type { Metadata } from "next";
import RainExperience from "./rain-experience";

export const metadata: Metadata = {
  title: "DATA RAIN — 实时降雨可视化",
  description: "一场由 GPU 实时计算的沉浸式数据暴雨。",
};

export default function Home() {
  return <RainExperience />;
}
