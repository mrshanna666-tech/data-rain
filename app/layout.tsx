import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DATA RAIN — 实时降雨可视化",
  description: "一场由 GPU 实时计算、自动适配设备性能的沉浸式数据暴雨。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
  openGraph: {
    title: "DATA RAIN",
    description: "GPU 实时降雨可视化",
    images: ["/og.png"],
  },
  twitter: {
    card: "summary_large_image",
    title: "DATA RAIN",
    description: "GPU 实时降雨可视化",
    images: ["/og.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
