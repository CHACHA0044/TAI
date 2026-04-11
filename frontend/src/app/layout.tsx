import type { Metadata } from "next";
import "animate.css";
import "tippy.js/dist/tippy.css";
import "./globals.css";
import { Navbar } from "@/components/navbar";

export const metadata: Metadata = {
  title: {
    default: "TruthGuard AI – Multi-Modal Fake News & Deepfake Detection",
    template: "%s | TruthGuard AI",
  },
  description:
    "TruthGuard AI detects fake news and deepfakes across text, images, and videos using advanced multi-modal AI technology.",
  keywords: [
    "fake news detection",
    "deepfake detection",
    "AI misinformation",
    "media verification",
    "TruthGuard",
  ],
  openGraph: {
    title: "TruthGuard AI",
    description: "Multi-Modal Fake News & Deepfake Detection System",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark font-sans" suppressHydrationWarning>
      <head>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" />
      </head>
      <body className="min-h-screen bg-[#080818] text-white antialiased overflow-x-hidden">
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
