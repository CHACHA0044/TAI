import type { Metadata, Viewport } from "next";
import "animate.css";
import "tippy.js/dist/tippy.css";
import "./globals.css";
import { Navbar } from "@/components/navbar";
import { Starfield } from "@/components/starfield";

export const viewport: Viewport = {
  themeColor: "#080818",
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  title: {
    default: "TruthGuard AI – Multi-Modal Forensic Intelligence",
    template: "%s | TruthGuard AI",
  },
  description:
    "The world's most advanced multi-modal forensic intelligence system. Detect deepfakes and misinformation across text, images, and videos.",
  metadataBase: new URL("https://truthguard.ai"),
  openGraph: {
    title: "TruthGuard AI",
    description: "Advanced Multi-Modal Forensic Intelligence",
    type: "website",
    siteName: "TruthGuard AI",
  },
  twitter: {
    card: "summary_large_image",
    title: "TruthGuard AI",
    description: "Advanced Multi-Modal Forensic Intelligence",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth" suppressHydrationWarning>
      <head>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@400;500;600;700;800;900&display=swap" />
      </head>
      <body className="min-h-screen bg-[#080818] text-white antialiased overflow-x-hidden font-sans">
        <Starfield />
        <Navbar />
        <main className="relative z-10">{children}</main>
        
        {/* Global Ambient Glows */}
        <div className="fixed top-0 left-0 w-full h-full pointer-events-none z-0 overflow-hidden">
          <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-emerald-500/5 blur-[120px] rounded-full" />
          <div className="absolute -bottom-[10%] -right-[10%] w-[40%] h-[40%] bg-teal-500/5 blur-[120px] rounded-full" />
        </div>
      </body>
    </html>
  );
}
