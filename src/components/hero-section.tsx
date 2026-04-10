"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ShieldAlert, ArrowRight, Play } from "lucide-react";
import { SectionWrapper } from "./section-wrapper";

export function HeroSection() {
  return (
    <div className="relative min-h-[90vh] flex items-center justify-center pt-24 overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a1f] via-[#080818] to-[#080818] z-10" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-900/20 via-[#080818]/80 to-[#080818] z-20" />
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-blue-500/10 blur-[120px] rounded-full z-0" />
        <div className="absolute top-1/3 left-1/4 w-[400px] h-[400px] bg-purple-500/10 blur-[120px] rounded-full z-0" />
      </div>

      <SectionWrapper className="relative z-30 container max-w-5xl mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border-white/10 mb-8"
        >
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500"></span>
          </span>
          <span className="text-xs font-semibold text-white/80 tracking-wide uppercase">
            TruthGuard AI Engine v2.0 Live
          </span>
        </motion.div>

        <h1 className="text-5xl md:text-7xl font-black tracking-tight mb-8 leading-[1.1]">
          Detect Misinformation
          <br />
          <span className="text-gradient">With Pinpoint Accuracy.</span>
        </h1>

        <p className="text-lg md:text-xl text-white/60 mb-12 max-w-2xl mx-auto font-medium leading-relaxed">
          The ultimate multi-modal fake news and deepfake detection system.
          Analyze text, images, and videos using state-of-the-art AI.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/text"
            className="group relative px-8 py-4 w-full sm:w-auto rounded-2xl bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold text-lg hover:shadow-[0_0_40px_rgba(59,130,246,0.5)] transition-all duration-300 flex items-center justify-center gap-3 overflow-hidden"
          >
            <div className="absolute inset-0 bg-white/20 group-hover:translate-x-full transition-transform duration-500 -translate-x-full skew-x-12" />
            <ShieldAlert className="w-5 h-5 relative z-10" />
            <span className="relative z-10">Explore System</span>
          </Link>
          <Link
            href="#how-it-works"
            className="px-8 py-4 w-full sm:w-auto rounded-2xl glass-strong text-white font-semibold text-lg hover:bg-white/10 transition-all flex items-center justify-center gap-3"
          >
            <Play className="w-5 h-5 text-white/70" />
            How it works
          </Link>
        </div>
      </SectionWrapper>
    </div>
  );
}
