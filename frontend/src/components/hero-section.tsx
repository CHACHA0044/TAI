"use client";

/* Breakpoints Addressed:
 * xs (below 640px): Stacked vertical layout, fluid fonts clamp(), min 44px touch targets.
 * sm/md: Flex row layout for CTA, scaled typography.
 * lg/xl: Restored full desktop typography and max width.
 */

import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { ShieldAlert, Play } from "lucide-react";
import { SectionWrapper } from "./section-wrapper";
import { Button } from "./ui/button";

export function HeroSection() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <div className="relative min-h-[90vh] flex items-center justify-center pt-24 pb-16 overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 z-0" aria-hidden="true">
        <div className="absolute inset-0 bg-gradient-to-b from-[#0a0a1f] via-[#080818] to-[#080818] z-10" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-emerald-900/20 via-[#080818]/80 to-[#080818] z-20" />
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-emerald-500/10 blur-[120px] rounded-full z-0 opacity-50 md:opacity-100 transition-opacity duration-500" />
        <div className="absolute top-1/3 left-1/4 w-[400px] h-[400px] bg-teal-500/10 blur-[120px] rounded-full z-0 opacity-50 md:opacity-100 transition-opacity duration-500" />
      </div>

      <SectionWrapper className="relative z-30 container w-full max-w-[1440px] mx-auto px-4 sm:px-6 text-center">
        <motion.div
          initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border-white/10 mb-6 sm:mb-8 will-change-transform"
        >
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
          </span>
          <span className="text-xs sm:text-sm font-semibold text-white/80 tracking-wide uppercase">
            TruthGuard AI Engine v2.0 Live
          </span>
        </motion.div>

        <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-black tracking-tight mb-6 sm:mb-8 leading-[1.1] sm:leading-[1.1]">
          Detect Misinformation
          <br className="hidden sm:block" />
          <span className="block sm:inline sm:ml-3 bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent transform-gpu">
            With Pinpoint Accuracy.
          </span>
        </h1>

        <p className="text-base sm:text-lg md:text-xl text-white/60 mb-10 max-w-2xl mx-auto font-medium leading-relaxed px-2">
          The ultimate multi-modal fake news and deepfake detection system.
          Analyze text, images, and videos using state-of-the-art AI.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-6 w-full px-4 sm:px-0 mt-4">
          <Button variant="primary" onClick={() => window.location.href = '/text'}>
            Explore System
          </Button>
          <Button variant="secondary" onClick={() => window.location.href = '#how-it-works'}>
            How it works
          </Button>
        </div>
      </SectionWrapper>
    </div>
  );
}
