"use client";

import { motion } from "framer-motion";
import { GridBackground } from "./grid-background";
import { SectionWrapper } from "./section-wrapper";
import { Button } from "./ui/button";
import { ShieldCheck, Zap, Globe, Lock } from "lucide-react";

export function HeroSection() {
  return (
    <div className="relative min-h-screen flex items-center justify-center pt-32 pb-20 overflow-hidden">
      <GridBackground />

      <SectionWrapper className="relative z-10 container mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="inline-flex items-center gap-2.5 px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 mb-10 shadow-[0_0_20px_rgba(16,185,129,0.1)]"
        >
          <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_#10b981]" />
          <span className="text-xs font-bold text-emerald-400 tracking-[0.2em] uppercase">
            TruthGuard AI — Enterprise Forensics v2.0
          </span>
        </motion.div>

        <motion.h1 
          className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-black tracking-tighter mb-8 leading-[0.95] text-white"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Detect MisInformation
          <br />
          <span className="bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-500 bg-clip-text text-transparent">
            With PinPoint Accuracy.
          </span>
        </motion.h1>

        <motion.p 
          className="text-lg md:text-xl text-white/50 mb-12 max-w-3xl mx-auto font-medium leading-relaxed"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          The world's most advanced multi-modal forensic intelligence system. 
          Detect deepfakes, neutralize fake news, and verify authenticity across 
          text, image, and video with sub-pixel precision.
        </motion.p>

        <motion.div 
          className="flex flex-col sm:flex-row items-center justify-center gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <Button variant="primary" className="px-10 h-14" onClick={() => window.location.href = '/text'}>
            Deploy Analysis
          </Button>
          <Button variant="secondary" className="px-10 h-14" onClick={() => window.location.href = '#how-it-works'}>
            System Architecture
          </Button>
        </motion.div>

        {/* Feature Tags */}
        <motion.div 
          className="mt-20 flex flex-wrap justify-center gap-8 md:gap-16 opacity-40 hover:opacity-100 transition-opacity duration-500"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.4 }}
          transition={{ duration: 1, delay: 0.6 }}
        >
          <div className="flex items-center gap-3">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-widest">Real-time Verification</span>
          </div>
          <div className="flex items-center gap-3">
            <Zap className="w-5 h-5 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-widest">Neural Analysis</span>
          </div>
          <div className="flex items-center gap-3">
            <Globe className="w-5 h-5 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-widest">Global Intelligence</span>
          </div>
          <div className="flex items-center gap-3">
            <Lock className="w-5 h-5 text-emerald-400" />
            <span className="text-xs font-bold uppercase tracking-widest">Secure & Private</span>
          </div>
        </motion.div>
      </SectionWrapper>
    </div>
  );
}
