"use client";

/* Breakpoints Addressed:
 * xs/sm: Reduced card paddings, gaps, and icon sizes to prevent extreme text wrapping. Fluid typography added.
 * md+: Standard layout preserved (row flex, 2/3 column layout).
 */

import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { FileText, Image as ImageIcon, Video, ArrowRight } from "lucide-react";

export function SolutionSection() {
  const prefersReducedMotion = useReducedMotion();

  const features = [
    {
      id: "text",
      icon: FileText,
      title: "Text Analysis",
      desc: "Detect propaganda, bias, and AI-generated articles using NLP.",
      color: "from-emerald-500/20 to-teal-500/20",
      border: "hover:border-emerald-500/50",
      textGlow: "group-hover:text-emerald-400",
      href: "/text",
    },
    {
      id: "image",
      icon: ImageIcon,
      title: "Image Forensics",
      desc: "Identify GAN anomalies, diffusion traces, and manipulated pixels.",
      color: "from-blue-500/20 to-cyan-500/20",
      border: "hover:border-blue-500/50",
      textGlow: "group-hover:text-blue-400",
      href: "/image",
    },
    {
      id: "video",
      icon: Video,
      title: "Deepfake Detection",
      desc: "Analyze micro-expressions, sync, and frame inconsistencies.",
      color: "from-purple-500/20 to-pink-500/20",
      border: "hover:border-purple-500/50",
      textGlow: "group-hover:text-purple-400",
      href: "/video",
    },
  ];

  return (
    <section className="py-16 sm:py-24 relative z-10 bg-[#0a0a1a]">
      <SectionWrapper className="container w-full max-w-5xl mx-auto px-4 sm:px-6">
        <div className="flex flex-col md:flex-row gap-8 md:gap-12 items-center md:items-start">
          <div className="w-full md:w-1/3 text-center md:text-left">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-black mb-4 sm:mb-6 leading-tight">
              One Platform. <br className="hidden md:block" />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-400 transform-gpu">
                Multi-Modal Truth.
              </span>
            </h2>
            <p className="text-white/50 text-base sm:text-lg mb-8 leading-relaxed max-w-md mx-auto md:mx-0">
              TruthGuard AI is the unified solution to digital deception. Our
              models analyze content across all major mediums simultaneously.
            </p>
          </div>

          <div className="w-full md:w-2/3 grid gap-4">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <Link 
                  key={feature.id} 
                  href={feature.href}
                  className="rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 focus-visible:ring-offset-4 focus-visible:ring-offset-[#0a0a1a] block"
                >
                  <motion.div
                    initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true, margin: "-50px" }}
                    transition={{ duration: 0.4, delay: prefersReducedMotion ? 0 : i * 0.1 }}
                    className={`group relative glass p-4 sm:p-6 rounded-2xl border border-white/5 ${feature.border} transition-all duration-300 flex items-center gap-3 sm:gap-6 overflow-hidden cursor-pointer`}
                  >
                    <div
                      className={`absolute inset-0 bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-10 transition-opacity aria-hidden="true"`}
                    />
                    <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-xl bg-white/5 flex items-center justify-center shrink-0">
                      <Icon className={`w-6 h-6 sm:w-8 sm:h-8 text-white/50 ${feature.textGlow} transition-colors`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className={`text-lg sm:text-xl font-bold mb-0.5 sm:mb-1 text-white ${feature.textGlow} transition-colors truncate`}>
                        {feature.title}
                      </h3>
                      <p className="text-[13px] sm:text-sm text-white/50 group-hover:text-white/70 transition-colors leading-relaxed line-clamp-2 sm:line-clamp-none">
                        {feature.desc}
                      </p>
                    </div>
                    <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full border border-white/10 flex items-center justify-center group-hover:bg-white group-hover:text-black transition-colors shrink-0">
                      <ArrowRight className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                    </div>
                  </motion.div>
                </Link>
              );
            })}
          </div>
        </div>
      </SectionWrapper>
    </section>
  );
}
