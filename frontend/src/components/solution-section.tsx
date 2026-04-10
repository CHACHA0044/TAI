"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { FileText, Image as ImageIcon, Video, ArrowRight } from "lucide-react";

export function SolutionSection() {
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
    <section className="py-24 relative z-10 bg-[#0a0a1a]">
      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="flex flex-col md:flex-row gap-12 items-center">
          <div className="w-full md:w-1/3">
            <h2 className="text-3xl md:text-5xl font-black mb-6 leading-tight">
              One Platform. <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
                Multi-Modal Truth.
              </span>
            </h2>
            <p className="text-white/50 text-lg mb-8 leading-relaxed">
              TruthGuard AI is the unified solution to digital deception. Our
              models analyze content across all major mediums simultaneously.
            </p>
          </div>

          <div className="w-full md:w-2/3 grid gap-4">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <Link key={feature.id} href={feature.href}>
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: i * 0.1 }}
                    className={`group relative glass p-6 rounded-2xl border border-white/5 ${feature.border} transition-all duration-300 flex items-center gap-6 overflow-hidden cursor-pointer`}
                  >
                    <div
                      className={`absolute inset-0 bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-10 transition-opacity`}
                    />
                    <div className="w-16 h-16 rounded-xl bg-white/5 flex items-center justify-center shrink-0">
                      <Icon className={`w-8 h-8 text-white/50 ${feature.textGlow} transition-colors`} />
                    </div>
                    <div className="flex-1">
                      <h3 className={`text-xl font-bold mb-1 text-white ${feature.textGlow} transition-colors`}>
                        {feature.title}
                      </h3>
                      <p className="text-sm text-white/50 group-hover:text-white/70 transition-colors">
                        {feature.desc}
                      </p>
                    </div>
                    <div className="w-10 h-10 rounded-full border border-white/10 flex items-center justify-center group-hover:bg-white group-hover:text-black transition-colors shrink-0">
                      <ArrowRight className="w-4 h-4" />
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
