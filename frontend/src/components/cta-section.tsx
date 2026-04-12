"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { FileText, Image as ImageIcon, Video, ArrowUpRight } from "lucide-react";

export function CtaSection() {
  const cards = [
    {
      href: "/text",
      icon: FileText,
      title: "Tactical Text",
      desc: "Neutralize propaganda and verify semantic integrity in articles & URLs.",
      color: "from-emerald-500 to-teal-500",
      accent: "shadow-emerald-500/10"
    },
    {
      href: "/image",
      icon: ImageIcon,
      title: "Visual Forensics",
      desc: "Expose GAN artifacts and diffusion traces in high-resolution imagery.",
      color: "from-teal-500 to-cyan-500",
      accent: "shadow-teal-500/10"
    },
    {
      href: "/video",
      icon: Video,
      title: "Deepfake Radar",
      desc: "Analyze temporal consistency and biometric signatures in video feeds.",
      color: "from-cyan-500 to-blue-500",
      accent: "shadow-cyan-500/10"
    },
  ];

  return (
    <section className="py-32 relative z-10 bg-[#060610]">
      <SectionWrapper className="container max-w-7xl mx-auto px-6">
        <div className="text-center mb-24">
          <h2 className="text-5xl md:text-7xl font-black mb-8 tracking-tighter text-white">
            Choose Your <span className="text-emerald-500">Theater.</span>
          </h2>
          <p className="text-white/40 text-lg md:text-xl font-medium max-w-2xl mx-auto">
            TruthGuard modules are fully integrated but specialized for every medium.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <Link key={i} href={card.href} className="block group">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: i * 0.1 }}
                  className={`h-full p-10 rounded-[2.5rem] bg-white/[0.02] border border-white/5 group-hover:bg-white/[0.04] group-hover:border-white/10 transition-all duration-500 flex flex-col items-center text-center shadow-2xl ${card.accent}`}
                >
                  <div className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${card.color} flex items-center justify-center mb-10 group-hover:scale-110 transition-all duration-500 shadow-xl shadow-black/50`}>
                    <Icon className="w-10 h-10 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold mb-4 text-white group-hover:text-emerald-400 transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-base text-white/40 mb-10 flex-grow font-medium group-hover:text-white/60 transition-colors">
                    {card.desc}
                  </p>
                  <div className="flex items-center gap-2 text-xs font-black uppercase tracking-[0.2em] text-emerald-500/60 group-hover:text-emerald-400 group-hover:gap-4 transition-all duration-500">
                    Deploy Hardware <ArrowUpRight className="w-4 h-4" />
                  </div>
                </motion.div>
              </Link>
            );
          })}
        </div>
      </SectionWrapper>
    </section>
  );
}
