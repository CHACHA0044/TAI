"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { FileText, Image as ImageIcon, Video, ArrowRight } from "lucide-react";

export function CtaSection() {
  const cards = [
    {
      href: "/text",
      icon: FileText,
      title: "Analyze Text",
      desc: "Paste articles, tweets, or URLs to detect fake news.",
      gradient: "from-blue-600 to-cyan-500",
      glow: "group-hover:shadow-[0_0_40px_rgba(6,182,212,0.3)]",
    },
    {
      href: "/image",
      icon: ImageIcon,
      title: "Analyze Image",
      desc: "Upload photos to detect AI generation and manipulation.",
      gradient: "from-purple-600 to-pink-500",
      glow: "group-hover:shadow-[0_0_40px_rgba(236,72,153,0.3)]",
    },
    {
      href: "/video",
      icon: Video,
      title: "Analyze Video",
      desc: "Scan videos for deepfakes and altered audio sync.",
      gradient: "from-emerald-600 to-teal-500",
      glow: "group-hover:shadow-[0_0_40px_rgba(20,184,166,0.3)]",
    },
  ];

  return (
    <section className="py-24 relative z-10 border-t border-white/5 bg-[#080815]">
      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-6xl font-black mb-6">
            Ready to Find the <span className="text-gradient">Truth?</span>
          </h2>
          <p className="text-white/50 text-xl max-w-2xl mx-auto">
            Select a module to start analyzing content immediately.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <Link key={i} href={card.href} className="block group">
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.1 }}
                  className={`relative p-[1px] rounded-3xl overflow-hidden ${card.glow} transition-shadow duration-500`}
                >
                  <div
                    className={`absolute inset-0 bg-gradient-to-br ${card.gradient} opacity-50 group-hover:opacity-100 transition-opacity duration-300`}
                  />
                  <div className="relative h-full glass-strong bg-[#0a0a1a]/90 rounded-3xl p-8 flex flex-col items-center text-center">
                    <div className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${card.gradient} flex items-center justify-center mb-6`}>
                      <Icon className="w-10 h-10 text-white" />
                    </div>
                    <h3 className="text-2xl font-bold mb-3">{card.title}</h3>
                    <p className="text-sm text-white/60 mb-8 flex-grow">
                      {card.desc}
                    </p>
                    <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-white/50 group-hover:text-white transition-colors">
                      Launch Tool <ArrowRight className="w-4 h-4" />
                    </div>
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
