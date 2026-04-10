"use client";

import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { Layers, Globe2, ShieldCheck, Zap, Activity, Database } from "lucide-react";

export function FeaturesSection() {
  const features = [
    {
      icon: Layers,
      title: "Multi-Modal Synergy",
      desc: "Cross-references text context with image authenticity for comprehensive checks.",
    },
    {
      icon: Zap,
      title: "Explainable AI",
      desc: "Don't just get a score. See exactly which sentences or pixels triggered the alert.",
    },
    {
      icon: ShieldCheck,
      title: "Source Credibility",
      desc: "Real-time lookups against databases of known disinformation networks.",
    },
    {
      icon: Globe2,
      title: "Multilingual",
      desc: "Detect propaganda and bias in 50+ languages natively without translation loss.",
    },
    {
      icon: Activity,
      title: "Real-time Processing",
      desc: "Sub-second inference times for text and rapid processing for high-res media.",
    },
    {
      icon: Database,
      title: "Enterprise Dashboard",
      desc: "Track trends, manage API usage, and export detailed forensic reports. (Coming Soon)",
    },
  ];

  return (
    <section className="py-24 relative z-10 bg-[#0a0a1a]">
      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-black mb-6">
            System Features
          </h2>
          <p className="text-white/50 text-lg max-w-2xl mx-auto">
            Engineered for precision. Built for scale.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => {
            const Icon = f.icon;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.1 }}
                className="glass p-6 rounded-2xl border border-white/5 hover:-translate-y-2 hover:border-purple-500/30 hover:bg-purple-500/5 transition-all duration-300 group"
              >
                <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-purple-500/20 group-hover:text-purple-400 transition-colors">
                  <Icon className="w-6 h-6 text-white/50 group-hover:text-purple-400 transition-colors" />
                </div>
                <h3 className="text-lg font-bold mb-2 text-white/90">
                  {f.title}
                </h3>
                <p className="text-sm text-white/50 leading-relaxed group-hover:text-white/70 transition-colors">
                  {f.desc}
                </p>
              </motion.div>
            );
          })}
        </div>
      </SectionWrapper>
    </section>
  );
}
