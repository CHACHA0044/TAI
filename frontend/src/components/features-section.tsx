"use client";

import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { 
  Layers, 
  Globe2, 
  ShieldCheck, 
  Zap, 
  Activity, 
  Database, 
  Cpu, 
  Eye, 
  Binary 
} from "lucide-react";

export function FeaturesSection() {
  const features = [
    {
      icon: Cpu,
      title: "Neural Cross-Referencing",
      desc: "Our engine simultaneously processes metadata, visual artifacts, and semantic consistency for a 360° verification profile.",
      tag: "Advanced"
    },
    {
      icon: Eye,
      title: "Explainable Forensics",
      desc: "Full transparency. We provide heatmaps showing exactly which pixels or sentence structures triggered AI suspicion.",
      tag: "Feature"
    },
    {
      icon: Binary,
      title: "Diffusion Fingerprinting",
      desc: "Detect the specific generator (Midjourney, DALL-E 3, Stable Diffusion) by identifying unique model-specific noise signatures.",
      tag: "Patented"
    },
    {
      icon: Globe2,
      title: "Geopolitical Context",
      desc: "Native understanding of regional propaganda patterns across 50+ languages without the loss of nuance from translation.",
      tag: "Global"
    },
    {
      icon: Activity,
      title: "Asynchronous Scalability",
      desc: "Distributed task queue architecture allows for high-throughput batch processing of massive media libraries in real-time.",
      tag: "Fast"
    },
    {
      icon: Database,
      title: "Blockchain Anchoring",
      desc: "Verified content hashes can be anchored to public ledgers to ensure permanent, immutable proof of authenticity.",
      tag: "Coming Soon"
    },
  ];

  return (
    <section className="py-32 relative z-10 bg-[#080816]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_50%,rgba(16,185,129,0.05),transparent_50%)] pt-20" />
      
      <SectionWrapper className="container max-w-7xl mx-auto px-6">
        <div className="flex flex-col md:flex-row items-end justify-between mb-20 gap-8">
          <div className="max-w-2xl">
            <h2 className="text-4xl md:text-6xl font-black mb-8 tracking-tighter text-white">
              Forensic <span className="text-emerald-500">Excellence.</span>
            </h2>
            <p className="text-white/40 text-lg md:text-xl font-medium leading-relaxed">
              We leverage the intersection of deep learning and digital forensics 
              to provide an unhackable layer of trust for the digital age.
            </p>
          </div>
          <div className="hidden md:block">
            <motion.div 
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="w-32 h-32 rounded-full border border-dashed border-emerald-500/20 flex items-center justify-center"
            >
              <div className="w-24 h-24 rounded-full border border-emerald-500/10 flex items-center justify-center">
                <ShieldCheck className="w-10 h-10 text-emerald-500/30" />
              </div>
            </motion.div>
          </div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((f, i) => {
            const Icon = f.icon;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="group p-8 rounded-[2rem] bg-white/[0.02] border border-white/5 hover:border-emerald-500/30 hover:bg-emerald-500/5 transition-all duration-500"
              >
                <div className="flex items-start justify-between mb-10">
                  <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 flex items-center justify-center group-hover:scale-110 group-hover:bg-emerald-500/20 transition-all duration-500">
                    <Icon className="w-7 h-7 text-emerald-400" />
                  </div>
                  <span className="text-[10px] font-black tracking-widest uppercase text-white/20 px-3 py-1 rounded-full border border-white/5">
                    {f.tag}
                  </span>
                </div>
                <h3 className="text-xl font-bold mb-4 text-white group-hover:text-emerald-400 transition-colors">
                  {f.title}
                </h3>
                <p className="text-base text-white/40 leading-relaxed font-medium group-hover:text-white/60 transition-colors">
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
