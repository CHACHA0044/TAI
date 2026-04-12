"use client";

import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { Cpu, Server, Layout } from "lucide-react";

export function ArchitectureSection() {
  const layers = [
    {
      icon: Layout,
      title: "Tactical Interface",
      tech: "Next.js 15 • Framer Motion • GSAP Ultra",
      color: "border-emerald-500/20 bg-emerald-500/5",
      glow: "from-emerald-500/40 to-transparent",
      textColor: "text-emerald-400"
    },
    {
      icon: Server,
      title: "Orchestration Kernel",
      tech: "FastAPI • Redis Queue • Worker Pools",
      color: "border-teal-500/20 bg-teal-500/5",
      glow: "from-teal-500/40 to-transparent",
      textColor: "text-teal-400"
    },
    {
      icon: Cpu,
      title: "Neural Inference Core",
      tech: "PyTorch • Transformers • Vision Layers",
      color: "border-cyan-500/20 bg-cyan-500/5",
      glow: "from-cyan-500/40 to-transparent",
      textColor: "text-cyan-400"
    }
  ];

  return (
    <section className="py-32 relative z-10 overflow-hidden bg-[#080816]">
      <SectionWrapper className="container max-w-5xl mx-auto px-6 text-center">
        <div className="mb-20">
          <h2 className="text-4xl md:text-6xl font-black mb-8 tracking-tighter text-white">
            Architecture <span className="text-emerald-500">Stack.</span>
          </h2>
          <p className="text-white/40 text-lg md:text-xl font-medium max-w-2xl mx-auto">
            TruthGuard is built on a high-concurrency, low-latency stack designed 
            to process massive datasets with millisecond-grade precision.
          </p>
        </div>

        <div className="relative max-w-3xl mx-auto flex flex-col items-center">
          {layers.map((layer, i) => (
            <div key={i} className="flex flex-col items-center w-full">
              <motion.div 
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
                className={`w-full p-8 rounded-[2rem] border backdrop-blur-md relative group hover:bg-white/5 transition-all duration-500 ${layer.color}`}
              >
                <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                  <div className="flex items-center gap-4">
                    <div className={`p-3 rounded-xl bg-white/5 ${layer.textColor}`}>
                      <layer.icon className="w-6 h-6" />
                    </div>
                    <div className="text-left">
                      <h3 className={`text-xl font-bold ${layer.textColor}`}>{layer.title}</h3>
                      <p className="text-sm text-white/40 font-medium">{layer.tech}</p>
                    </div>
                  </div>
                </div>
                
                {/* Internal Glow Effect */}
                <div className={`absolute inset-x-0 top-0 h-px bg-gradient-to-r opacity-30 ${layer.glow}`} />
              </motion.div>
              
              {i < layers.length - 1 && (
                <div className="w-px h-12 bg-gradient-to-b from-emerald-500/20 to-transparent" />
              )}
            </div>
          ))}
          
          {/* Depth Background Glow */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-[600px] bg-emerald-500/5 blur-[120px] rounded-full z-[-1]" />
        </div>
      </SectionWrapper>
    </section>
  );
}
