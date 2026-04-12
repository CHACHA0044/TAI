"use client";

import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { AlertTriangle, BarChart3, Fingerprint, ShieldAlert } from "lucide-react";

export function ProblemSection() {
  const cards = [
    {
      icon: AlertTriangle,
      stat: "400%",
      title: "Synthetic Surge",
      desc: "AI-generated manipulated media availability has outpaced detection capabilities 4:1 in the last 24 months.",
      color: "border-rose-500/20 hover:border-rose-500/50 hover:bg-rose-500/5",
      iconColor: "text-rose-400"
    },
    {
      icon: Fingerprint,
      stat: "Sub-pixel",
      title: "Deceptive Precision",
      desc: "Modern diffusion models create textures so realistic they bypass traditional biometric and forensic layers.",
      color: "border-orange-500/20 hover:border-orange-500/50 hover:bg-orange-500/5",
      iconColor: "text-orange-400"
    },
    {
      icon: BarChart3,
      stat: "$78B",
      title: "Economic Impact",
      desc: "Market manipulation and corporate disinformation campaigns cost the global economy billions annually.",
      color: "border-red-500/20 hover:border-red-500/50 hover:bg-red-500/5",
      iconColor: "text-red-400"
    },
  ];

  return (
    <section className="py-32 relative z-10 overflow-hidden">
      {/* Background Distortion Effect */}
      <div className="absolute top-0 left-0 w-full h-px bg-gradient-to-r from-transparent via-rose-500/20 to-transparent" />
      
      <SectionWrapper className="container max-w-6xl mx-auto px-6">
        <div className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            whileInView={{ opacity: 1, scale: 1 }}
            className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-rose-500/10 border border-rose-500/20 text-[10px] font-black tracking-[0.3em] text-rose-500 uppercase mb-6"
          >
            <ShieldAlert className="w-3 h-3" /> Critical Vulnerability
          </motion.div>
          <h2 className="text-4xl md:text-6xl font-black mb-8 tracking-tighter text-white">
            Truth is Under <span className="text-rose-500">Attack.</span>
          </h2>
          <p className="text-white/40 text-lg md:text-xl max-w-3xl mx-auto font-medium leading-relaxed">
            The era of "believing is seeing" has ended. AI-driven misinformation 
            is now weaponized at scale, threatening democratic processes, 
            financial stability, and social cohesion.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
                className={`relative group p-10 rounded-[2.5rem] border bg-black/20 backdrop-blur-sm transition-all duration-500 ${card.color}`}
              >
                <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:opacity-10 transition-opacity">
                  <Icon className="w-24 h-24" />
                </div>
                
                <div className={`w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-500`}>
                  <Icon className={`w-7 h-7 ${card.iconColor}`} />
                </div>
                
                <h3 className="text-5xl font-black mb-3 tracking-tighter text-white">
                  {card.stat}
                </h3>
                <p className="text-xl font-bold text-white/90 mb-4">{card.title}</p>
                <p className="text-base text-white/40 leading-relaxed font-medium group-hover:text-white/60 transition-colors">
                  {card.desc}
                </p>
              </motion.div>
            );
          })}
        </div>
      </SectionWrapper>
    </section>
  );
}
