"use client";

import { motion } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { AlertCircle, TrendingUp, Users } from "lucide-react";

export function ProblemSection() {
  const cards = [
    {
      icon: TrendingUp,
      stat: "400%",
      title: "Increase in Deepfakes",
      desc: "AI-generated manipulated media has surged exponentially.",
    },
    {
      icon: Users,
      stat: "78%",
      title: "Of Users Misled",
      desc: "People struggle to distinguish AI content from reality.",
    },
    {
      icon: AlertCircle,
      stat: "$78B",
      title: "Global Cost",
      desc: "Annual estimated economic damage from misinformation.",
    },
  ];

  return (
    <section className="py-24 relative z-10">
      <SectionWrapper className="container max-w-5xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-black mb-6">
            The Misinformation <span className="text-red-400">Crisis</span>
          </h2>
          <p className="text-white/50 text-lg max-w-2xl mx-auto">
            We are losing the war on truth. Traditional fact-checking cannot
            scale to meet the volume of AI-generated fake news and media.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {cards.map((card, i) => {
            const Icon = card.icon;
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="glass p-8 rounded-3xl border border-white/5 hover:border-red-500/30 hover:bg-red-500/5 transition-all group"
              >
                <div className="w-12 h-12 rounded-2xl bg-white/5 flex items-center justify-center mb-6 group-hover:bg-red-500/20 group-hover:text-red-400 transition-colors">
                  <Icon className="w-6 h-6 text-white/40 group-hover:text-red-400 transition-colors" />
                </div>
                <h3 className="text-4xl font-black mb-4 text-white group-hover:text-red-400 transition-colors">
                  {card.stat}
                </h3>
                <p className="text-lg font-bold text-white/90 mb-2">{card.title}</p>
                <p className="text-sm text-white/50 leading-relaxed">
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
