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
      title: "Tactical Text Engine",
      desc: "Neutralize propaganda, detect bias, and verify semantic integrity.",
      color: "from-emerald-500/10 to-teal-500/5",
      iconColor: "text-emerald-400",
      href: "/text",
    },
    {
      id: "image",
      icon: ImageIcon,
      title: "Visual Forensic Lab",
      desc: "Identify GAN anomalies, diffusion traces, and manipulated pixels.",
      color: "from-teal-500/10 to-cyan-500/5",
      iconColor: "text-teal-400",
      href: "/image",
    },
    {
      id: "video",
      icon: Video,
      title: "Deepfake Radar",
      desc: "Analyze temporal consistency and biometric signatures in video.",
      color: "from-cyan-500/10 to-blue-500/5",
      iconColor: "text-cyan-400",
      href: "/video",
    },
  ];

  return (
    <section className="py-32 relative z-10 bg-[#08081a]">
      <SectionWrapper className="container max-w-7xl mx-auto px-6">
        <div className="flex flex-col md:flex-row gap-20 items-center">
          <div className="md:w-5/12">
            <h2 className="text-4xl md:text-6xl font-black mb-8 tracking-tighter leading-[1.1]">
              One Command. <br />
              <span className="text-emerald-500">Total Truth.</span>
            </h2>
            <p className="text-white/40 text-lg md:text-xl font-medium leading-relaxed mb-10">
              TruthGuard AI is the unified solution to digital deception. Our 
              proprietary models deconstruct content across every major 
              medium to ensure permanent verification.
            </p>
            <Link 
              href="/video"
              className="inline-flex items-center gap-2 text-sm font-black uppercase tracking-[0.2em] text-emerald-400 hover:text-emerald-300 transition-colors"
            >
              Explore Unified Dashboard <ArrowRight className="w-4 h-4" />
            </Link>
          </div>

          <div className="md:w-7/12 grid gap-6">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <Link 
                  key={feature.id} 
                  href={feature.href}
                  className="group block"
                >
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: i * 0.1 }}
                    className={`relative p-8 rounded-[2rem] border border-white/5 bg-gradient-to-br ${feature.color} group-hover:border-white/10 group-hover:bg-white/[0.04] transition-all duration-500 flex items-center gap-8`}
                  >
                    <div className="w-20 h-20 rounded-2xl bg-black/40 border border-white/5 flex items-center justify-center shrink-0 group-hover:scale-110 group-hover:border-emerald-500/20 transition-all duration-500">
                      <Icon className={`w-8 h-8 ${feature.iconColor}`} />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold mb-2 text-white group-hover:text-emerald-400 transition-colors">
                        {feature.title}
                      </h3>
                      <p className="text-base text-white/40 font-medium group-hover:text-white/60 transition-colors">
                        {feature.desc}
                      </p>
                    </div>
                    <div className="w-12 h-12 rounded-full border border-white/10 flex items-center justify-center group-hover:bg-emerald-500 group-hover:border-emerald-500 group-hover:text-black transition-all duration-500 shrink-0">
                      <ArrowRight className="w-5 h-5" />
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
