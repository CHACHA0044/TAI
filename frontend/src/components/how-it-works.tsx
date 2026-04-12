"use client";

import { motion, AnimatePresence } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { Upload, Cpu, Zap, CheckCircle, Search, Activity, FileCheck } from "lucide-react";
import { useState, useEffect } from "react";

const steps = [
  {
    icon: Upload,
    title: "Intelligence Intake",
    desc: "Ingest multi-modal raw data via secure API or direct forensic upload.",
  },
  {
    icon: Search,
    title: "Feature Extraction",
    desc: "Neural layers decompose media into frame-by-frame forensic metadata.",
  },
  {
    icon: Cpu,
    title: "Neural Inference",
    desc: "Parallel models analyze pixel anomalies, and semantic consistency.",
  },
  {
    icon: FileCheck,
    title: "Forensic Verdict",
    desc: "Generate detailed score reports with explainable heatmap indicators.",
  },
];

export function HowItWorksSection() {
  const [activeStep, setActiveStep] = useState(0);
  const [highlightedStep, setHighlightedStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStep((prev) => {
        const next = (prev + 1) % steps.length;
        if (next === 0) {
          setHighlightedStep(0);
        } else {
          setTimeout(() => setHighlightedStep(next), 800);
        }
        return next;
      });
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section id="how-it-works" className="py-32 relative z-10 overflow-hidden bg-[#08081a]">
      <SectionWrapper className="container max-w-6xl mx-auto px-6 text-center">
        <div className="mb-24">
          <h2 className="text-4xl md:text-6xl font-black mb-8 tracking-tighter text-white">
            The <span className="text-emerald-500">Forensic</span> Pipeline
          </h2>
          <p className="text-white/40 text-lg md:text-xl font-medium max-w-2xl mx-auto">
            Our multi-layered inspection process ensures truth at every stage, 
            from metadata validation to deep neural analysis.
          </p>
        </div>

        <div className="flex flex-col md:flex-row items-start justify-between gap-12 relative">
          {/* Progress Line (Desktop) */}
          <div className="hidden md:block absolute top-[48px] left-[10%] right-[10%] h-[1px] bg-white/5 z-0">
            <motion.div
              initial={false}
              animate={{
                width: `${(activeStep * 100) / (steps.length - 1)}%`,
              }}
              transition={{ duration: activeStep === 0 ? 0 : 1.2, ease: "easeInOut" }}
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-emerald-500 to-teal-500 shadow-[0_0_15px_rgba(16,185,129,0.4)]"
            />
          </div>

          {steps.map((step, i) => {
            const Icon = step.icon;
            const isHighlighted = highlightedStep === i;
            const isPassed = i <= activeStep;
            
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                className="relative z-10 flex flex-col items-center text-center w-full md:w-1/4 group"
              >
                <div className="relative mb-10">
                  <motion.div
                    animate={isHighlighted ? {
                      boxShadow: [
                        "0 0 0px rgba(16,185,129,0)",
                        "0 0 40px rgba(16,185,129,0.2)",
                        "0 0 20px rgba(16,185,129,0.1)"
                      ]
                    } : {}}
                    transition={{ duration: 2, repeat: Infinity }}
                    className={`w-24 h-24 rounded-[2rem] flex items-center justify-center transition-all duration-1000 border bg-black/40 backdrop-blur-md ${
                      isHighlighted 
                        ? "border-emerald-500/50 shadow-[0_0_30px_rgba(16,185,129,0.15)]" 
                        : isPassed 
                          ? "border-emerald-500/20" 
                          : "border-white/5 opacity-50"
                    }`}
                  >
                    <Icon className={`w-9 h-9 transition-all duration-1000 ${
                      isHighlighted ? "text-emerald-400 scale-110" : isPassed ? "text-emerald-500/40" : "text-white/20"
                    }`} />
                    
                    <AnimatePresence>
                      {isHighlighted && (
                        <motion.div
                          layoutId="active-step-ring"
                          className="absolute inset-0 rounded-[2rem] border border-emerald-400/30"
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1.4, opacity: 0 }}
                          transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
                        />
                      )}
                    </AnimatePresence>
                  </motion.div>
                </div>

                <h3 className={`text-xl font-bold mb-4 transition-colors duration-1000 ${
                  isHighlighted ? "text-white" : isPassed ? "text-white/80" : "text-white/40"
                }`}>
                  {step.title}
                </h3>
                <p className={`text-sm font-medium transition-colors duration-1000 max-w-[200px] leading-relaxed ${
                  isHighlighted ? "text-white/60" : isPassed ? "text-white/40" : "text-white/20"
                }`}>
                  {step.desc}
                </p>
              </motion.div>
            );
          })}
        </div>
      </SectionWrapper>
    </section>
  );
}
