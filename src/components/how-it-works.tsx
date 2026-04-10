"use client";

import { motion, AnimatePresence } from "framer-motion";
import { SectionWrapper } from "./section-wrapper";
import { Upload, Cpu, Zap, CheckCircle } from "lucide-react";
import { useState, useEffect } from "react";

const steps = [
  {
    icon: Upload,
    title: "1. Input",
    desc: "Upload media or paste a URL/Text.",
  },
  {
    icon: Cpu,
    title: "2. AI Analysis",
    desc: "Deep-learning models process content.",
  },
  {
    icon: Zap,
    title: "3. Detection",
    desc: "AI highlights manipulated zones.",
  },
  {
    icon: CheckCircle,
    title: "4. Result",
    desc: "Receive verdict & confidence score.",
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
          // Delay circle glow until the line reaches it
          setTimeout(() => setHighlightedStep(next), 800);
        }
        return next;
      });
    }, 3000); // 3s per step gives enough time for transitions and reading
    return () => clearInterval(interval);
  }, []);

  return (
    <section id="how-it-works" className="py-24 relative z-10 overflow-hidden bg-dark-950">
      <SectionWrapper className="container max-w-5xl mx-auto px-6 text-center">
        <motion.h2 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-3xl md:text-5xl font-black mb-20"
        >
          The <span className="text-gradient">Detection Pipeline</span>
        </motion.h2>

        <div className="flex flex-col md:flex-row items-center justify-between gap-12 relative">
          {/* Progress Line (Desktop) */}
          <div className="hidden md:block absolute top-[48px] left-[12%] right-[12%] h-[2px] bg-white/5 z-0">
            <motion.div
              initial={false}
              animate={{
                width: `${(activeStep * 100) / (steps.length - 1)}%`,
              }}
              transition={{ duration: activeStep === 0 ? 0 : 0.8, ease: "easeInOut" }}
              className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 to-purple-500 shadow-[0_0_15px_rgba(59,130,246,0.6)]"
            />
          </div>

          {/* Progress Line (Mobile - Vertical) */}
          <div className="md:hidden absolute left-[48px] top-[10%] bottom-[10%] w-[2px] bg-white/5 z-0">
            <motion.div
              initial={false}
              animate={{
                height: `${(activeStep * 100) / (steps.length - 1)}%`,
              }}
              transition={{ duration: activeStep === 0 ? 0 : 0.8, ease: "easeInOut" }}
              className="absolute top-0 left-0 w-full bg-gradient-to-b from-blue-500 to-purple-500 shadow-[0_0_15px_rgba(59,130,246,0.6)]"
            />
          </div>

          {steps.map((step, i) => {
            const Icon = step.icon;
            const isHighlighted = highlightedStep === i;
            const isPassed = i <= activeStep;
            
            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="relative z-10 flex flex-col items-center group w-full md:w-1/4"
              >
                {/* Step Circle */}
                <div className="relative mb-8">
                  <motion.div
                    animate={isHighlighted ? {
                      scale: [1, 1.05, 1],
                      boxShadow: [
                        "0 0 0px rgba(59,130,246,0)",
                        "0 0 40px rgba(59,130,246,0.3)",
                        "0 0 20px rgba(59,130,246,0.15)"
                      ]
                    } : {
                      scale: 1,
                      boxShadow: "0 0 0px rgba(59,130,246,0)"
                    }}
                    transition={{ duration: 2, ease: "easeInOut", repeat: Infinity }}
                    className={`w-24 h-24 rounded-full flex items-center justify-center transition-all duration-700 border relative bg-[#05050a] ${
                      isHighlighted 
                        ? "border-blue-500/50 shadow-[0_0_30px_rgba(59,130,246,0.2)]" 
                        : isPassed 
                          ? "border-blue-500/20" 
                          : "border-white/5 opacity-50"
                    }`}
                  >
                    <Icon className={`w-10 h-10 transition-all duration-700 ${
                      isHighlighted ? "text-blue-400 scale-110 drop-shadow-[0_0_10px_rgba(59,130,246,0.8)]" : isPassed ? "text-blue-500/50" : "text-white/20"
                    }`} />
                    
                    {/* Active highlight pulse */}
                    <AnimatePresence>
                      {isHighlighted && (
                        <motion.div
                          layoutId="active-step-ring"
                          className="absolute inset-0 rounded-full border border-blue-400/50"
                          initial={{ scale: 0.8, opacity: 0 }}
                          animate={{ scale: 1.5, opacity: 0 }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
                        />
                      )}
                    </AnimatePresence>
                  </motion.div>
                  
                  {/* Step Number Badge */}
                  <div className={`absolute -top-2 -right-2 w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold border backdrop-blur-md transition-all duration-700 ${
                    isHighlighted 
                      ? "bg-blue-600 border-blue-400 text-white shadow-[0_0_15px_rgba(37,99,235,0.5)]" 
                      : isPassed
                        ? "bg-[#05050a] border-blue-500/30 text-blue-400/80"
                        : "bg-[#05050a] border-white/10 text-white/40"
                  }`}>
                    {i + 1}
                  </div>
                </div>

                <h3 className={`text-xl font-bold mb-3 transition-colors duration-700 ${
                  isHighlighted ? "text-white drop-shadow-md" : isPassed ? "text-white/80" : "text-white/40"
                }`}>
                  {step.title}
                </h3>
                <p className={`text-sm transition-colors duration-700 max-w-[200px] leading-relaxed ${
                  isHighlighted ? "text-white/80" : isPassed ? "text-white/60" : "text-white/30"
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
