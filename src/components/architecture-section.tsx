"use client";

import { SectionWrapper } from "./section-wrapper";

export function ArchitectureSection() {
  return (
    <section className="py-24 relative z-10 overflow-hidden">
      <SectionWrapper className="container max-w-5xl mx-auto px-6 text-center">
        <h2 className="text-3xl md:text-5xl font-black mb-16">
          System Architecture
        </h2>

        <div className="relative max-w-3xl mx-auto flex flex-col items-center gap-6">
          {/* Layer 1 */}
          <div className="w-full glass p-6 rounded-2xl border border-blue-500/30 bg-blue-500/5 relative group hover:bg-blue-500/10 transition-all">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-blue-500/50 to-transparent" />
            <h3 className="text-xl font-bold text-blue-400 mb-2">Presentation Layer</h3>
            <p className="text-sm text-white/60">Next.js App Router • Tailwind CSS • Framer Motion</p>
          </div>

          <div className="w-px h-6 bg-gradient-to-b from-blue-500/30 to-purple-500/30" />

          {/* Layer 2 */}
          <div className="w-[90%] glass p-6 rounded-2xl border border-purple-500/30 bg-purple-500/5 relative group hover:bg-purple-500/10 transition-all">
            <h3 className="text-xl font-bold text-purple-400 mb-2">API & Orchestration</h3>
            <p className="text-sm text-white/60">Node.js / Express • Request Validation • Rate Limiting</p>
          </div>

          <div className="w-px h-6 bg-gradient-to-b from-purple-500/30 to-emerald-500/30" />

          {/* Layer 3 */}
          <div className="w-[80%] glass p-6 rounded-2xl border border-emerald-500/30 bg-emerald-500/5 relative group hover:bg-emerald-500/10 transition-all">
            <h3 className="text-xl font-bold text-emerald-400 mb-2">Inference Engines</h3>
            <div className="flex justify-center gap-4 text-xs font-semibold text-white/50 mt-4">
              <span className="px-3 py-1 rounded-full bg-white/5">NLP Models</span>
              <span className="px-3 py-1 rounded-full bg-white/5">Vision Transformers</span>
              <span className="px-3 py-1 rounded-full bg-white/5">Audio Analysis</span>
            </div>
          </div>
          
          {/* Glow Behind */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full max-w-[600px] bg-purple-500/5 blur-[100px] rounded-full z-[-1]" />
        </div>
      </SectionWrapper>
    </section>
  );
}
